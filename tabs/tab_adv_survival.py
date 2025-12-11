import streamlit as st
import pandas as pd
import survival_lib
import matplotlib.pyplot as plt

# 🟢 FIX 1: ต้องรับ var_meta เพื่อให้สอดคล้องกับ app.py และโครงสร้างโปรเจกต์
def render(df, var_meta): 
    st.subheader("⏳ Advanced Survival Analysis")
    st.info("""
    **Modules:**
    * **Time-Dependent Cox:** For variables that change over time (Requires Long-Format Data: Start-Stop).
    """)

    # 🟢 FIX 2: กำหนดค่า all_cols ที่หายไป
    all_cols = df.columns.tolist() 

    # ==========================
    # 2. Time-Dependent Cox
    # ==========================
    st.warning("⚠️ **Requirement:** Data must be in **Long Format** (Start-Stop rows).")
        
    c1, c2, c3, c4 = st.columns(4)
    id_col = c1.selectbox("🆔 ID Column:", all_cols, key='td_id')
    
    # เพื่อความง่ายในการเลือก: ใช้ all_cols ใน selectbox แรก, และใช้ logic กรองใน list comprehensions
    start_col = c2.selectbox("▶️ Start Time:", [c for c in all_cols if c != id_col], key='td_start')
    stop_col = c3.selectbox("⏹️ Stop Time:", [c for c in all_cols if c not in [id_col, start_col]], key='td_stop')
    event_col = c4.selectbox("💀 Event (at Stop):", [c for c in all_cols if c not in [id_col, start_col, stop_col]], key='td_event')
        
    covs = st.multiselect("Select Time-Dependent Covariates:", 
                             [c for c in all_cols if c not in [id_col, start_col, stop_col, event_col]], 
                             key='td_covs')
        
    if st.button("Run Time-Dependent Model", key='btn_td'):
        if not covs:
            st.error("Select covariates first.")
        else:
            # Validate that required columns are numeric
            numeric_cols = [start_col, stop_col, event_col] + covs
            non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                st.error(f"The following columns must be numeric: {', '.join(non_numeric)}")
                return

            with st.spinner("Fitting Model..."):
                ctv, res, err = survival_lib.fit_cox_time_varying(df, id_col, event_col, start_col, stop_col, covs)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Model Converged!")
                    st.dataframe(res.style.format("{:.4f}"))
                    st.caption("Interpretation: HR > 1 indicates increased risk.")
