import streamlit as st
import pandas as pd
import survival_lib
import matplotlib.pyplot as plt

def render(df):
    st.subheader("⏳ Advanced Survival Analysis")
    st.info("""
    **Modules:**
    1. **Landmark Analysis:** For eliminating *Immortal Time Bias* by analyzing only patients who survived up to a specific time.
    2. **Time-Dependent Cox:** For variables that change over time (Requires Long-Format Data: Start-Stop).
    """)

    all_cols = df.columns.tolist()
    
    # เมนูเลือกโหมด
    mode = st.radio("Select Method:", ["📍 Landmark Analysis", "⏱️ Time-Dependent Cox Regression"], horizontal=True)
    st.markdown("---")

    # ==========================
    # 1. Landmark Analysis
    # ==========================
    if mode == "📍 Landmark Analysis":
        c1, c2 = st.columns(2)
        # Auto-detect
        time_idx = next((i for i, c in enumerate(all_cols) if 'time' in c.lower() or 'dur' in c.lower()), 0)
        event_idx = next((i for i, c in enumerate(all_cols) if 'event' in c.lower() or 'dead' in c.lower()), min(1, len(all_cols)-1))

        col_time = c1.selectbox("⏳ Time Variable:", all_cols, index=time_idx, key='lm_time')
        col_event = c2.selectbox("💀 Event Variable (1=Event):", [c for c in all_cols if c != col_time], key='lm_event')
        
        # Landmark Slider
        max_t = df[col_time].dropna().max() if not df.empty and pd.api.types.is_numeric_dtype(df[col_time]) and df[col_time].notna().any() else 100.0
        landmark_t = st.slider(f"Select Landmark Time ({col_time}):", 0.0, float(max_t), float(max_t) * 0.1, key='lm_slider')
        
        col_group = st.selectbox("Compare Group (Optional):", ["None"] + all_cols, key='lm_group')

        if st.button("Run Landmark Analysis", key='btn_lm'):
            # 🟢 Filter Data (หัวใจของ Landmark)
            mask = df[col_time] >= landmark_t
            df_lm = df[mask].copy()
            
            n_excl = len(df) - len(df_lm)
            st.success(f"**Included:** {len(df_lm)} patients. (**Excluded:** {n_excl} early events/censored)")
            
            if len(df_lm) < 5:
                st.error("Sample size too small after filtering.")
            else:
                grp = None if col_group == "None" else col_group
                fig, stats = survival_lib.fit_km_logrank(df_lm, col_time, col_event, grp)
                
                # วาดเส้น Landmark ลงไปในกราฟ
                ax = fig.gca()
                ax.axvline(landmark_t, color='red', linestyle='--', label=f'Landmark t={landmark_t}')
                ax.legend()
                ax.set_title(f"Landmark Analysis (Survival given t >= {landmark_t})")
                
                st.pyplot(fig)
                st.dataframe(stats)

    # ==========================
    # 2. Time-Dependent Cox
    # ==========================
    elif mode == "⏱️ Time-Dependent Cox Regression":
        st.warning("⚠️ **Requirement:** Data must be in **Long Format** (Start-Stop rows).")
        
        c1, c2, c3, c4 = st.columns(4)
        id_col = c1.selectbox("🆔 ID Column:", all_cols, key='td_id')
        start_col = c2.selectbox("▶️ Start Time:", all_cols, key='td_start')
        stop_col = c3.selectbox("⏹️ Stop Time:", all_cols, index=min(1, len(all_cols)-1), key='td_stop')
        event_col = c4.selectbox("💀 Event (at Stop):", all_cols, index=min(2, len(all_cols)-1), key='td_event')
        
        covs = st.multiselect("Select Time-Dependent Covariates:", 
                              [c for c in all_cols if c not in [id_col, start_col, stop_col, event_col]], 
                              key='td_covs')
        
        if st.button("Run Time-Dependent Model", key='btn_td'):
            if not covs:
                st.error("Select covariates first.")
            else:
                with st.spinner("Fitting Model..."):
                    ctv, res, err = survival_lib.fit_cox_time_varying(df, id_col, event_col, start_col, stop_col, covs)
                    if err:
                        st.error(f"Error: {err}")
                    else:
                        st.success("Model Converged!")
                        st.dataframe(res.style.format("{:.4f}"))
                        st.caption("Interpretation: HR > 1 indicates increased risk.")
