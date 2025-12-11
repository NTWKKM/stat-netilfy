import streamlit as st
import pandas as pd
import numpy as np
import psm_lib
import matplotlib.pyplot as plt

def render(df, var_meta):
    st.subheader("⚖️ Propensity Score Matching (PSM)")
    st.info("""
    **💡 Concept:** Matches patients in the Treatment group with similar patients in the Control group using **Propensity Scores**.
    * **Goal:** To reduce selection bias and mimic a Randomized Controlled Trial (RCT).
    * **Algorithm:** Nearest Neighbor 1:1 Matching (Greedy, Without Replacement).
    """)

    all_cols = df.columns.tolist()

    # --- 1. Variable Selection ---
    c1, c2 = st.columns(2)
    
    # Auto-detect Treatment (Look for binary variables)
    treat_idx = 0
    for i, c in enumerate(all_cols):
        if df[c].nunique() == 2 and ('group' in c.lower() or 'treat' in c.lower()):
            treat_idx = i; break
            
    treat_col = c1.selectbox("💊 Treatment Variable (Binary):", all_cols, index=treat_idx, key='psm_treat')
    outcome_col = c2.selectbox("🎯 Outcome Variable (Optional):", ["None"] + all_cols, key='psm_outcome')
    
    # Covariates Selection
    cov_candidates = [c for c in all_cols if c not in [treat_col, outcome_col]]
    default_covs = [c for c in cov_candidates if any(x in c.lower() for x in ['age', 'sex', 'bmi', 'score', 'hyper'])]
    cov_cols = st.multiselect("📊 Covariates (Confounders):", cov_candidates, 
                              default=default_covs,
                              key='psm_cov')

    # --- 2. Data Preparation (Fixing the Error) ---
    # ตรวจสอบว่า Treatment เป็น 0/1 หรือไม่ ถ้าไม่ใช่ต้องแปลง
    df_analysis = df.copy()
    unique_treat = df_analysis[treat_col].dropna().unique()
    
    # ตรวจสอบว่ามีแค่ 2 กลุ่มจริงไหม
    if len(unique_treat) != 2:
        st.warning(f"⚠️ Variable '{treat_col}' must have exactly 2 unique values (Found: {unique_treat}). PSM cannot run.")
        return

    # เช็คว่าเป็น 0/1 อยู่แล้วหรือไม่
    is_numeric_binary = set(unique_treat).issubset({0, 1}) or set(unique_treat).issubset({0.0, 1.0})
    
    target_val = None
    final_treat_col = treat_col

    if not is_numeric_binary:
        st.markdown("---")
        st.warning(f"⚠️ variable '{treat_col}' is text/categorical. Please specify which value is the **Treatment Group**.")
        c_map, _ = st.columns(2)
        target_val = c_map.selectbox(f"Select value for 'Treatment/Case' (will be mapped to 1):", unique_treat, key='psm_target_select')
        
        # สร้างคอลัมน์ใหม่ที่เป็น 0/1 เพื่อใช้คำนวณ
        final_treat_col = f"{treat_col}_encoded"
        df_analysis[final_treat_col] = np.where(df_analysis[treat_col] == target_val, 1, 0)
        st.success(f"Mapped: '{target_val}' = 1, Others = 0")
    
    # จัดการ Covariates ที่เป็น Text (One-hot Encoding)
    # ถ้าตัวแปรไหนเป็น Text ระบบจะแปลงเป็นตัวเลขให้ (เช่น Sex: Male/Female -> Sex_Male: 0/1)
    if cov_cols:
        df_analysis = pd.get_dummies(df_analysis, columns=[c for c in cov_cols if df_analysis[c].dtype == 'object'], drop_first=True)
        # อัปเดตรายชื่อ cov_cols หลังแปลง (เอาชื่อคอลัมน์ใหม่มาใช้)
        # (หมายเหตุ: วิธีนี้ง่ายสุดสำหรับ sklearn)
        # แต่เพื่อความง่ายใน library เราจะใช้เฉพาะคอลัมน์ที่เป็นตัวเลขแล้ว
        final_cov_cols = [c for c in df_analysis.columns if c not in [treat_col, outcome_col, final_treat_col] and c in df_analysis.columns]
        # กรองเอาเฉพาะที่เกี่ยวกับ cov เดิม (ป้องกันตัวแปรอื่นปนมา)
        # ง่ายที่สุดคือใช้ pd.get_dummies เฉพาะ X
        X_encoded = pd.get_dummies(df[cov_cols], drop_first=True)
        final_cov_cols = X_encoded.columns.tolist()
        # รวม X_encoded กลับเข้าไปใน df_analysis
        df_analysis = pd.concat([df_analysis, X_encoded], axis=1)
        # ลบคอลัมน์ซ้ำ (ถ้ามี)
        df_analysis = df_analysis.loc[:, ~df_analysis.columns.duplicated()]

    # PSM Settings
    with st.expander("⚙️ Advanced Settings"):
        caliper = st.slider("Caliper Width (SD of Logit):", 0.05, 1.0, 0.2, 0.05)
    
    # --- 3. Run Analysis ---
    if st.button("🚀 Run Matching", key='btn_psm'):
        if not cov_cols:
            st.error("Please select at least one covariate.")
        else:
            try:
                # A. Calculate PS
                with st.spinner("Calculating Propensity Scores..."):
                    # ส่ง df_analysis ที่แปลงเป็นตัวเลขแล้วเข้าไป
                    df_ps, model = psm_lib.calculate_ps(df_analysis, final_treat_col, final_cov_cols)
                
                # B. Perform Matching
                with st.spinner("Matching Patients..."):
                    df_matched, msg = psm_lib.perform_matching(df_ps, final_treat_col, 'ps_logit', caliper)
                
                if df_matched is None:
                    st.error(msg)
                else:
                    st.success(f"Matching Complete! {msg}")
                    
                    # C. Check Balance (SMD)
                    smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
                    smd_post = psm_lib.calculate_smd(df_matched, final_treat_col, final_cov_cols)
                    
                    # Tabs for results
                    t_res1, t_res2, t_res3 = st.tabs(["📊 Balance Check (Love Plot)", "📋 Matched Data", "📉 Outcome Analysis"])
                    
                    with t_res1:
                        c_plot, c_tab = st.columns([2, 1])
                        fig_love = psm_lib.plot_love_plot(smd_pre, smd_post)
                        c_plot.pyplot(fig_love)
                        
                        c_tab.markdown("**SMD Table:**")
                        smd_merge = pd.merge(smd_pre, smd_post, on='Variable', suffixes=('_Pre', '_Post'))
                        c_tab.dataframe(smd_merge.style.format("{:.4f}").background_gradient(cmap='Reds', subset=['SMD_Post']))
                        
                    with t_res2:
                        st.write(f"Matched Dataset ({len(df_matched)} rows):")
                        # โชว์ข้อมูลดิบ (ดึงข้อมูลเดิมกลับมาแสดงคู่กับ ID)
                        # แต่ใน df_matched ตอนนี้เป็นข้อมูลที่ encode แล้ว
                        # ถ้าอยากโชว์ข้อมูลเดิม อาจจะต้อง map index กลับ (ขั้นสูง)
                        # เบื้องต้นโชว์ df_matched ไปก่อน
                        st.dataframe(df_matched.head(50))
                        
                        csv = df_matched.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Matched CSV", csv, "matched_data.csv", "text/csv")

                    with t_res3:
                        if outcome_col != "None":
                            st.markdown(f"##### Outcome Comparison: {outcome_col}")
                            # ใช้ final_treat_col (0/1) ในการ group
                            res_stats = df_matched.groupby(final_treat_col)[outcome_col].mean()
                            st.write(res_stats)
                            
                            st.info("💡 Note: This is a simple comparison. For statistical significance, use the Hypothesis Testing tab with the 'Matched Dataset'.")
                        else:
                            st.write("Select an Outcome variable to see comparison.")
                            
                    # Generate Report
                    elements = [
                        {'type':'text', 'data': f"PSM Matching Report. Treatment: {treat_col} (Mapped 1={target_val if target_val else '1'})"},
                        {'type':'table', 'data': smd_merge},
                        {'type':'plot', 'data': fig_love}
                    ]
                    html_rep = psm_lib.generate_psm_report("PSM Analysis", elements)
                    st.download_button("📥 Download Report HTML", html_rep, "psm_report.html", "text/html")
                    
            except Exception as e:
                st.error(f"Error during PSM: {e}")
