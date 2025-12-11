import streamlit as st
import pandas as pd
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

    # 1. Select Variables
    c1, c2 = st.columns(2)
    # Auto-detect Treatment (Binary)
    treat_idx = 0
    for i, c in enumerate(all_cols):
        if df[c].nunique() == 2 and ('group' in c.lower() or 'treat' in c.lower()):
            treat_idx = i; break
            
    treat_col = c1.selectbox("💊 Treatment Variable (Binary 0/1):", all_cols, index=treat_idx, key='psm_treat')
    outcome_col = c2.selectbox("🎯 Outcome Variable (Optional):", ["None"] + all_cols, key='psm_outcome')
    
    # Select Covariates (Excluding treat/outcome)
    cov_candidates = [c for c in all_cols if c not in [treat_col, outcome_col]]
    cov_cols = st.multiselect("📊 Covariates (Confounders):", cov_candidates, 
                              default=[c for c in cov_candidates if 'age' in c.lower() or 'sex' in c.lower() or 'score' in c.lower()],
                              key='psm_cov')

    # PSM Settings
    with st.expander("⚙️ Advanced Settings"):
        caliper = st.slider("Caliper Width (SD of Logit):", 0.05, 1.0, 0.2, 0.05, help="Tighter caliper = Better balance but fewer matches.")
    
    # Run Button
    if st.button("🚀 Run Matching", key='btn_psm'):
        if not cov_cols:
            st.error("Please select at least one covariate.")
        else:
            try:
                # A. Calculate PS
                with st.spinner("Calculating Propensity Scores..."):
                    df_ps, model = psm_lib.calculate_ps(df, treat_col, cov_cols)
                
                # B. Perform Matching
                with st.spinner("Matching Patients..."):
                    df_matched, msg = psm_lib.perform_matching(df_ps, treat_col, 'ps_logit', caliper)
                
                if df_matched is None:
                    st.error(msg)
                else:
                    st.success(f"Matching Complete! {msg}")
                    
                    # C. Check Balance (SMD)
                    smd_pre = psm_lib.calculate_smd(df_ps, treat_col, cov_cols)
                    smd_post = psm_lib.calculate_smd(df_matched, treat_col, cov_cols)
                    
                    # Tabs for results
                    t_res1, t_res2, t_res3 = st.tabs(["📊 Balance Check (Love Plot)", "📋 Matched Data", "📉 Outcome Analysis"])
                    
                    with t_res1:
                        c_plot, c_tab = st.columns([2, 1])
                        fig_love = psm_lib.plot_love_plot(smd_pre, smd_post)
                        c_plot.pyplot(fig_love)
                        
                        c_tab.markdown("**SMD Table:**")
                        smd_merge = pd.merge(smd_pre, smd_post, on='Variable', suffixes=('_Pre', '_Post'))
                        c_tab.dataframe(smd_merge.style.format("{:.4f}").background_gradient(cmap='Reds', subset=['SMD_Post']))
                        c_tab.caption("*SMD < 0.1 indicates good balance.*")

                    with t_res2:
                        st.write(f"Matched Dataset ({len(df_matched)} rows):")
                        st.dataframe(df_matched.head(50))
                        
                        # Download
                        csv = df_matched.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Matched CSV", csv, "matched_data.csv", "text/csv")

                    with t_res3:
                        if outcome_col != "None":
                            st.markdown(f"##### Simple Comparison: {outcome_col}")
                            # Simple stats
                            res_stats = df_matched.groupby(treat_col)[outcome_col].mean()
                            st.write(res_stats)
                            
                            st.info("💡 Note: For rigorous analysis, take this 'Matched Dataset' and run it in the 'Hypothesis Testing' or 'Survival' tab.")
                        else:
                            st.write("Select an Outcome variable to see a quick comparison.")
                            
                    # Generate Report
                    elements = [
                        {'type':'text', 'data': f"PSM Matching Report: {treat_col} (Covariates: {', '.join(cov_cols)})"},
                        {'type':'table', 'data': smd_merge},
                        {'type':'plot', 'data': fig_love}
                    ]
                    html_rep = psm_lib.generate_psm_report("PSM Analysis", elements)
                    st.download_button("📥 Download Report HTML", html_rep, "psm_report.html", "text/html")
                    
            except Exception as e:
                st.error(f"Error during PSM: {e}")
