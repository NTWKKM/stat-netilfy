import streamlit as st
import table_one # Import จาก root directory

def render(df, var_meta):
    st.subheader("1. Baseline Characteristics (Table 1)")
    ##### Baseline Characteristics (Table 1)
    st.info("""
    **💡 Guide:** Used to summarize key patient demographics and characteristics, often stratified by a primary **Grouping Variable** (e.g., Treatment Arm or Outcome Status). This table is essential for checking group comparability.

    * **Numeric Variables:** Displayed as **Mean ± Standard Deviation (SD)** for normally distributed data, or **Median (Interquartile Range - IQR)** for non-normally distributed data.
    * **Categorical Variables:** Displayed as **Count (Percentage)**.
    * **P-value (Comparison):** Shows if there is a statistically significant difference between the **Grouping Variable's** levels for each characteristic (e.g., Is the average age significantly different between Treatment Group A and Group B?).

    **Variable Selection:**
    * **Grouping Variable (Split):** The **primary categorical variable** used to divide the cohort (e.g., 'Treatment' or 'Outcome').
    * **Characteristic Variables:** All other variables (numeric and categorical) to be summarized and compared across the groups.
    """)
    
    all_cols = df.columns.tolist()
    grp_idx = 0
    for i, c in enumerate(all_cols):
        if 'group' in c.lower() or 'treat' in c.lower(): grp_idx = i; break
    
    c1, c2 = st.columns([1, 2])
    with c1:
        col_group = st.selectbox("Group By (Column):", ["None"] + all_cols, index=grp_idx+1, key='t1_group')
    with c2:
        def_vars = [c for c in all_cols if c != col_group]
        selected_vars = st.multiselect("Include Variables:", all_cols, default=def_vars, key='t1_vars')
        
    run_col, dl_col = st.columns([1, 1])
    
    if 'html_output_t1' not in st.session_state:
        st.session_state.html_output_t1 = None

    if run_col.button("📊 Generate Table 1", type="primary"):
        with st.spinner("Generating..."):
            try:
                grp = None if col_group == "None" else col_group
                html_t1 = table_one.generate_table(df, selected_vars, grp, var_meta)
                st.session_state.html_output_t1 = html_t1 
                st.components.v1.html(html_t1, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.html_output_t1 = None 
                
    with dl_col:
        if st.session_state.html_output_t1:
            st.download_button("📥 Download HTML", st.session_state.html_output_t1, "table1.html", "text/html", key='dl_btn_t1')
        else:
            st.button("📥 Download HTML", disabled=True, key='ph_t1')
