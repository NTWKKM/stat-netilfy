import streamlit as st
import table_one # Import จาก root directory

def render(df, var_meta):
    st.subheader("1. Baseline Characteristics (Table 1)")
    st.info("""
    **💡 Guide:** Summarizes key demographics and patient characteristics, stratified by a **Grouping Variable**, to assess **group comparability**.

    **Presentation:**
    * **Numeric:** Mean ± SD (Normally Distributed Data) or Median (IQR) (**Non**-Normally Distributed Data).
    * **Categorical:** Count (Percentage).
    * **P-value & Test Used:** Tests for statistically significant differences in characteristics across groups.
    **Automatically selects the appropriate test** (e.g., t-test, Chi-square, Kruskal-Wallis) based on the variable type and distribution.

    **Variable Selection:**
    * **Grouping Variable (Split):** The primary categorical variable used to stratify the **dataset** (e.g., 'Treatment' or 'Outcome').
    * **Characteristics:** All other variables (numeric/categorical) to be summarized and compared.
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
                # Calling the generate_table function in table_one.py
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
