import streamlit as st
import pandas as pd
import numpy as np
# ⚠️ ตรวจสอบว่าไฟล์ logic.py มีอยู่จริงและชื่อถูกต้อง
from logic import process_data_and_generate_html 

st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")

st.title("📊 Auto Statistical Analysis")

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {} 

# --- Sidebar: Data Input ---
st.sidebar.header("1. Data Input")

# Load Example
if st.sidebar.button("📄 Load Example Data"):
    data = {
        'age': [55, 60, 45, 70, 80, 52, 66, 48, 75, 82] * 5,
        'sex': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0] * 5,
        'hypertension': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 5,
        'rv_dysfunction': [0, 1, 2, 3, 0, 1, 0, 2, 1, 0] * 5, 
        'outcome_died': [0, 1, 0, 1, 1, 0, 0, 0, 1, 1] * 5
    }
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {} 
    st.sidebar.success("Loaded example data!")

# Upload File
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# --- Main Logic ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    # --- Sidebar: Variable Settings (Coding) ---
    st.sidebar.header("2. Variable Settings (Optional)")
    
    all_cols = df.columns.tolist()
    selected_col = st.sidebar.selectbox("Select Variable to Edit:", all_cols)
    
    if selected_col:
        # 1. Type Override
        current_meta = st.session_state.var_meta.get(selected_col, {})
        current_type = current_meta.get('type', 'Auto-detect')
        
        col_type = st.sidebar.radio(
            f"Type for '{selected_col}':", 
            ['Auto-detect', 'Categorical', 'Continuous'],
            index=['Auto-detect', 'Categorical', 'Continuous'].index(current_type)
        )
        
        # 2. Value Labels (Coding)
        st.sidebar.markdown("**Value Labels (Coding):**")
        st.sidebar.caption("Example: 0=No, 1=Yes")
        
        current_map = current_meta.get('map', {})
        map_str = "\n".join([f"{k}={v}" for k, v in current_map.items()])
        
        user_labels = st.sidebar.text_area("Define Labels:", value=map_str, height=100)
        
        # 3. Save Button
        if st.sidebar.button("💾 Save Settings"):
            new_map = {}
            if user_labels.strip():
                for line in user_labels.split('\n'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        try:
                            k_clean = k.strip()
                            if k_clean.replace('.','',1).isdigit():
                                if '.' in k_clean: k_key = float(k_clean)
                                else: k_key = int(k_clean)
                            else:
                                k_key = k_clean
                            new_map[k_key] = v.strip()
                        except: pass
            
            if selected_col not in st.session_state.var_meta:
                st.session_state.var_meta[selected_col] = {}
            
            st.session_state.var_meta[selected_col]['type'] = col_type
            st.session_state.var_meta[selected_col]['map'] = new_map
            
            st.sidebar.success(f"Saved settings for {selected_col}")
            
            # --- FIX: Rerun compatible code ---
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    # --- Preview Data ---
    st.subheader("Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # --- Analysis Execution ---
    st.subheader("3. Run Analysis")
    
    default_idx = 0
    for i, c in enumerate(all_cols):
        if 'outcome' in c.lower() or 'died' in c.lower() or 'sumoutcome' in c.lower():
            default_idx = i
            break
            
    target_outcome = st.selectbox("Select Main Outcome (Y)", all_cols, index=default_idx)
    
    if st.button("🚀 Run Analysis", type="primary"):
        if df[target_outcome].nunique() < 2:
            st.error("Outcome must have at least 2 values (e.g. 0, 1)")
        else:
            with st.spinner("Processing..."):
                try:
                    html = process_data_and_generate_html(df, target_outcome, var_meta=st.session_state.var_meta)
                    st.components.v1.html(html, height=800, scrolling=True)
                    st.download_button("📥 Download Report", html, "report.html", "text/html")
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👈 Please upload a file to start.")
