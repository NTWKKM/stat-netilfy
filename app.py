import streamlit as st
import pandas as pd
import numpy as np
# ⚠️ ตรวจสอบชื่อไฟล์ให้ตรงกับไฟล์ logic ของคุณ
from logic import process_data_and_generate_html 

st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")

st.title("📊 Auto Statistical Analysis")

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_meta' not in st.session_state:
    st.session_state.var_meta = {} 

# --- Helper Function: Check Separation ---
def check_perfect_separation(df, target_col):
    risky_vars = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: return []
    except: return []

    for col in df.columns:
        if col == target_col: continue
        if df[col].nunique() < 10: 
            try:
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except: pass
    return risky_vars

# --- Sidebar: Data Input ---
st.sidebar.header("1. Data Input")

if st.sidebar.button("📄 Load Example Data"):
    # ข้อมูลตัวอย่าง (เหมือนเดิม)
    data = {
        'age': [
            25, 28, 30, 35, 40, 42, 45, 22, 29, 33, 
            50, 55, 52, 58, 60, 62, 51, 59, 54, 57, 
            65, 70, 72, 75, 80, 82, 85, 78, 88, 90, 
            26, 31, 38, 41, 44, 46, 53, 56, 61, 63, 
            66, 71, 73, 76, 81, 83, 86, 79, 89, 21,
            34, 49, 64, 74, 84, 27, 39, 69, 77, 87
        ],
        'sex': [0, 1] * 30,
        'shock_state': [
            0,0,0,0,0, 0,0,0,0,0, 
            0,1,0,1,0, 0,1,0,1,0, 
            1,1,0,1,1, 1,1,0,1,1, 
            0,0,0,0,0, 0,1,0,1,0,
            1,1,1,0,1, 1,0,1,0,0,
            0,0,1,1,1, 0,0,1,1,1
        ],
        'outcome_died': [
            0,0,0,0,0, 0,0,0,0,1, 
            0,1,0,0,0, 0,1,0,0,0, 
            1,1,0,1,1, 1,1,0,1,1, 
            0,0,0,0,0, 0,0,0,0,1,
            1,1,0,0,1, 1,0,1,0,0,
            0,0,0,1,1, 0,0,1,1,1
        ]
    }
    st.session_state.df = pd.DataFrame(data)
    st.session_state.var_meta = {
        'sex': {'type': 'Categorical', 'map': {0:'Female', 1:'Male'}},
        'shock_state': {'type': 'Categorical', 'map': {0:'No', 1:'Yes'}},
        'outcome_died': {'type': 'Categorical', 'map': {0:'Survived', 1:'Died'}}
    }
    st.sidebar.success("Loaded clean example data!")

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
    
    # --- Sidebar: Variable Settings ---
    st.sidebar.header("2. Variable Settings")
    all_cols = df.columns.tolist()
    selected_col = st.sidebar.selectbox("Select Variable to Edit:", all_cols)
    
    if selected_col:
        current_meta = st.session_state.var_meta.get(selected_col, {})
        current_type = current_meta.get('type', 'Auto-detect')
        col_type = st.sidebar.radio(f"Type for '{selected_col}':", ['Auto-detect', 'Categorical', 'Continuous'], index=['Auto-detect', 'Categorical', 'Continuous'].index(current_type))
        map_str = "\n".join([f"{k}={v}" for k, v in current_meta.get('map', {}).items()])
        user_labels = st.sidebar.text_area("Define Labels:", value=map_str, height=100)
        
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
                            else: k_key = k_clean
                            new_map[k_key] = v.strip()
                        except: pass
            if selected_col not in st.session_state.var_meta: st.session_state.var_meta[selected_col] = {}
            st.session_state.var_meta[selected_col]['type'] = col_type
            st.session_state.var_meta[selected_col]['map'] = new_map
            st.sidebar.success(f"Saved!")
            if hasattr(st, "rerun"): st.rerun()
            else: st.experimental_rerun()

    # --- 🟢 DATA PREVIEW (แก้ไขส่วนนี้) ---
    st.subheader("2. Review & Edit Data")
    st.info("💡 You can edit data directly in this table. Scroll to view all rows.")
    
    # ใช้ data_editor แทน dataframe เพื่อให้เลื่อนดูได้ครบ และแก้ค่าได้ด้วย
    # num_rows="dynamic" อนุญาตให้เพิ่ม/ลบแถวได้
    edited_df = st.data_editor(
        df, 
        num_rows="dynamic", 
        use_container_width=True,
        height=400 # กำหนดความสูงเริ่มต้น (เลื่อน scrollbar ได้ถ้าข้อมูลเยอะ)
    )
    
    # อัปเดต df ใน session state เมื่อมีการแก้ไข
    # (Streamlit จะจัดการให้อัตโนมัติ แต่เราใช้ตัวแปร edited_df ต่อไปคำนวณ)

    # --- Analysis Execution ---
    st.subheader("3. Run Analysis")
    
    default_idx = 0
    for i, c in enumerate(all_cols):
        if any(x in c.lower() for x in ['outcome', 'died', 'sumoutcome']):
            default_idx = i
            break
    target_outcome = st.selectbox("Select Main Outcome (Y)", all_cols, index=default_idx)
    
    # Check Separation
    # ใช้ edited_df (ข้อมูลที่อาจถูกแก้ไขแล้ว) ในการตรวจสอบ
    risky_vars = check_perfect_separation(edited_df, target_outcome)
    
    exclude_cols = []
    if risky_vars:
        st.warning(f"⚠️ **Perfect Separation Risk Detected!**")
        st.markdown(f"ตัวแปรเหล่านี้อาจทำให้การคำนวณผิดพลาด: {', '.join(risky_vars)}")
        exclude_cols = st.multiselect("Select variables to EXCLUDE:", options=all_cols, default=risky_vars)
    else:
        exclude_cols = st.multiselect("Select variables to EXCLUDE (Optional):", options=all_cols)

    if st.button("🚀 Run Analysis", type="primary"):
        if edited_df[target_outcome].nunique() < 2:
            st.error("Outcome must have at least 2 values (e.g. 0, 1)")
        else:
            with st.spinner("Processing..."):
                try:
                    final_df = edited_df.drop(columns=exclude_cols, errors='ignore')
                    # ส่ง edited_df ที่แก้ไขแล้วไปคำนวณ
                    html = process_data_and_generate_html(final_df, target_outcome, var_meta=st.session_state.var_meta)
                    st.components.v1.html(html, height=800, scrolling=True)
                    st.download_button("📥 Download Report", html, "report.html", "text/html")
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👈 Please upload a file to start.")
