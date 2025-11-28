import streamlit as st
import pandas as pd
import numpy as np

# Import logic
from logic import process_data_and_generate_html

# ฟังก์ชันสำหรับตรวจสอบคุณภาพข้อมูล (Duplicate Logic with stat.py for checking purpose)
def is_problematic(val):
    """เช็คว่าค่านี้นำไปคำนวณได้ไหม"""
    if pd.isna(val) or val == "":
        return False # ค่าว่างไม่ใช่ปัญหา (แค่ Missing)
    
    # ลอง Clean แบบเดียวกับ Backend
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    
    try:
        float(s)
        return False # แปลงได้ = รอด
    except:
        return True # แปลงไม่ได้ = ปัญหา (เช่น '87(baseline)')

st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")

st.title("📊 Auto Statistical Analysis")
st.markdown("""
**Privacy-First Statistical Tool** (Run locally in your browser)
""")

# --- 1. Data Input ---
st.sidebar.header("1. Data Input")

if 'df' not in st.session_state:
    st.session_state.df = None

# ปุ่ม Load Example
if st.sidebar.button("📄 Load Example Data"):
    data = {
        'age': [55, 60, 45, '87(baseline)', 80], # มีค่า Error
        'sex': [1, 0, 1, 0, 1],
        'outcome_died': [0, 1, 0, 1, 1] 
    }
    st.session_state.df = pd.DataFrame(data)
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

# --- 2. Review & Data Cleaning Check ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.subheader("2. Review & Fix Data")
    
    # --- 🔍 AUTO-DETECT PROBLEMS ---
    problems = []
    # วนลูปเช็คทุกช่อง (อาจช้าหน่อยถ้าไฟล์ใหญ่มาก แต่ปลอดภัย)
    # เช็คเฉพาะคอลัมน์ที่เป็น Object (String) เพราะถ้าเป็น Int/Float อยู่แล้วแปลว่าปลอดภัย
    cols_to_check = df.select_dtypes(include=['object']).columns
    
    for col in cols_to_check:
        for idx, val in df[col].items():
            if is_problematic(val):
                problems.append({
                    "Row Index": idx,
                    "Column": col,
                    "Invalid Value": val,
                    "Suggestion": "Please remove text (keep only numbers)"
                })
    
    # ถ้าเจอปัญหา แสดงตือนก่อนตาราง
    if problems:
        problem_df = pd.DataFrame(problems)
        st.error(f"⚠️ Found {len(problems)} values that cannot be calculated!")
        st.markdown("ค่าเหล่านี้จะถูกมองเป็น **ว่าง (Missing)** หากไม่แก้ไข (เครื่องหมาย >,< ใช้ได้ไม่ต้องแก้)")
        
        # แสดงรายการที่ผิด
        st.dataframe(problem_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Data looks clean! (Standard symbols >, <, , are accepted)")

    # Data Editor (แก้ไขค่าผิดได้ตรงนี้เลย)
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    # --- 3. Analysis ---
    st.subheader("3. Analysis Settings")
    
    # หา Outcome
    all_columns = edited_df.columns.tolist()
    default_idx = 0
    for i, col in enumerate(all_columns):
        if any(x in col.lower() for x in ["outcome", "died", "status", "sumoutcome"]):
            default_idx = i
            break
            
    target_outcome = st.selectbox("Select Outcome (Y)", all_columns, index=default_idx)

    if st.button("🚀 Run Analysis", type="primary"):
        # เช็คอีกทีว่า Outcome มีค่าพอไหม
        if edited_df[target_outcome].nunique() < 2:
            st.error("❌ Outcome must have at least 2 groups (e.g., 0 and 1).")
        else:
            with st.spinner('Calculating...'):
                try:
                    html_result = process_data_and_generate_html(edited_df, target_outcome=target_outcome)
                    st.components.v1.html(html_result, height=800, scrolling=True)
                    st.download_button("📥 Download HTML Report", html_result, "report.html", "text/html")
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👈 Please upload a file to start.")
