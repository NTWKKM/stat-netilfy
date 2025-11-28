import streamlit as st
import pandas as pd
import io

# Import function จาก stat.py
# (ตรวจสอบให้แน่ใจว่าไฟล์ stat.py อยู่ในโฟลเดอร์เดียวกัน)
from stat import process_data_and_generate_html

st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")

st.title("📊 Auto Statistical Analysis")
st.markdown("""
เครื่องมือวิเคราะห์สถิติอัตโนมัติ (Univariate & Multivariate Logistic Regression)
* **Privacy-First:** ประมวลผลบน Browser ของคุณ 100% ข้อมูลไม่ถูกส่งไป Server
* **Flexible:** รองรับ CSV/Excel และเลือกตัวแปร Outcome ได้เอง
""")

# --- 1. ส่วนจัดการข้อมูล (Data Handling) ---
st.sidebar.header("1. Data Input")

# ตัวแปร session state เพื่อเก็บข้อมูล
if 'df' not in st.session_state:
    st.session_state.df = None

# ปุ่มโหลดข้อมูลตัวอย่าง
if st.sidebar.button("📄 Load Example Data"):
    # สร้างข้อมูลจำลอง (Mockup Data)
    data = {
        'age': [55, 60, 45, 70, 80, 52, 66, 48, 75, 82] * 5,
        'sex': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0] * 5, # 1=Male, 0=Female
        'systolic_bp': [120, 140, 110, 160, 150, 130, 135, 125, 155, 145] * 5,
        'diabetes': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 5,
        'outcome_died': [0, 1, 0, 1, 1, 0, 0, 0, 1, 1] * 5  # Outcome หลัก
    }
    st.session_state.df = pd.DataFrame(data)
    st.sidebar.success("Loaded example data!")

# อัพโหลดไฟล์
uploaded_file = st.sidebar.file_uploader("Or Upload CSV/Excel", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# --- 2. ส่วนแสดงผลและตั้งค่า (Display & Settings) ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.subheader("2. Review & Edit Data")
    # ให้แก้ไขข้อมูลได้สดๆ
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    st.subheader("3. Analysis Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        # เลือก Outcome (Y)
        all_columns = edited_df.columns.tolist()
        # พยายามหา column ที่ชื่อเหมือน outcome เพื่อตั้งเป็น default
        default_idx = 0
        for i, col in enumerate(all_columns):
            if "outcome" in col.lower() or "died" in col.lower() or "status" in col.lower():
                default_idx = i
                break
                
        target_outcome = st.selectbox(
            "Select Main Outcome (Y) for Logistic Regression", 
            all_columns,
            index=default_idx,
            help="เลือกคอลัมน์ที่เป็นผลลัพธ์ (เช่น ตาย/รอด, เป็นโรค/ไม่เป็น) ค่าต้องเป็น 0 หรือ 1"
        )
    
    with col2:
        st.info(f"Selected Outcome: **{target_outcome}**")
        # เช็คว่า Outcome เป็น Binary (0/1) หรือไม่
        if edited_df[target_outcome].nunique() > 2:
            st.warning("⚠️ Warning: Selected outcome has more than 2 categories. Logistic regression might fail.")

    # --- 3. ปุ่มรัน (Action) ---
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner('Calculating stats...'):
            try:
                # ส่ง Dataframe และ Outcome ที่เลือกไปให้ stat.py
                html_result = process_data_and_generate_html(edited_df, target_outcome=target_outcome)
                
                # แสดงผล HTML
                st.components.v1.html(html_result, height=800, scrolling=True)
                
                # ปุ่มดาวน์โหลด
                st.download_button(
                    label="📥 Download Report (HTML)",
                    data=html_result,
                    file_name="analysis_report.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

else:
    st.info("👈 Please upload a file or click 'Load Example Data' to start.")
