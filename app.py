# app.py
import streamlit as st
import pandas as pd
# import function จากไฟล์ stat ของคุณ
# (สมมติว่าคุณแก้ stat.py ให้มีฟังก์ชัน process_data แล้ว)
# from stat import process_data_and_generate_html 

st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")

st.title("📊 Auto Statistical Analysis")
st.markdown("อัพโหลดไฟล์ Excel/CSV หรือแปะข้อมูลเพื่อวิเคราะห์สถิติ (รันบนเครื่องของคุณ 100%)")

# 1. ส่วนอัพโหลดไฟล์
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

# 2. หรือส่วน Copy Paste (Data Editor)
st.subheader("Or paste/edit data here:")
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    # สร้างตารางเปล่าหรือโหลดตัวอย่าง
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}) 

# ตารางที่แก้ไขได้
edited_df = st.data_editor(df, num_rows="dynamic")

# 3. ปุ่มรัน
if st.button("🚀 Run Analysis"):
    with st.spinner('Calculating... (This runs inside your browser)'):
        try:
            # เรียกใช้ฟังก์ชันคำนวณของคุณตรงนี้
            # html_result = process_data_and_generate_html(edited_df)
            
            # (จำลองผลลัพธ์)
            html_result = "<h1>Results</h1><p>Table...</p>" 
            
            st.success("Done!")
            
            # 4. แสดงผลและปุ่ม Download
            st.components.v1.html(html_result, height=600, scrolling=True)
            
            st.download_button(
                label="📥 Download HTML Report",
                data=html_result,
                file_name="stat_report.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"Error: {e}")
