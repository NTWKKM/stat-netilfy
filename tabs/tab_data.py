import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    รับ container มาเพิ่ม เพื่อระบุว่าจะให้ไปแสดงข้อความเตือนที่ไหน
    """
    warnings = []
    
    for col in df.columns:
        # 1. ลองแปลงค่าเป็นตัวเลข
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        
        # 2. เช็คว่าเป็น Text ที่ไม่ใช่ค่าว่าง
        original_vals = df[col].astype(str).str.strip()
        is_text_error = numeric_vals.isna() & (original_vals != '') & (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        
        if is_text_error.any():
            total_rows = len(df)
            text_count = is_text_error.sum()
            
            # กฎ: ถ้าส่วนใหญ่เป็นตัวเลข (>80%) แต่มี text ปน -> เตือน
            if text_count < (total_rows * 0.8): 
                bad_values = df.loc[is_text_error, col].unique()
                example_vals = ", ".join(map(str, bad_values[:3]))
                warnings.append(f"⚠️ **Column '{col}':** พบข้อความ {text_count} จุด (เช่น `{example_vals}` ...) -> ค่าเหล่านี้จะถูกนับเป็น **Missing Value**")

    # แสดงผลลงใน Container ที่จองไว้ (แทนที่จะใช้ st.warning โดยตรง)
    if warnings:
        container.warning("### 🧐 Data Quality Check\n" + "\n\n".join(warnings))
    else:
        # แสดงสีเขียวถ้าข้อมูลเรียบร้อย (ถ้าไม่อยากให้รกพื้นที่ด้านบน ลบบรรทัดนี้ออกได้ครับ)
        container.success("✅ Data Clean! (ข้อมูลตัวเลขถูกต้อง หรือถูกเว้นว่างไว้อย่างเหมาะสม)")

def render(df):
    st.subheader("Raw Data Table")
    st.info("💡 You can view, scroll, and edit your raw data directly in this table. (พิมพ์ตัวอักษรผสมตัวเลขได้)")
    
    # 🟢 1. สร้าง Placeholder จองพื้นที่ไว้ตรงนี้ (ต่อจาก Info)
    warning_container = st.empty()
    
    # 2. เตรียมข้อมูล (String conversion)
    df_display = df.astype(str).replace('nan', '')
    
    # 3. แสดงตาราง Editor (Code จะรันส่วนนี้และรอ user แก้ไข)
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # 4. แปลงข้อมูลกลับเป็นตัวเลข
    df_final = edited_df.copy()
    for col in df_final.columns:
        try:
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            df_final[col] = pd.to_numeric(df_final[col], errors='ignore')

    # 🟢 5. ส่ง container ไปให้ฟังก์ชันเขียนผลลัพธ์ใส่ (มันจะเด้งไปแสดงข้างบน)
    check_data_quality(df_final, warning_container)

    return df_final