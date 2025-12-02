import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker: ตรวจสอบและแจ้งเตือนจากข้อมูลดิบ (edited_df)
    """
    warnings = []
    
    for col in df.columns:
        # 1. ลองแปลงเป็นตัวเลขเพื่อเช็ค
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        
        # 2. เช็คว่าเป็นข้อความที่ผิดปกติหรือไม่ (ไม่ใช่ตัวเลข และไม่ใช่ค่าว่าง)
        original_vals = df[col].astype(str).str.strip()
        is_text_error = numeric_vals.isna() & (original_vals != '') & \
                        (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        
        if is_text_error.any():
            total_rows = len(df)
            error_count = is_text_error.sum()
            
            # ถ้ามี Error แจ้งเตือน
            if error_count < (total_rows * 0.9): 
                error_rows = df.index[is_text_error].tolist()
                bad_values = df.loc[is_text_error, col].unique()
                
                row_str = ",".join(map(str, error_rows[:5])) 
                if len(error_rows) > 5: row_str += "..."
                
                val_str = ",".join(map(str, bad_values[:3])) 
                if len(bad_values) > 3: val_str += "..."

                msg = (f"⚠️ **Column '{col}':** Found {error_count} non-numeric values at **Rows:** `{row_str}` "
                       f"(Values: `{val_str}`). Analysis will treat these as Missing (NaN).")
                warnings.append(msg)

    if warnings:
        container.warning("Data Quality Issue Detected\n" + "\n".join(warnings), icon="⚠️")

def render(df):
    st.subheader("Raw Data Table")
    
    # Layout
    col_info, col_btn = st.columns([4, 1.5], vertical_alignment="center")
    
    with col_info:
        st.info("You can view, scroll, and edit your raw data below. (Text inputs allowed)", icon="💡")

    with col_btn:
        with st.popover("⚙️ Config Missing Values", use_container_width=True):
            st.markdown("**Define Custom Missing Values**")
            st.caption("Values to treat as **NaN** (e.g. `-99`, `?`)")
            missing_input = st.text_input("Enter values separated by comma", value="", placeholder="e.g. -99, 999")
    
    # Placeholder for Warnings
    warning_container = st.empty()
    
    custom_na_list = [x.strip() for x in missing_input.split(',') if x.strip() != '']
    
    # 🟢 1. เตรียมข้อมูลสำหรับแสดงผล (เป็น String เพื่อให้ User แก้ไขได้อิสระ)
    st.write("") 
    st.write("") 
    df_display = df.astype(str).replace('nan', '')
    
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # 🟢 2. ตรวจสอบ Error จากข้อมูลดิบที่ User เห็น (edited_df)
    # ต้องตรวจก่อนแปลงเป็น NaN ไม่งั้นจะหา Text ไม่เจอ
    check_data_quality(edited_df, warning_container)

    # 🟢 3. สร้างข้อมูลสำหรับส่งไปคำนวณ (Analysis Data)
    df_final = edited_df.copy()
    
    for col in df_final.columns:
        # Replace Custom Missing
        if custom_na_list:
            df_final[col] = df_final[col].replace(custom_na_list, np.nan)
        
        # Trim whitespace
        if df_final[col].dtype == 'object':
             df_final[col] = df_final[col].astype(str).str.strip()

        # Try Convert to Numeric
        try:
            # ถ้าแปลงได้ปกติ ก็จบ
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            # 🟢 ถ้าแปลงไม่ได้ (เช่น "abc") -> บังคับเปลี่ยนเป็น NaN เฉพาะใน df_final
            # เพื่อให้หน้า Analyze เอาไปคำนวณได้โดยไม่พัง
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
            
    return df_final
