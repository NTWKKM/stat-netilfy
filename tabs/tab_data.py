import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker (English Version - Compact Mode)
    Identifies non-numeric values and reports them concisely.
    """
    warnings = []
    
    for col in df.columns:
        # 1. Try converting to numeric (for checking purpose only)
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Identify text errors
        # แปลงเป็น String เพื่อเช็คว่ามีตัวอักษรแปลกปลอมไหม
        original_vals = df[col].astype(str).str.strip()
        
        # เงื่อนไข: แปลงเป็นตัวเลขไม่ได้ AND ไม่ใช่ช่องว่าง AND ไม่ใช่คำว่า nan/none
        is_text_error = numeric_vals.isna() & (original_vals != '') & \
                        (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        
        if is_text_error.any():
            total_rows = len(df)
            error_count = is_text_error.sum()
            
            # ถ้ามี Error (แต่ไม่เยอะจนเกินไป เหมือนเป็น Text Column ทั้งอัน)
            if error_count < (total_rows * 0.9): 
                error_rows = df.index[is_text_error].tolist()
                bad_values = df.loc[is_text_error, col].unique()
                
                # Format Lists nicely
                row_str = ",".join(map(str, error_rows[:5])) 
                if len(error_rows) > 5: row_str += "..."
                
                val_str = ",".join(map(str, bad_values[:3])) 
                if len(bad_values) > 3: val_str += "..."

                # 🟢 Warning Message (แจ้งเตือนอย่างเดียว ไม่บอกว่าแก้ให้แล้ว)
                msg = (f"⚠️ **Column '{col}':** Found {error_count} non-numeric values at **Rows:** `{row_str}` "
                       f"(Values: `{val_str}`). Please check your data.")
                warnings.append(msg)

    # Display Warnings cleanly
    if warnings:
        container.warning("### Data Quality Issue Detected\n" + "\n".join(warnings), icon="⚠️")

def render(df):
    st.subheader("Raw Data Table")
    
    # 🟢 ปรับ Layout แนวนอน: Info Box (ซ้าย) + ปุ่ม Popover (ขวา)
    col_info, col_btn = st.columns([4, 1.5], vertical_alignment="center")
    
    with col_info:
        st.info("You can view, scroll, and edit your raw data below. (Text inputs allowed)", icon="💡")

    with col_btn:
        with st.popover("⚙️ Config Missing Values", use_container_width=True):
            st.markdown("**Define Custom Missing Values**")
            st.caption("Values to treat as **NaN** (e.g. `-99`, `?`)")
            
            missing_input = st.text_input(
                "Enter values separated by comma", 
                value="", 
                placeholder="e.g. -99, 999"
            )
    
    # 1. Placeholder for Warnings
    warning_container = st.empty()
    
    # 2. Prepare custom missing list
    custom_na_list = [x.strip() for x in missing_input.split(',') if x.strip() != '']
    
    # 3. Convert to String for Editor (เพื่อให้แก้ไขได้อิสระ)
    df_display = df.astype(str).replace('nan', '')
    
    # 🟢 เพิ่มระยะห่างก่อนเริ่มตาราง (แก้ปัญหา Popup บัง Input)
    st.write("") 
    st.write("") 

    # 4. Render Editor
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # 5. Process Data (Without Auto-Delete)
    df_final = edited_df.copy()
    
    for col in df_final.columns:
        # 5.1: Replace Custom Missing Values
        if custom_na_list:
            df_final[col] = df_final[col].replace(custom_na_list, np.nan)
        
        # 5.2: Trim Whitespace
        if df_final[col].dtype == 'object':
             df_final[col] = df_final[col].astype(str).str.strip()

        # 5.3: Try Convert to Numeric (Strictly)
        try:
            # ลองแปลงเป็นตัวเลข ถ้าได้ก็เปลี่ยนเลย
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            # 🟢 ถ้าแปลงไม่ได้ (แสดงว่ามีตัวอักษรปน)
            # ของเดิม: แปลงเป็น NaN (errors='coerce') -> ทำให้ข้อมูลหายและไม่เตือน
            # ของใหม่: ไม่ต้องทำอะไร (pass) -> ปล่อยให้เป็น String คาไว้แบบนั้น
            # ผลลัพธ์: check_data_quality จะมาตรวจเจอทีหลังและแจ้งเตือน User เอง
            pass
            
    # 6. Check Quality (ตรวจจับ String ที่หลงเหลืออยู่ใน df_final)
    check_data_quality(df_final, warning_container)

    return df_final
