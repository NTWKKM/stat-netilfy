import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker: 
    1. Numeric Column -> หา Text แปลกปลอม (เช่น 'abc' ใน Age)
    2. Text Column    -> หาตัวเลขหลงมา (เช่น '1' ใน Group) 
                         และหากลุ่มประชากรน้อยผิดปกติ (Rare Category เช่น 'Old Drug')
    """
    warnings = []
    total_rows = len(df)
    
    for col in df.columns:
        # เตรียมข้อมูลเช็ค
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        original_vals = df[col].astype(str).str.strip()
        
        # นับจำนวนค่าที่ไม่ใช่ตัวเลข (Non-Numeric)
        is_non_numeric = numeric_vals.isna() & (original_vals != '') & \
                         (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        non_numeric_count = is_non_numeric.sum()

        # ======================================================
        # CASE 1: คอลัมน์นี้ควรเป็น "ตัวเลข" (Numeric)
        # (ถ้าส่วนใหญ่เป็นตัวเลข คือมี non-numeric น้อยกว่า 90%)
        # ======================================================
        if non_numeric_count < (total_rows * 0.9):
            if non_numeric_count > 0:
                error_rows = df.index[is_non_numeric].tolist()
                bad_values = df.loc[is_non_numeric, col].unique()
                
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")

                msg = (f"⚠️ **Column '{col}' (Numeric):** Found {non_numeric_count} text values at rows `{row_str}` "
                       f"(Values: `{val_str}`). Analysis will treat these as Missing (NaN).")
                warnings.append(msg)

        # ======================================================
        # CASE 2: คอลัมน์นี้เป็น "ข้อความ" (Categorical/Text)
        # ======================================================
        else:
            # 2.1: เช็คว่ามี "ตัวเลข" หลงมาไหม? (เช่น 1, 0 ปนใน Group)
            is_numeric_in_text = (~numeric_vals.isna()) & (original_vals != '')
            numeric_in_text_count = is_numeric_in_text.sum()
            
            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                bad_values = df.loc[is_numeric_in_text, col].unique()
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")
                
                msg = (f"⚠️ **Column '{col}' (Text):** Found {numeric_in_text_count} numeric values at rows `{row_str}` "
                       f"(Values: `{val_str}`). This might be inconsistent data.")
                warnings.append(msg)

            # 🟢 [ส่วนที่เพิ่มกลับมา] 2.2: เช็คว่ามี "คำที่โผล่มาน้อยผิดปกติ" (Rare Category) ไหม?
            # จะเช็คเฉพาะคอลัมน์ที่ไม่ใช่ ID (โดยดูว่าค่าไม่ซ้ำกันเกิน 80% ของข้อมูลทั้งหมด)
            unique_ratio = df[col].nunique() / total_rows
            if unique_ratio < 0.8: 
                val_counts = df[col].value_counts()
                
                # เงื่อนไข Rare: ปรากฏน้อยกว่า 5 ครั้ง (ปรับเลขได้)
                rare_threshold = 5 
                rare_vals = val_counts[val_counts < rare_threshold].index.tolist()
                
                if len(rare_vals) > 0:
                     val_str = ", ".join(map(str, rare_vals[:5])) + ("..." if len(rare_vals) > 5 else "")
                     msg = (f"❓ **Column '{col}' (Text):** Found rare categories (appear < {rare_threshold} times): `{val_str}`. "
                            f"Please check for typos")
                     warnings.append(msg)

    if warnings:
        container.warning("Data Quality Issue Detected\n" + "\n".join(warnings), icon="⚠️")

def get_clean_data(df, custom_na_list=None):
    """
    สร้างสำเนาข้อมูลที่ 'Clean' แล้วสำหรับนำไปคำนวณ (Analysis Data)
    โดยไม่กระทบกับข้อมูลดิบที่แสดงบนหน้าจอ
    """
    df_clean = df.copy()
    total_rows = len(df_clean)

    for col in df_clean.columns:
        # 1. จัดการ Custom Missing Values
        if custom_na_list:
             df_clean[col] = df_clean[col].replace(custom_na_list, np.nan)

        # 2. Trim whitespace
        if df_clean[col].dtype == 'object':
             df_clean[col] = df_clean[col].astype(str).str.strip()

        # 3. ตัดสินใจว่าจะแปลงเป็นตัวเลขหรือไม่
        numeric_vals = pd.to_numeric(df_clean[col], errors='coerce')
        is_non_numeric = numeric_vals.isna()
        
        # ถ้าแปลงแล้วเป็น NaN น้อยกว่า 90% (แสดงว่าเป็น Numeric Column) ให้แปลงเลย
        if is_non_numeric.sum() < (total_rows * 0.9):
             df_clean[col] = numeric_vals
        
    return df_clean

def render(df):
    st.subheader("Raw Data Table")
    
    col_info, col_btn = st.columns([4, 1.5], vertical_alignment="center")
    with col_info:
        st.info("You can view, scroll, and edit your raw data below. (Text inputs allowed)", icon="💡")

    with col_btn:
        with st.popover("⚙️ Config Missing Values", use_container_width=True):
            st.markdown("**Define Custom Missing Values**")
            st.caption("Values to treat as **NaN** (e.g. `-99`, `?`)")
            missing_input = st.text_input("Enter values separated by comma", value="", placeholder="e.g. -99, 999")
    
    warning_container = st.empty()
    custom_na_list = [x.strip() for x in missing_input.split(',') if x.strip() != '']
    
    st.write("") 
    st.write("") 
    
    # แปลงเป็น String เพื่อให้แก้ไขได้อิสระ
    df_display = df.astype(str).replace('nan', '')
    
    # RAW DATA EDITOR
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # ตรวจสอบ Error
    check_data_quality(edited_df, warning_container)
    
    # ฝาก custom_na_list ไว้ใน session_state
    st.session_state['custom_na_list'] = custom_na_list
    
    # ส่งคืนข้อมูลดิบ (edited_df) กลับไปเลย (Text แปลกๆ จะยังอยู่ครบ)
    return edited_df
