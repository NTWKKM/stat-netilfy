import streamlit as st
import pandas as pd
import numpy as np
import re

def check_data_quality(df, container):
    """
    Data Quality Checker: 
    1. Numeric Column -> หา Text แปลกปลอม (รวมถึงค่าที่ติด <, >)
    2. Text Column    -> หาตัวเลขหลงมา และหากลุ่มประชากรน้อยผิดปกติ (Rare Category)
    
    Format: แสดงผล 1 บรรทัดต่อ 1 Column เพื่อความเป็นระเบียบ
    """
    warnings = [] # เก็บข้อความรวมระดับคอลัมน์
    total_rows = len(df)
    
    for col in df.columns:
        col_issues = [] # เก็บปัญหาย่อยๆ ภายในคอลัมน์นี้
        
        # เตรียมข้อมูลสำหรับเช็ค 2 แบบ
        original_vals = df[col].astype(str).str.strip()
        
        # 1. Strict Check: แปลงตรงๆ (ใช้สำหรับหา Error แจ้งเตือน)
        numeric_strict = pd.to_numeric(df[col], errors='coerce')
        is_strict_nan = numeric_strict.isna() & (original_vals != '') & \
                        (~original_vals.str.lower().isin(['nan', 'none', '']))
        strict_nan_count = is_strict_nan.sum()

        # 2. Relaxed Check: ลองลบสัญลักษณ์พิเศษออกก่อน (ใช้สำหรับตัดสิน Type)
        # ลบ <, >, ,, % ออก (เพิ่ม % เข้ามาด้วย)
        clean_vals_for_check = original_vals.str.replace(r'[<>,%]', '', regex=True)
        numeric_relaxed = pd.to_numeric(clean_vals_for_check, errors='coerce')
        
        # นับจำนวนข้อมูลที่ 'น่าจะเป็นตัวเลข'
        is_relaxed_numeric = (~numeric_relaxed.isna()) & (original_vals != '') & \
                             (~original_vals.str.lower().isin(['nan', 'none', '']))
        relaxed_numeric_count = is_relaxed_numeric.sum()
        
        # นับจำนวนข้อมูลที่ไม่ใช่ค่าว่างทั้งหมด
        non_empty_mask = (original_vals != '') & (~original_vals.str.lower().isin(['nan', 'none', '']))
        total_data_count = non_empty_mask.sum()

        # ตรวจสอบว่ามีเครื่องหมาย < หรือ > หรือไม่ (เป็นเอกลักษณ์ของ Lab Value)
        has_inequality = original_vals.str.contains(r'[<>]', regex=True).any()

        # ======================================================
        # DECISION LOGIC: เป็น Numeric หรือไม่?
        # ======================================================
        is_numeric_col = False
        if total_data_count > 0:
            ratio = relaxed_numeric_count / total_data_count
            
            # เกณฑ์ใหม่:
            # 1. ถ้ามีข้อมูล > 60% เป็นตัวเลข (ลดจาก 80%) -> Numeric
            # 2. หรือถ้ามีเครื่องหมาย <, > (Lab Value) และมีตัวเลข > 40% -> Numeric (ช่วยเคส Lab สกปรก)
            if ratio > 0.6:
                is_numeric_col = True
            elif has_inequality and ratio > 0.4:
                is_numeric_col = True
                
        else:
            # Fallback เดิม (ถ้าข้อมูลว่างเยอะๆ)
            if strict_nan_count < (total_rows * 0.9):
                is_numeric_col = True

        # ======================================================
        # CASE 1: คอลัมน์นี้ควรเป็น "ตัวเลข" (Numeric)
        # ======================================================
        if is_numeric_col:
            # ใช้ Strict Check เพื่อแจ้งเตือนค่าที่ผิดปกติ (เช่น >100, 1,000)
            if strict_nan_count > 0:
                error_rows = df.index[is_strict_nan].tolist()
                bad_values = df.loc[is_strict_nan, col].unique()
                
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")

                # เพิ่มเข้า list ย่อย (แจ้งเตือนแต่ไม่แก้)
                col_issues.append(f"Found {strict_nan_count} non-standard numeric values (e.g. with symbols <,>) at rows `{row_str}` (Values: `{val_str}`). Stats analysis will try to clean these.")

        # ======================================================
        # CASE 2: คอลัมน์นี้เป็น "ข้อความ" (Categorical/Text)
        # ======================================================
        else:
            # 2.1: เช็คว่ามี "ตัวเลข" หลงมาไหม?
            is_numeric_in_text = (~numeric_strict.isna()) & (original_vals != '')
            numeric_in_text_count = is_numeric_in_text.sum()
            
            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                bad_values = df.loc[is_numeric_in_text, col].unique()
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")
                
                col_issues.append(f"Found {numeric_in_text_count} numeric values at rows `{row_str}` (Values: `{val_str}`).")

            # 2.2: เช็ค Rare Category (คำที่โผล่มาน้อยๆ)
            unique_ratio = df[col].nunique() / total_rows
            if unique_ratio < 0.8: 
                val_counts = df[col].value_counts()
                rare_threshold = 5 
                rare_vals = val_counts[val_counts < rare_threshold].index.tolist()
                
                if len(rare_vals) > 0:
                     val_str = ", ".join(map(str, rare_vals[:5])) + ("..." if len(rare_vals) > 5 else "")
                     col_issues.append(f"Found rare categories (<{rare_threshold} times): `{val_str}`. Check for typos.")

        # 🟢 สรุปรวมปัญหาของคอลัมน์นี้ (ถ้ามี) ให้เป็น 1 บรรทัด
        if col_issues:
            full_msg = " ".join(col_issues)
            warnings.append(f"**Column '{col}':** {full_msg}")

    # แสดงผล
    if warnings:
        container.warning("Data Quality Issues Detected\n\n" + "\n\n".join([f"- {w}" for w in warnings]), icon="🧐")

def get_clean_data(df, custom_na_list=None):
    """
    สร้างสำเนาข้อมูลที่ 'Clean' แล้วสำหรับนำไปคำนวณ
    ปรับปรุง: พยายามแปลงเป็น Numeric ให้ฉลาดขึ้น (รองรับ <, >) เพื่อให้ Stat มองเป็น Continuous
    """
    df_clean = df.copy()
    total_rows = len(df_clean)

    for col in df_clean.columns:
        # 1. Custom Missing
        if custom_na_list:
             df_clean[col] = df_clean[col].replace(custom_na_list, np.nan)

        # 2. Trim
        if df_clean[col].dtype == 'object':
             df_clean[col] = df_clean[col].astype(str).str.strip()

        # 3. Numeric Conversion Logic (Improved)
        # ใช้ Logic เดียวกับ check_data_quality ในการตัดสินใจเปลี่ยน Type
        
        # ลองแปลงแบบ Clean (ลบ <, >, %)
        clean_vals = df_clean[col].astype(str).str.replace(r'[<>,%]', '', regex=True)
        numeric_relaxed = pd.to_numeric(clean_vals, errors='coerce')
        
        # เช็คว่าควรเป็น Numeric หรือไม่
        original_vals = df_clean[col].astype(str)
        non_empty_mask = (original_vals != '') & (~original_vals.str.lower().isin(['nan', 'none']))
        total_data_count = non_empty_mask.sum()
        relaxed_numeric_count = (~numeric_relaxed.isna() & non_empty_mask).sum()
        has_inequality = original_vals.str.contains(r'[<>]', regex=True).any()
        
        is_numeric_col = False
        if total_data_count > 0:
             ratio = relaxed_numeric_count / total_data_count
             # ใช้เกณฑ์เดียวกับ check_data_quality (0.6 หรือ 0.4+symbol)
             if ratio > 0.6: 
                 is_numeric_col = True
             elif has_inequality and ratio > 0.4:
                 is_numeric_col = True
        else:
             # Fallback เดิม
             if pd.to_numeric(df_clean[col], errors='coerce').isna().sum() < (total_rows * 0.9):
                 is_numeric_col = True

        if is_numeric_col:
             # ถ้าตัดสินว่าเป็น Numeric -> ใช้ค่าที่ Clean แล้ว (แปลง >100 เป็น 100.0)
             # ค่าที่แปลงไม่ได้จะเป็น NaN
             df_clean[col] = numeric_relaxed
        
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
    
    # Editor
    df_display = df.astype(str).replace('nan', '')
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # Check Quality
    check_data_quality(edited_df, warning_container)
    
    # Save State
    st.session_state['custom_na_list'] = custom_na_list
    
    return edited_df
