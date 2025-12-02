import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker: 
    1. Numeric Column -> หา Text แปลกปลอม
    2. Text Column    -> หาตัวเลขหลงมา และหากลุ่มประชากรน้อยผิดปกติ (Rare Category)
    
    Format: แสดงผล 1 บรรทัดต่อ 1 Column เพื่อความเป็นระเบียบ
    """
    warnings = [] # เก็บข้อความรวมระดับคอลัมน์
    total_rows = len(df)
    
    for col in df.columns:
        col_issues = [] # เก็บปัญหาย่อยๆ ภายในคอลัมน์นี้
        
        # เตรียมข้อมูลเช็ค
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        original_vals = df[col].astype(str).str.strip()
        
        # จำนวนค่าที่ไม่ใช่ตัวเลข
        is_non_numeric = numeric_vals.isna() & (original_vals != '') & \
                         (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        non_numeric_count = is_non_numeric.sum()

        # ======================================================
        # CASE 1: คอลัมน์นี้ควรเป็น "ตัวเลข" (Numeric)
        # ======================================================
        if non_numeric_count < (total_rows * 0.9):
            if non_numeric_count > 0:
                error_rows = df.index[is_non_numeric].tolist()
                bad_values = df.loc[is_non_numeric, col].unique()
                
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")

                # เพิ่มเข้า list ย่อย
                col_issues.append(f"Found {non_numeric_count} non-numeric values at rows `{row_str}` (Values: `{val_str}`). Analysis will treat these as NaN.")

        # ======================================================
        # CASE 2: คอลัมน์นี้เป็น "ข้อความ" (Categorical/Text)
        # ======================================================
        else:
            # 2.1: เช็คว่ามี "ตัวเลข" หลงมาไหม?
            is_numeric_in_text = (~numeric_vals.isna()) & (original_vals != '')
            numeric_in_text_count = is_numeric_in_text.sum()
            
            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                bad_values = df.loc[is_numeric_in_text, col].unique()
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")
                
                col_issues.append(f"Found {numeric_in_text_count} numeric values (e.g. 1, 0) at rows `{row_str}` (Values: `{val_str}`).")

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
            # รวมทุกปัญหาในคอลัมน์นี้ด้วยเว้นวรรค
            full_msg = " ".join(col_issues)
            # สร้างข้อความเตือนแบบมีหัวข้อคอลัมน์ชัดเจน
            warnings.append(f"**Column '{col}':** {full_msg}")

    # แสดงผล (ใช้ \n\n เพื่อเว้นบรรทัดให้ห่างกันชัดเจน)
    if warnings:
        container.warning("Data Quality Issues Detected\n\n" + "\n\n".join([f"- {w}" for w in warnings]), icon="🧐")

def get_clean_data(df, custom_na_list=None):
    """
    สร้างสำเนาข้อมูลที่ 'Clean' แล้วสำหรับนำไปคำนวณ
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

        # 3. Numeric Conversion Logic
        numeric_vals = pd.to_numeric(df_clean[col], errors='coerce')
        is_non_numeric = numeric_vals.isna()
        
        # ถ้าแปลงแล้ว NaN น้อยกว่า 90% (เป็น Numeric) -> ใช้ค่าที่แปลงแล้ว
        if is_non_numeric.sum() < (total_rows * 0.9):
             df_clean[col] = numeric_vals
        
    return df_clean

def render(df):
    st.subheader("Raw Data Table")
    
    col_info, col_btn = st.columns([4, 1.5], vertical_alignment="center")
    with col_info:
        st.info("💡 You can view, scroll, and edit your raw data below. (Text inputs allowed)", icon="💡")

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
