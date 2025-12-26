import streamlit as st
import pandas as pd
import numpy as np
import math

def _is_numeric_column(col_series: pd.Series, total_rows: int) -> tuple:
    """
    Determine if a column should be treated as numeric.
    Returns: (is_numeric, strict_numeric_series, relaxed_numeric_series, strict_nan_mask, strict_nan_count)
    """
    original_vals = col_series.astype(str).str.strip()
    
    # Strict check
    numeric_strict = pd.to_numeric(col_series, errors='coerce')
    is_strict_nan = numeric_strict.isna() & (original_vals != '') & \
                    (~original_vals.str.lower().isin(['nan', 'none', '']))
    strict_nan_count = is_strict_nan.sum()
    
    # Relaxed check
    clean_vals = original_vals.str.replace(r'[<>,%]', '', regex=True)
    numeric_relaxed = pd.to_numeric(clean_vals, errors='coerce')
    
    is_relaxed_numeric = (~numeric_relaxed.isna()) & (original_vals != '') & \
                         (~original_vals.str.lower().isin(['nan', 'none', '']))
    relaxed_numeric_count = is_relaxed_numeric.sum()
    
    non_empty_mask = (original_vals != '') & (~original_vals.str.lower().isin(['nan', 'none']))
    total_data_count = non_empty_mask.sum()
    has_inequality = original_vals.str.contains(r'[<>]', regex=True).any()
    
    # Decision logic
    is_numeric_col = False
    if total_data_count > 0:
        ratio = relaxed_numeric_count / total_data_count
        if ratio > 0.6:
            is_numeric_col = True
        elif has_inequality and ratio > 0.4:
            is_numeric_col = True
    else:
        if strict_nan_count < (total_rows * 0.9):
            is_numeric_col = True
    
    return is_numeric_col, numeric_strict, numeric_relaxed, is_strict_nan, strict_nan_count

def check_data_quality(df, container):
    """
    Data Quality Checker (ตรวจสอบเฉพาะข้อมูลในหน้านั้นๆ เพื่อความรวดเร็ว): 
    1. Numeric Column -> หา Text แปลกปลอม (รวมถึงค่าที่ติด <, >) และแจ้งเตือนแบบ Strict
    2. Text Column    -> หาตัวเลขหลงมา และหากลุ่มประชากรน้อยผิดปกติ
    """
    warnings = [] 
    total_rows = len(df)
    
    # ถ้าไม่มีข้อมูลในหน้านี้ (กรณีลบจนหมด) ให้ข้าม
    if total_rows == 0:
        return

    for col in df.columns:
        col_issues = []
        
        # Use helper function
        is_numeric_col, numeric_strict, _, is_strict_nan, strict_nan_count = _is_numeric_column(df[col], total_rows)

        # CASE 1: Numeric
        if is_numeric_col:
            if strict_nan_count > 0:
                error_rows = df.index[is_strict_nan].tolist()
                bad_values = df.loc[is_strict_nan, col].unique()
                row_str = ",".join(map(str, error_rows[:3])) + ("..." if len(error_rows) > 3 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")
                col_issues.append(f"Found {strict_nan_count} non-standard values at rows `{row_str}` (Values: `{val_str}`).")

        # CASE 2: Categorical
        else:
            original_vals = df[col].astype(str).str.strip()
            is_numeric_in_text = (~numeric_strict.isna()) & (original_vals != '')
            numeric_in_text_count = is_numeric_in_text.sum()
            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                bad_values = df.loc[is_numeric_in_text, col].unique()
                row_str = ",".join(map(str, error_rows[:3])) + ("..." if len(error_rows) > 3 else "")
                col_issues.append(f"Found {numeric_in_text_count} numeric values inside categorical column at rows `{row_str}`.")

            unique_ratio = df[col].nunique() / total_rows
            if unique_ratio < 0.8: 
                val_counts = df[col].value_counts()
                rare_threshold = 5 
                rare_vals = val_counts[val_counts < rare_threshold].index.tolist()
                if len(rare_vals) > 0:
                     val_str = ", ".join(map(str, rare_vals[:3])) + ("..." if len(rare_vals) > 3 else "")
                     col_issues.append(f"Found rare categories (<{rare_threshold} times): `{val_str}`.")

        if col_issues:
            full_msg = " ".join(col_issues)
            warnings.append(f"**Column '{col}':** {full_msg}")

    if warnings:
        container.warning("Data Quality Issues (Current Page)\n\n" + "\n\n".join([f"- {w}" for w in warnings]), icon="🧐")

def get_clean_data(df, custom_na_list=None):
    """
    สร้างสำเนาข้อมูลที่ 'Clean' แล้ว (Logic เดิม)
    """
    df_clean = df.copy()
    total_rows = len(df_clean)

    for col in df_clean.columns:
        if custom_na_list:
             df_clean[col] = df_clean[col].replace(custom_na_list, np.nan)

        if df_clean[col].dtype == 'object':
             df_clean[col] = df_clean[col].astype(str).str.strip()

        # Use helper function
        is_numeric_col, _, numeric_relaxed, _, _ = _is_numeric_column(df_clean[col], total_rows)

        if is_numeric_col:
             df_clean[col] = numeric_relaxed
        
    return df_clean

def render(df):
    st.subheader("Raw Data Table")
    
    # --- Config Section ---
    col_info, col_btn = st.columns([4, 1.5], vertical_alignment="center")
    with col_info:
        st.info("You can view and edit your raw data below.", icon="💡")

    with col_btn:
        with st.popover("⚙️ Config Missing Values", use_container_width=True):
            st.markdown("**Define Custom Missing Values**")
            missing_input = st.text_input("Values separated by comma", value="", placeholder="e.g. -99, 999")
    
    warning_container = st.empty()
    custom_na_list = [x.strip() for x in missing_input.split(',') if x.strip() != '']
    st.session_state['custom_na_list'] = custom_na_list
    
    st.write("") 

    # --- ⚡ PAGINATION LOGIC ---
    batch_size = 600
    total_rows = len(df)
    total_pages = math.ceil(total_rows / batch_size) if total_rows > 0 else 1

    # สร้าง container สำหรับเลือกหน้า
    if total_pages > 1:
        c1, c2, _ = st.columns([1, 2, 8])
        with c1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        with c2:
            st.write("") # Spacer
            st.markdown(f"<div style='padding-top: 10px;'>of {total_pages} ({total_rows} rows)</div>", unsafe_allow_html=True)
    else:
        page = 1
        st.caption(f"Showing all {total_rows} rows")

    # คำนวณ Index เริ่มและจบของหน้านั้น
    start_idx = (page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_rows)

    # ตัดข้อมูลมาแสดง (Slice)
    # ใช้ .copy() เพื่อป้องกัน SettingWithCopyWarning
    df_slice = df.iloc[start_idx:end_idx].copy()
    
    # แปลงเป็น string เพื่อแสดงผลให้เห็นครบทุก format (เช่น 001, >100)
    df_display_slice = df_slice.astype(str).replace('nan', '')

    # --- EDITOR ---
    # เราแสดงผลแค่ Slice แต่เมื่อ User แก้ไข เราต้องเอาค่าไปอัปเดต df ตัวแม่
    edited_slice = st.data_editor(
        df_display_slice, 
        num_rows="dynamic", # อนุญาตให้เพิ่ม/ลบแถวได้ (แต่จะเพิ่มในหน้านี้)
        use_container_width=True, 
        height=450, 
        key=f'editor_raw_page_{page}' # Key เปลี่ยนตามหน้าเพื่อ refresh editor
    )

    # --- UPDATE LOGIC ---
    # ตรวจสอบว่ามีการแก้ไขข้อมูลใน Slice หรือไม่
    # หมายเหตุ: การเทียบแบบนี้อาจช้าถ้าข้อมูลเยอะ แต่สำหรับ slice 100 แถว ถือว่าเร็วมาก
    if not df_display_slice.equals(edited_slice):
        # 1. อัปเดตข้อมูลกลับไปยัง DataFrame ตัวแม่ (df)
        # เราใช้ index ของ slice เพื่อระบุตำแหน่งใน df ตัวแม่
        
        # ต้องระวังเรื่อง Data Type ตอน update กลับ
        # พยายามคงค่าเดิมไว้ถ้าไม่ได้แก้ แต่ถ้าแก้จะกลายเป็น object (text)
        
        # กรณีมีการเพิ่ม/ลบแถว จำนวน index อาจไม่เท่ากัน
        if len(edited_slice) != len(df_slice):
             st.warning("Adding/Deleting rows in pagination mode handles complexity. Please reset data if indices mismatch.")
             # กรณี Simple Update:
             # การจัดการเพิ่ม/ลบแถวใน Pagination ซับซ้อนมาก แนะนำให้ User ระวัง
             # แต่นี่จะรองรับการ Edit ค่า (Cell Edit) เป็นหลัก
        
        # Update ค่ากลับ
        try:
             # ใช้ loop update เฉพาะค่าที่เปลี่ยน เพื่อความปลอดภัย
             # หรือใช้ update ของ pandas (เร็วแต่ต้องระวัง index)
             
             # แปลงกลับเป็น Original Type เท่าที่ทำได้ หรือเก็บเป็น Object ไปก่อน
             # เพราะเราต้องการ Raw Data แบบ Text
             
             # สร้าง df update ที่มี index ตรงกับ master
             df.loc[edited_slice.index, edited_slice.columns] = edited_slice
             
             # Force rerun เพื่อให้ข้อมูลอัปเดตทันที
             # st.rerun() # อาจจะทำให้กระพริบ ปิดไว้ก่อน
        except Exception as e:
             st.error(f"Error updating data: {e}")

    # --- CHECK QUALITY (เฉพาะหน้านี้) ---
    # ส่ง edited_slice ไปเช็ค เพื่อให้ User เห็น Warning ทันทีที่แก้
    check_data_quality(edited_slice, warning_container)

    # Return ตัว df ทั้งหมดที่มีการอัปเดตแล้ว
    return df
