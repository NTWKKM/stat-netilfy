import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker: 
    1. Numeric Column -> หา Text แปลกปลอม (เช่น 'abc' ใน Age)
    2. Text Column    -> หาตัวเลขหลงมา (เช่น '1' ใน Group) 
                         และหากลุ่มประชากรน้อยผิดปกติ (Rare Category)
    """
    warnings = []
    
    total_rows = len(df)
    
    for col in df.columns:
        # เตรียมข้อมูลเช็ค
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        original_vals = df[col].astype(str).str.strip()
        
        # จำนวนค่าที่ไม่ใช่ตัวเลข (Non-Numeric)
        # (คือค่าที่แปลงเป็นตัวเลขแล้วเป็น NaN แต่ค่าเดิมต้องไม่ว่าง)
        is_non_numeric = numeric_vals.isna() & (original_vals != '') & \
                         (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        non_numeric_count = is_non_numeric.sum()

        # ======================================================
        # CASE 1: คอลัมน์นี้ควรเป็น "ตัวเลข" (Numeric)
        # (ถ้าส่วนใหญ่เป็นตัวเลข คือมี non-numeric น้อยกว่า 90%)
        # ======================================================
        if non_numeric_count < (total_rows * 0.9):
            if non_numeric_count > 0: # เจอ Text ปนมาบ้าง
                error_rows = df.index[is_non_numeric].tolist()
                bad_values = df.loc[is_non_numeric, col].unique()
                
                # Format ข้อความเตือน
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")

                msg = (f"⚠️ **Column '{col}' (Numeric):** Found {non_numeric_count} text values at rows `{row_str}` "
                       f"(Values: `{val_str}`). These will be treated as NaN.")
                warnings.append(msg)

        # ======================================================
        # CASE 2: คอลัมน์นี้เป็น "ข้อความ" (Categorical/Text)
        # (คือมี non-numeric เยอะเกิน 90% เช่น Group Treatment)
        # ======================================================
        else:
            # 2.1: เช็คว่ามี "ตัวเลข" หลงมาไหม? (เช่น 1, 0 ปนใน Group)
            # คือค่าที่แปลงเป็นตัวเลขได้ (ไม่เป็น NaN)
            is_numeric_in_text = (~numeric_vals.isna()) & (original_vals != '')
            numeric_in_text_count = is_numeric_in_text.sum()
            
            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                bad_values = df.loc[is_numeric_in_text, col].unique()
                
                row_str = ",".join(map(str, error_rows[:5])) + ("..." if len(error_rows) > 5 else "")
                val_str = ",".join(map(str, bad_values[:3])) + ("..." if len(bad_values) > 3 else "")
                
                msg = (f"⚠️ **Column '{col}' (Text):** Found {numeric_in_text_count} numeric values (e.g. 1, 0) at rows `{row_str}` "
                       f"(Values: `{val_str}`). This might be inconsistent data.")
                warnings.append(msg)

            # 2.2: เช็คว่ามี "คำที่โผล่มาน้อยผิดปกติ" (Rare Category / Typo) ไหม?
            # จะเช็คเฉพาะคอลัมน์ที่ไม่ใช่ ID (Unique values ต้องไม่เยอะเกินไป)
            unique_ratio = df[col].nunique() / total_rows
            if unique_ratio < 0.8: # ถ้าไม่ใช่ ID (เช่น ID คนไข้จะไม่เช็ค)
                val_counts = df[col].value_counts()
                
                # เงื่อนไข Rare: ปรากฏน้อยกว่า 5 ครั้ง (ปรับเลขนี้ได้ตามความเหมาะสม)
                rare_threshold = 5 
                rare_vals = val_counts[val_counts < rare_threshold].index.tolist()
                
                if len(rare_vals) > 0:
                     val_str = ", ".join(map(str, rare_vals[:5])) + ("..." if len(rare_vals) > 5 else "")
                     msg = (f"❓ **Column '{col}' (Text):** Found rare categories (appear < {rare_threshold} times): `{val_str}`. "
                            f"Please check for typos (e.g. 'Old drug', 'Alternative').")
                     warnings.append(msg)

    if warnings:
        container.warning("### 🧐 Data Quality Issue Detected\n" + "\n".join(warnings), icon="⚠️")

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
    # แปลงเป็น String เพื่อให้ User แก้ไขได้อิสระ และเห็น Text ที่ตัวเองพิมพ์ผิด
    df_display = df.astype(str).replace('nan', '')
    
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # ตรวจสอบจากสิ่งที่ User เห็น
    check_data_quality(edited_df, warning_container)

    # สร้างข้อมูลสำหรับส่งไปคำนวณ (Analysis Data)
    df_final = edited_df.copy()
    
    for col in df_final.columns:
        # Replace Custom Missing
        if custom_na_list:
            df_final[col] = df_final[col].replace(custom_na_list, np.nan)
        
        # Trim whitespace
        if df_final[col].dtype == 'object':
             df_final[col] = df_final[col].astype(str).str.strip()

        # 🟢 [จุดสำคัญ 2] Logic การแปลงข้อมูลให้ปลอดภัย
        try:
            # 1. ลองแปลงแบบ Strict (ถ้าเป็นตัวเลขหมด จะผ่าน)
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            # 2. ถ้า Error แสดงว่ามี Text ปนอยู่
            # ลองแปลงแบบ Coerce (ให้ Text กลายเป็น NaN) เก็บใส่ตัวแปรไว้ก่อน
            converted_col = pd.to_numeric(df_final[col], errors='coerce')
            
            # 3. เช็คว่า "นี่คือคอลัมน์ Text หรือเปล่า?"
            # ถ้าแปลงแล้วกลายเป็น NaN ทั้งหมด (หรือเกือบทั้งหมด) แสดงว่ามันเป็น Text Column (เช่น Group) -> ห้ามแปลง!
            # (เช็คว่า original ไม่ได้ว่างเปล่า แต่แปลงแล้วว่างเปล่า)
            if converted_col.isna().all() and not df_final[col].isna().all():
                # ✅ กรณีนี้คือ Text Column (เช่น Group Treatment) -> ให้ใช้ค่าเดิม (Text)
                pass 
            else:
                # ✅ กรณีนี้คือ Numeric Column ที่มีขยะปน (เช่น Age มี 'abc') -> ให้ใช้ค่าที่แปลงแล้ว (abc กลายเป็น NaN)
                df_final[col] = converted_col
            
    return df_final
