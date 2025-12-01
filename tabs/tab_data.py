import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker (English Version - Compact Mode)
    Identifies non-numeric values and reports them concisely in 1-2 lines.
    """
    warnings = []
    
    for col in df.columns:
        # 1. Try converting to numeric
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Identify text errors
        original_vals = df[col].astype(str).str.strip()
        is_text_error = numeric_vals.isna() & (original_vals != '') & \
                        (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        
        if is_text_error.any():
            total_rows = len(df)
            error_count = is_text_error.sum()
            
            if error_count < (total_rows * 0.8): 
                error_rows = df.index[is_text_error].tolist()
                bad_values = df.loc[is_text_error, col].unique()
                
                # Format Lists nicely
                row_str = ",".join(map(str, error_rows[:5])) # Show top 5 rows
                if len(error_rows) > 5: row_str += "..."
                
                val_str = ",".join(map(str, bad_values[:3])) # Show top 3 values
                if len(bad_values) > 3: val_str += "..."

                # 🟢 Compact Message (1-2 Lines)
                msg = (f"⚠️ **Column '{col}':** Found {error_count} non-numeric values at **Rows:** `{row_str}` "
                       f"(Values: `{val_str}`). **Action:** Treated as Missing (NaN).")
                warnings.append(msg)

    # Display Warnings cleanly
    if warnings:
        # ใช้ \n ตัวเดียวเพื่อให้บรรทัดชิดกันมากขึ้น
        container.warning("### 🧐 Data Quality Issue Detected\n" + "\n".join(warnings), icon="⚠️")

def render(df):
    st.subheader("Raw Data Table")
    
    # 🟢 UPDATE: ปรับ Layout ให้ Input สั้นลง (ใช้ 3 columns)
    # c1 = Label & Caption (กว้างหน่อย)
    # c2 = Input Box (กว้างพอดีๆ ไม่ยาวเกิน)
    # c3 = Spacer (พื้นที่ว่างด้านขวา เพื่อบีบ c2 ให้ไม่ยืดจนสุด)
    c1, c2, c3 = st.columns([3, 2, 5]) 
    
    with c1:
        st.markdown("**⚙️ Custom Missing Values:**")
        st.caption("Values to treat as **NaN** (e.g. `-99`, `?`)")
        
    with c2:
        missing_input = st.text_input(
            "Define Missing Values", 
            value="", 
            placeholder="e.g. -99, 999",
            label_visibility="collapsed"
        )
    # c3 ปล่อยว่างไว้

    st.info("💡 You can view, scroll, and edit your raw data below. (Text inputs allowed)")
    
    # 1. Placeholder for Warnings
    warning_container = st.empty()
    
    # 2. Prepare custom missing list
    custom_na_list = [x.strip() for x in missing_input.split(',') if x.strip() != '']
    
    # 3. Convert to String for Editor
    df_display = df.astype(str).replace('nan', '')
    
    # 4. Render Editor
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # 5. Process & Convert back to Numeric
    df_final = edited_df.copy()
    
    for col in df_final.columns:
        # 5.1: Replace Custom Missing Values
        if custom_na_list:
            df_final[col] = df_final[col].replace(custom_na_list, np.nan)
        
        # 🟢 เพิ่ม: ลบช่องว่างหัวท้าย (Trim) สำหรับคอลัมน์ที่เป็นข้อความก่อนแปลง
        # ช่วยแก้เคส "12.5 " (มี space) ให้กลายเป็น "12.5" ปกติ
        if df_final[col].dtype == 'object':
             df_final[col] = df_final[col].astype(str).str.strip()

        try:
            # 1. ลองแปลงเป็นตัวเลขแบบปกติ (ถ้าข้อมูลสะอาดจะผ่านบรรทัดนี้)
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            # 2. ถ้ามี Error (แสดงว่ามีค่าแปลกปลอม หรือเป็นข้อความ)
            # ลองแปลงแบบ coerce ดูผลลัพธ์ก่อน (ค่าแปลกปลอมจะกลายเป็น NaN)
            converted = pd.to_numeric(df_final[col], errors='coerce')
            
            # 🟢 เช็คความปลอดภัย: 
            # ถ้าแปลงแล้วกลายเป็น NaN ทั้งหมด (แสดงว่าเป็น Text ล้วนๆ เช่น "Treatment Group")
            # และข้อมูลต้นฉบับไม่ได้ว่างเปล่า
            if converted.isna().all() and not df_final[col].isna().all():
                # ให้เก็บค่าเดิมที่เป็น Text ไว้ (เหมือน errors='ignore')
                pass 
            else:
                # กรณีนี้คือแปลงได้บ้าง (แสดงว่าเป็นคอลัมน์ตัวเลขที่มีขยะปน)
                # ให้ใช้ค่าที่แปลงแล้ว (ขยะจะถูกเปลี่ยนเป็น NaN ตามต้องการ)
                df_final[col] = converted
            
    # 6. Check Quality
    check_data_quality(df_final, warning_container)

    return df_final
