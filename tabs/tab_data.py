import streamlit as st
import pandas as pd
import numpy as np

def check_data_quality(df, container):
    """
    Data Quality Checker (English Version)
    Identifies non-numeric values in numeric-like columns and reports specific rows.
    """
    warnings = []
    
    for col in df.columns:
        # 1. Try converting to numeric
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Identify text errors (neither number, nor empty/NaN)
        original_vals = df[col].astype(str).str.strip()
        
        is_text_error = numeric_vals.isna() & (original_vals != '') & \
                        (original_vals.str.lower() != 'nan') & (original_vals.str.lower() != 'none')
        
        if is_text_error.any():
            total_rows = len(df)
            error_count = is_text_error.sum()
            
            # Rule: If >80% is numeric but has some text -> Warn
            if error_count < (total_rows * 0.8): 
                # Get specific rows and values
                error_rows = df.index[is_text_error].tolist()
                bad_values = df.loc[is_text_error, col].unique()
                
                # Format the row list
                row_str = ", ".join(map(str, error_rows[:10]))
                if len(error_rows) > 10: row_str += ", ..."
                
                val_str = ", ".join(map(str, bad_values[:3]))
                if len(bad_values) > 3: val_str += ", ..."

                # Markdown Syntax Message
                msg = (f"**Column '{col}':** Found {error_count} non-numeric values.\n"
                       f"- **Rows:** {row_str}\n"
                       f"- **Values:** `{val_str}`\n"
                       f"*(These will be treated as Missing Values)*")
                warnings.append(msg)

    # Display Warning in the placeholder container
    if warnings:
        container.warning("### 🧐 Data Quality Issue Detected\n\n" + "\n\n".join(warnings), icon="⚠️")

def render(df):
    st.subheader("Raw Data Table")
    
    # 🟢 UPDATE: ปรับขนาดตัวอักษร และเพิ่มคำอธิบาย
    # ใช้สัดส่วน 1.5 : 2.5 เพื่อให้ข้อความคำอธิบายมีพื้นที่พอไม่บีบเกินไป
    c1, c2 = st.columns([1.5, 2.5]) 
    with c1:
        # ปรับเป็นตัวหนาธรรมดา (ขนาดเท่าข้อความปกติ)
        st.markdown("**⚙️ Custom Missing Values:**")
        # เพิ่มคำอธิบายด้านล่าง
        st.caption("Specify values to be treated as **Missing Data (NaN)** (e.g. `-99`, `999`, `?`)")
        
    with c2:
        # ขยับ Input ลงมานิดหน่อยเพื่อให้ตรงกับ Text ด้านซ้าย (ใช้ container ช่วยจัดได้ แต่ปล่อยธรรมชาติก็พอไหว)
        missing_input = st.text_input(
            "Define Missing Values", 
            value="", 
            placeholder="e.g. -99, 999, ?",
            label_visibility="collapsed"
        )

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
        
        try:
            # Try converting strict numeric
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            # Mixed type fallback
            df_final[col] = pd.to_numeric(df_final[col], errors='ignore')

    # 6. Check Quality
    check_data_quality(df_final, warning_container)

    return df_final