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
                
                # Format the row list (show first 10 rows to save space)
                row_str = ", ".join(map(str, error_rows[:10]))
                if len(error_rows) > 10: row_str += ", ..."
                
                val_str = ", ".join(map(str, bad_values[:3]))
                if len(bad_values) > 3: val_str += ", ..."

                # Construct detailed message
                msg = (f"⚠️ **Column '{col}':** Found {error_count} non-numeric values.<br>"
                       f"&nbsp;&nbsp;&nbsp;&nbsp;• **Rows:** {row_str}<br>"
                       f"&nbsp;&nbsp;&nbsp;&nbsp;• **Values:** `{val_str}`<br>"
                       f"&nbsp;&nbsp;&nbsp;&nbsp;*(These will be treated as Missing Values)*")
                warnings.append(msg)

    # Display Warning in the placeholder container
    if warnings:
        container.warning("### 🧐 Data Quality Issue Detected\n" + "\n\n".join(warnings), icon="⚠️")
    else:
        # Optional: Show green success message
        # container.success("✅ Data Clean! (All numeric columns are valid)")
        pass

def render(df):
    st.subheader("Raw Data Table")
    st.info("💡 You can view, scroll, and edit your raw data below. (Text inputs allowed in numeric columns)")
    
    # 1. Placeholder for Warnings (appears above the table)
    warning_container = st.empty()
    
    # 2. Convert to String to allow free editing
    df_display = df.astype(str).replace('nan', '')
    
    # 3. Render Editor
    edited_df = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=500, 
        key='editor_raw'
    )

    # 4. Process & Convert back to Numeric
    df_final = edited_df.copy()
    for col in df_final.columns:
        try:
            # Try converting strict numeric first
            df_final[col] = pd.to_numeric(df_final[col], errors='raise')
        except:
            # If failed, convert only valid numbers, keep text as is (Mixed Type)
            df_final[col] = pd.to_numeric(df_final[col], errors='ignore')

    # 5. Check Quality and Update Warning Box
    check_data_quality(df_final, warning_container)

    return df_final