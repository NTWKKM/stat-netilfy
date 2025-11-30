import streamlit as st

def render(df):
    st.subheader("Raw Data Table")
    st.info("💡 You can view, scroll, and edit your raw data directly in this table.")
    
    # ส่งคืน df ที่แก้ไขแล้วกลับไป
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=500, key='editor_raw')
    return edited_df
