"""Shared utility for dataset selection across analysis tabs."""
import streamlit as st
import pandas as pd
from typing import Tuple


def get_dataset_for_analysis(
    df: pd.DataFrame,
    session_key: str,
    default_to_matched: bool = True,
    label_prefix: str = "📄 เลือกชุดข้อมูล:"
) -> Tuple[pd.DataFrame, str]:
    """
    Choose between the original and matched DataFrame for analysis and return the selected DataFrame with a descriptive label.
    
    If a matched dataset exists in Streamlit session state (keys "is_matched" and "df_matched"), the function offers a selection between the original and matched data; otherwise it returns the original DataFrame.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame to fall back to if matched data is unavailable or not selected.
        session_key (str): Unique key used for the Streamlit radio widget to store the user's selection in session state.
        default_to_matched (bool): If True and matched data exists, the matched dataset is selected by default.
        label_prefix (str): Label text shown next to the radio widget.
    
    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing the selected DataFrame and a descriptive label indicating which dataset was chosen and its row count (e.g., "✅ Matched Data (123 rows)" or "📊 Original Data (456 rows)").
    """
    has_matched = (
        st.session_state.get("is_matched", False)
        and st.session_state.get("df_matched") is not None
    )

    if has_matched:
        col1, _ = st.columns([2, 1])
        with col1:
            data_source = st.radio(
                label_prefix,
                ["📊 Original Data", "✅ Matched Data (จาก PSM)"],
                index=1 if default_to_matched else 0,
                horizontal=True,
                key=session_key,
            )

        if "✅" in data_source:
            selected_df = st.session_state.df_matched.copy()
            label = f"✅ Matched Data ({len(selected_df)} rows)"
        else:
            selected_df = df
            label = f"📊 Original Data ({len(df)} rows)"
    else:
        selected_df = df
        label = f"📊 Original Data ({len(df)} rows)"

    return selected_df, label