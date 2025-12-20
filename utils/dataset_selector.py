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
    Select between original and matched datasets.
    
    Parameters:
        df: Original DataFrame
        session_key: Unique key for the radio button (e.g., 'correlation_data_source')
        default_to_matched: If True, default to Matched Data when available
        label_prefix: Label text for the radio button
    
    Returns:
        (selected_df, label_str): Selected DataFrame and descriptive label
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
