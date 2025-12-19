from typing import List, Tuple, Dict
import pandas as pd

def get_pos_label_settings(df: pd.DataFrame, col_name: str) -> Tuple[List[str], int]:
    """
    Helper function to get unique values from a column, convert them to strings, 
    sort them, and determine a default index (preferring '1', then '0').

    Handles the case where the column might be empty after dropna.

    Args:
        df: The DataFrame containing the data.
        col_name: The name of the column to process.

    Returns:
        A tuple containing:
        1. A sorted list of unique non-null string values.
        2. The default index for selection (0, or index of '1'/'0').
    """
    # 🟢 NOTE: Need to handle the case where the column might be empty after dropna
    # Convert to string and drop NA values before getting unique values
    unique_vals = [str(x) for x in df[col_name].dropna().unique()]
    unique_vals.sort()
    
    default_idx = 0
    if '1' in unique_vals:
        # Default to '1' if available
        default_idx = unique_vals.index('1')
    elif len(unique_vals) > 0 and '0' in unique_vals:
        # Otherwise, default to '0' if available and there are unique values
        default_idx = unique_vals.index('0')
        
    return unique_vals, default_idx
