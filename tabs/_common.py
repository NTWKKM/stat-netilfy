from typing import List, Tuple  # Or use built-in: list, tuple
import pandas as pd

# ==========================================
# 🎨 UNIFIED COLOR PALETTE
# ==========================================
# All reports use this consistent teal/green color scheme
REPORT_COLORS = {
    "primary": "#218084",      # Teal-500: Main accent (titles, borders)
    "primary_light": "#50b8c6", # Teal-300: Highlights, hover states
    "primary_dark": "#1a6473",  # Teal-700: Darker accents
    "success": "#218084",       # Teal-500: Success/positive metrics
    "danger": "#c0152f",        # Red-500: Risk/negative metrics
    "warning": "#a84b2f",       # Orange-500: Warnings/cautions
    "neutral": "#62676c",       # Slate-500: Neutral/secondary text
    "bg_light": "#fcfcf9",      # Cream-50: Light background
    "bg_surface": "#fffffe",    # Cream-100: Surface background
    "border": "#5e5240",        # Brown-600 at 20% opacity in CSS
    "text_primary": "#134252",  # Slate-900: Main text
    "text_secondary": "#62676c" # Slate-500: Secondary text
}

def get_color_palette():
    """
    Returns the unified color palette for consistent styling across all report tabs.
    
    Returns:
        dict: Color palette with keys for primary, success, danger, warning, neutral colors
              and background/border colors.
    """
    return REPORT_COLORS.copy()

def get_pos_label_settings(df: pd.DataFrame, col_name: str) -> tuple[list[str], int]:
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
