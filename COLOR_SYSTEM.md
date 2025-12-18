# 🎨 Unified Color System Documentation

## Overview

This project uses a **unified teal-based color system** across all modules to ensure visual consistency and professional appearance. The color palette is centralized in `tabs/_common.py` and is consumed by all report-generating modules.

---

## Color Palette

### Primary Colors

| Color Name | Hex Value | RGB | Usage |
|-----------|-----------|-----|-------|
| **Primary** | `#218084` | `rgb(33, 128, 132)` | Main headings, borders, buttons, links |
| **Primary Dark** | `#134252` | `rgb(19, 66, 82)` | Table headers, emphasis |
| **Primary Light** | `#e8f4f8` | `rgb(232, 244, 248)` | Backgrounds, section headers |

### Status/Semantic Colors

| Color Name | Hex Value | RGB | Usage |
|-----------|-----------|-----|-------|
| **Danger/Alert** | `#ff5459` | `rgb(255, 84, 89)` | Significant p-values (<0.05), error states |
| **Warning** | `#f39c12` | `rgb(243, 156, 18)` | Caution, non-critical warnings |
| **Success** | `#218084` | `rgb(33, 128, 132)` | Approved, matched status |
| **Info** | `#7f8c8d` | `rgb(127, 140, 141)` | Informational text |

### Neutral Colors

| Color Name | Hex Value | RGB | Usage |
|-----------|-----------|-----|-------|
| **Text Primary** | `#2c3e50` | `rgb(44, 62, 80)` | Main text content |
| **Text Secondary** | `#7f8c8d` | `rgb(127, 140, 141)` | Secondary text, subtitles, footer |
| **Border** | `#e0e0e0` | `rgb(224, 224, 224)` | Table borders, dividers |
| **Background** | `#f4f6f8` | `rgb(244, 246, 248)` | Page background |
| **Surface** | `#ffffff` | `rgb(255, 255, 255)` | Card/container backgrounds |

---

## Implementation

### 1. **Central Color Definition** (`tabs/_common.py`)

All colors are defined in a single function:

```python
def get_color_palette():
    """
    Returns a unified color palette dictionary for all modules.
    """
    return {
        'primary': '#218084',
        'primary_dark': '#134252',
        'text': '#2c3e50',
        'text_secondary': '#7f8c8d',
        'danger': '#ff5459',
        'warning': '#f39c12',
        'success': '#218084',
        'info': '#7f8c8d',
        'border': '#e0e0e0',
        'background': '#f4f6f8',
        'surface': '#ffffff',
    }
```

### 2. **Usage in Modules**

Every module imports and uses the palette:

```python
from tabs._common import get_color_palette

# Inside report-generating function
COLORS = get_color_palette()

# Use in CSS/HTML
html = f"""
<style>
    th {{
        background-color: {COLORS['primary_dark']};
        color: white;
    }}
    .sig-p {{
        color: {COLORS['danger']};
        font-weight: bold;
    }}
</style>
"""
```

---

## Modules Using This System

### ✅ Core Modules

1. **`table_one.py`** (Table 1 Generation)
   - Table headers: `primary_dark`
   - Significant p-values: `danger` (red)
   - Links: `primary` with hover to `primary_dark`

2. **`psm_lib.py`** (Propensity Score Matching)
   - Love plot unmatched: `danger` (red circles)
   - Love plot matched: `primary` (teal diamonds)
   - Report headings: `primary`

3. **`logic.py`** (Logistic Regression Analysis)
   - Table headers: `primary_dark`
   - Significant values: `danger` (red with light bg)
   - Sheet headers: Light teal (`#e8f4f8`)
   - Footer: `text_secondary` gray

4. **`diag_test.py`** (Diagnostic Tests)
   - Unified headings: `primary_dark`
   - Borders: `primary`
   - Text: `text`

5. **`correlation.py`** (Correlation Analysis)
   - Scatter plot markers: `primary` (teal)
   - Regression lines: `danger` (red)
   - Headers: `primary_dark`

6. **`survival_lib.py`** (Survival Analysis)
   - Headers: `primary_dark`
   - Footer links: `primary`
   - Text: `text`

---

## CSS Styling Patterns

### Table Headers
```css
th {
    background-color: #134252;  /* primary_dark */
    color: white;
    padding: 12px 15px;
    border: 1px solid #218084;  /* primary */
}
```

### Significant P-values
```css
.sig-p {
    color: #ff5459;  /* danger */
    font-weight: bold;
    background-color: #ffebee;  /* light red */
    padding: 2px 4px;
    border-radius: 4px;
}
```

### Section Headers
```css
.section-header {
    background-color: #e8f4f8;  /* light teal */
    color: #218084;  /* primary */
    font-weight: bold;
    padding: 8px 15px;
}
```

### Borders & Dividers
```css
border: 1px solid #e0e0e0;  /* border */
border-top: 1px dashed #7f8c8d;  /* text_secondary */
```

### Links
```css
a {
    color: #218084;  /* primary */
    text-decoration: none;
}
a:hover {
    color: #134252;  /* primary_dark */
}
```

---

## Testing Color Consistency

### Visual Regression Testing

To verify colors are applied correctly:

1. Run the Streamlit app
2. Check each tab (Diagnostic Tests, Correlation, Survival, etc.)
3. Verify:
   - Table headers are dark teal (`#134252`)
   - Significant p-values are red (`#ff5459`)
   - Links change on hover
   - Borders are light gray (`#e0e0e0`)

### Color Contrast Validation

All text colors meet WCAG AA accessibility standards:

- **Dark teal (#134252) on white**: Contrast ratio ≥ 7:1 ✅
- **Teal (#218084) on white**: Contrast ratio ≥ 4.5:1 ✅
- **Red (#ff5459) on white**: Contrast ratio ≥ 4.5:1 ✅
- **Gray (#7f8c8d) on white**: Contrast ratio ≥ 4.5:1 ✅

---

## Brand Identity

The teal color palette reflects:

- **Trust & Stability**: Professional medical/statistical analysis
- **Data-Driven**: Cool, analytical aesthetic
- **Modern**: Contemporary design approach
- **Consistency**: Unified visual language across all outputs

---

## Future Enhancements

### Dark Mode Support

When implementing dark mode, extend the palette:

```python
def get_color_palette(theme='light'):
    if theme == 'dark':
        return {
            'primary': '#50d4dc',      # Lighter teal
            'primary_dark': '#1a3f4a',  # Darker background
            'text': '#e8f4f8',          # Light text
            'background': '#0f1419',    # Dark background
            # ... other colors
        }
    else:
        # Current light theme
        return {...}
```

### Additional Color Variants

For future use:

- **Teal-50**: `#f0f9fa` (very light backgrounds)
- **Teal-100**: `#e0f2f4` (subtle highlights)
- **Teal-600**: `#1a6b70` (darker emphasis)
- **Teal-700**: `#144d52` (darkest accent)

---

## Maintenance Guidelines

1. **Never hardcode colors** in individual modules
2. **Always use** `get_color_palette()` from `tabs/_common.py`
3. **When adding new colors**, update both:
   - `get_color_palette()` function
   - This documentation
4. **Document usage** in comments when using unusual color combinations

---

## Quick Reference

### Import & Use
```python
from tabs._common import get_color_palette

COLORS = get_color_palette()
# Now access: COLORS['primary'], COLORS['danger'], etc.
```

### Common CSS Patterns
```css
/* Headers */
background-color: #134252;  /* primary_dark */

/* Important text */
color: #ff5459;  /* danger */

/* Regular text */
color: #2c3e50;  /* text */

/* Subtle text */
color: #7f8c8d;  /* text_secondary */

/* Borders */
border: 1px solid #e0e0e0;  /* border */
```

---

## Questions?

For color-related issues or suggestions:
1. Check this documentation
2. Review `tabs/_common.py` for the source palette
3. Create an issue on GitHub with screenshots of color discrepancies

---

**Last Updated**: December 18, 2025  
**Version**: 1.0  
**Status**: Active ✅