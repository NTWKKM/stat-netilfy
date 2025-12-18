# 🤓 Color System Testing Guide

## Overview

This guide explains how to test the unified color system implementation and verify that all modules render with consistent colors.

---

## Quick Start

### Option 1: Run Streamlit Color Tests (Visual)

```bash
streamlit run test_color_system.py
```

This launches an interactive dashboard where you can:
- 🎨 View the complete color palette
- 📋 Test Table 1 rendering with colors
- 📊 Test Logistic Regression with colors
- ⚖️ Test PSM Love Plot with colors
- ♿ Verify accessibility compliance

### Option 2: Run Unit Tests (Automated)

```bash
pytest tests/test_color_palette.py -v
```

This runs automated tests to verify:
- Color palette structure
- Hex value formatting
- Required colors present
- Module imports
- Accessibility standards

---

## Detailed Testing

### Test Suite 1: Color Palette Validation

**File**: `tests/test_color_palette.py` > `TestColorPalette`

#### Tests

| Test | Verifies |
|------|----------|
| `test_palette_exists()` | Palette function returns a non-empty dict |
| `test_required_colors_present()` | All 11 required color keys exist |
| `test_color_hex_format()` | All colors are valid hex format (#RRGGBB) |
| `test_color_uniqueness()` | Different purposes have different colors |
| `test_specific_color_values()` | Exact hex values match expected |
| `test_color_accessibility()` | Colors meet WCAG standards |
| `test_palette_consistency()` | Multiple calls return identical palette |
| `test_no_color_typos()` | No obvious typos in color definitions |

**Run**:
```bash
pytest tests/test_color_palette.py::TestColorPalette -v
```

---

### Test Suite 2: Module Integration

**File**: `tests/test_color_palette.py` > `TestColorUsageInModules`

#### Tests

| Test | Verifies |
|------|----------|
| `test_table_one_imports_palette()` | table_one.py imports get_color_palette |
| `test_psm_lib_imports_palette()` | psm_lib.py imports get_color_palette |
| `test_logic_imports_palette()` | logic.py imports get_color_palette |

**Run**:
```bash
pytest tests/test_color_palette.py::TestColorUsageInModules -v
```

---

### Test Suite 3: HTML/CSS Integration

**File**: `tests/test_color_palette.py` > `TestColorHtmlIntegration`

#### Tests

| Test | Verifies |
|------|----------|
| `test_primary_color_in_css()` | Primary colors are correct hex values |
| `test_danger_color_for_alerts()` | Danger color is bright red (#ff5459) |
| `test_text_contrast_ratios()` | Text colors have proper contrast |

**Run**:
```bash
pytest tests/test_color_palette.py::TestColorHtmlIntegration -v
```

---

## Visual Testing (Streamlit)

### Tab 1: Color Palette Reference

**What to verify**:
- 📋 Color swatches are displayed
- 📄 Each color shows hex value
- 🤖 Table displays all 11 colors with usage descriptions

**Expected colors**:
```
Primary Colors:
- primary: #218084 (teal)
- primary_dark: #134252 (dark teal)

Status Colors:
- danger: #ff5459 (red)
- warning: #f39c12 (orange)
- success: #218084 (teal)
- info: #7f8c8d (gray)

Neutral Colors:
- text: #2c3e50 (charcoal)
- text_secondary: #7f8c8d (gray)
- border: #e0e0e0 (light gray)
- background: #f4f6f8 (light blue)
- surface: #ffffff (white)
```

### Tab 2: Table 1 Test

**What to verify**:
- 📋 Table headers are dark teal (#134252) with white text
- 📢 Significant p-values (p < 0.05) are red (#ff5459) with asterisk
- 🔗 Footer links are teal (#218084)
- 📋 Data cells have alternating backgrounds

**Expected appearance**:
```
┌───────────────┐  ← Dark teal header
│ Characteristic  │
├───────────────┤
│ Age (years)     │
│ BMI (kg/m²)    │
│ Gender          │
│ ❤️ 0.032*      │  ← Red significant p-value
└───────────────┘
```

### Tab 3: Logistic Regression Test

**What to verify**:
- 📋 Table headers are dark teal (#134252)
- 📢 Significant p-values are red (#ff5459) with light red background
- 📋 Sheet headers use light teal (#e8f4f8) background
- 🔗 Footer text is gray (#7f8c8d)

**Expected appearance**:
```
Outcome: disease_status (Total n=100)
┌───────────────┐  ← Dark teal header
│ ...
├───────────────┤
│ Demographics        │  ← Light teal background
│ Age        0.024* │  ← Significant (red)
│ BMI        0.156  │  ← Not significant
└───────────────┘
```

### Tab 4: PSM Love Plot Test

**What to verify**:
- 🕐 Unmatched points are red (#ff5459) circles
- 🕒 Matched points are teal (#218084) diamonds
- 📋 SMD = 0.1 reference line is gray dashed
- 📊 Plot title is visible

**Expected appearance**:
```
Covariate Balance (Love Plot)
│  
│  ● Red circles (unmatched)
│  ◊ Teal diamonds (matched)
│  
│      │---│  ← Reference line at SMD=0.1
│  Age ◊
│  BMI ◊
└─────────────────────────────▸
```

### Tab 5: Accessibility Information

**What to verify**:
- ♿ Color contrast ratios are listed
- ✅ All colors meet WCAG AA (4.5:1 minimum)
- 🔍 Dark teal meets AAA (7:1)
- 📋 Explanatory text about accessibility

**Expected table**:
```
| Color Combination | Contrast | WCAG AA |
|-------------------|----------|----------|
| Dark Teal + White | 7.5:1    | ✅ Pass |
| Teal + White      | 5.2:1    | ✅ Pass |
| Red + White       | 4.8:1    | ✅ Pass |
| Gray + White      | 4.6:1    | ✅ Pass |
```

---

## Test Execution Examples

### Run all tests
```bash
pytest tests/test_color_palette.py -v
```

### Run specific test class
```bash
pytest tests/test_color_palette.py::TestColorPalette -v
```

### Run specific test function
```bash
pytest tests/test_color_palette.py::TestColorPalette::test_color_hex_format -v
```

### Run with detailed output
```bash
pytest tests/test_color_palette.py -vv -s
```

### Run and generate coverage report
```bash
pytest tests/test_color_palette.py --cov=tabs._common --cov-report=html
```

---

## Manual Visual Inspection Checklist

- [ ] **Headers**: Dark teal (#134252) with white text
- [ ] **Links**: Teal (#218084) that darken on hover
- [ ] **Significant values**: Red (#ff5459) with bold font
- [ ] **Borders**: Light gray (#e0e0e0)
- [ ] **Text**: Charcoal (#2c3e50) readable on white
- [ ] **Section headers**: Light teal background (#e8f4f8)
- [ ] **Footer**: Gray text (#7f8c8d) with gray line separator
- [ ] **Hover effects**: Colors respond to interaction
- [ ] **Printability**: Colors are printer-friendly
- [ ] **Mobile**: Colors display correctly on small screens

---

## Troubleshooting

### Colors look different than expected

1. **Check display settings**: Monitor color accuracy may vary
2. **Clear browser cache**: `Ctrl+Shift+Del` (Windows/Linux) or `Cmd+Shift+Delete` (Mac)
3. **Restart Streamlit**: `Ctrl+C` and re-run
4. **Verify palette import**: Check if `get_color_palette()` is used

### Tests are failing

1. **Check Python version**: Requires Python 3.8+
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run from correct directory**: Must be project root
4. **Check file permissions**: Ensure files are readable

### Specific module not using colors

1. **Verify import**: Check if `from tabs._common import get_color_palette` exists
2. **Check CSS generation**: Look for `COLORS = get_color_palette()`
3. **Verify HTML template**: Look for `{COLORS['color_name']}` in f-strings
4. **Check for hardcoded colors**: Search for `#` followed by hex digits

---

## CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
name: Color System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/test_color_palette.py -v
```

---

## Performance Notes

- Color palette generation is lightweight (~1ms)
- No external API calls required
- Safe to call multiple times per request
- All colors defined as constants

---

## Next Steps

After testing:

1. ✅ Verify all tests pass locally
2. 🤓 Review color rendering in Streamlit app
3. 📝 Check accessibility compliance
4. 📝 Create PR with test results
5. 🌟 Merge to main branch

---

## Questions?

Refer to:
- [COLOR_SYSTEM.md](./COLOR_SYSTEM.md) - Color palette documentation
- [tabs/_common.py](./tabs/_common.py) - Source palette definition
- Individual module files (table_one.py, logic.py, etc.) - Usage examples

---

**Last Updated**: December 18, 2025  
**Version**: 1.0  
**Status**: Active ✅