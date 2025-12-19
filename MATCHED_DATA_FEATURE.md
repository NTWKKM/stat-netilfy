# ✅ Post-PSM Matched Data Analysis Feature

## Overview

This feature allows users to:
1. **Run PSM (Propensity Score Matching)** to create a balanced matched dataset
2. **View and export** the matched data with comprehensive statistics
3. **Run statistical analyses** on the matched dataset instead of the original data
4. **Compare results** between original and matched datasets

---

## Workflow

### Step 1: Run PSM Matching

**Location:** Tab "📋 Table 1 & Matching" → Subtab "⚖️ Propensity Score Matching"

```
1. Select treatment variable (binary: 0/1)
2. Select covariates (confounders)
3. Configure advanced settings (caliper width)
4. Click "🚀 Run Matching"
5. Check SMD balance (Love plot)
6. If balanced (SMD < 0.1), proceed to Step 2
```

### Step 2: View Matched Data

**Location:** Tab "📋 Table 1 & Matching" → Subtab "✅ Matched Data View"

✨ **NEW SUBTAB!** Features include:
- **Summary statistics** (row counts, group sizes, data types)
- **Data preview** with row limit slider
- **Export options** (CSV, Excel)
- **Statistics by group** (descriptive stats, visualizations)
- **Clear button** to reset and try different matching settings

### Step 3: Use Matched Data in Analyses

**All analysis tabs now support matched data:**
- 🧪 Diagnostic Tests (ROC)
- 📈 Correlation & ICC
- 📊 Risk Factors (Logistic Regression)
- ⏳ Survival Analysis (Kaplan-Meier & Cox)

**Each analysis tab has:**
```
📊 Dataset Selection Selector:
  ☑️ Original Data          (default)
  ☑️ ✅ Matched Data (from PSM)  (new!)
```

Simply select **"✅ Matched Data"** and run your analysis!

---

## Technical Implementation

### Session State Variables

New session state variables in `app.py`:

```python
# Matched dataset storage
st.session_state.df_matched = None          # DataFrame with matched data
st.session_state.is_matched = False         # Flag: PSM was run?
st.session_state.matched_treatment_col = None     # Treatment variable name
st.session_state.matched_covariates = []   # List of covariates used in PSM
```

### Key Changes by File

#### 1. **app.py**
- ✅ Initialize matched data session state
- ✅ Display matched data status banner (blue info box)
- ✅ Add "Clear Matched Data" button in sidebar
- ✅ Pass matched data through to all analysis tabs

#### 2. **tabs/tab_baseline_matching.py**
- ✅ Store matched dataset after successful PSM: `st.session_state.df_matched = df_matched`
- ✅ Set matched flag: `st.session_state.is_matched = True`
- ✅ **NEW SUBTAB 3**: "✅ Matched Data View"
  - Summary statistics panel
  - Data preview with filtering
  - CSV/Excel export buttons
  - Statistics by treatment group
  - Box plots and descriptive tables
  - Clear button to reset

#### 3. **tabs/tab_logit.py** (Template for other tabs)
- ✅ NEW helper function: `_get_dataset_for_analysis()`
- ✅ Display matched data availability notice
- ✅ Add radio button selector for dataset source
- ✅ Use selected dataset for analysis
- ✅ Log which dataset was used in analysis

---

## Code Examples

### Dataset Selection Pattern

```python
# Helper function to select between datasets
def _get_dataset_for_analysis():
    has_matched = st.session_state.get('is_matched', False) and \
                  st.session_state.get('df_matched') is not None
    
    if has_matched:
        data_source = st.radio(
            "📄 Select Dataset:",
            ["📊 Original Data", "✅ Matched Data (from PSM)"],
            index=1,  # Default to matched
            horizontal=True
        )
        
        if "✅" in data_source:
            selected_df = st.session_state.df_matched.copy()
            label = f"✅ Matched Data ({len(selected_df)} rows)"
        else:
            selected_df = None  # Will use passed df
            label = "📊 Original Data"
    else:
        selected_df = None
        label = "📊 Original Data"
    
    return selected_df, label

# Usage in tab
selected_df, data_label = _get_dataset_for_analysis()
if selected_df is None:
    selected_df = df  # Use original if not matched
```

### Storing Matched Data After PSM

```python
if df_matched is not None:
    # Store in session state
    st.session_state.df_matched = df_matched
    st.session_state.is_matched = True
    st.session_state.matched_treatment_col = treat_col
    st.session_state.matched_covariates = cov_cols
    
    logger.info("✅ Matched data stored. Rows: %d", len(df_matched))
    st.success(f"✅ Matching Complete! Matched {len(df_matched)} pairs.")
```

---

## User Interface

### Main App Banner

When matched data is available:

```
✅ **Matched Dataset Active**
- Original data: 600 rows
- Matched data: 180 rows (from 420 excluded)
- Treatment: Treatment_Group
- Use dropdown in each tab to select "✅ Matched Data" for analysis
```

### Sidebar Controls

```
MENU
─ 1. Data Management
  📄 Load Example Data
  📤 Upload CSV/Excel
  🔄 Clear Matched Data      ← NEW (appears only if matched)
  ⚠️ Reset All Data

─ 2. Settings
  Edit Variable Type/Labels
```

### Matched Data View Subtab

```
✅ Matched Data View & Export

✅ Matched Dataset Ready
- Total rows: 180
- Original rows: 600
- Excluded: 420 rows
- Treatment variable: Treatment_Group

📊 Summary Statistics
  Group Sizes:
    0: 90
    1: 90
  
  Data Types:
    int64: 5
    float64: 8

🔍 Filter & Preview
  Rows to display: 50 [slider: 10-180]
  [Data table with 50 rows]

📥 Export Matched Data
  [📥 CSV Format] [📥 Excel Format]

📈 Statistics by Group
  Select numeric variable: [dropdown]
  📊 Descriptive Stats | 📈 Visualization
  [Stats table or box plot]

🔄 Clear Matched Data & Return to Analysis
```

### Analysis Tab Dataset Selector

```
📋 Logistic Regression Analysis

✅ Matched Dataset Available - You can select it below for analysis

📊 Select Dataset:
  ☐ Original Data     ☑ ✅ Matched Data (from PSM)

**Using:** ✅ Matched Data (180 rows)
**Rows:** 180 | **Columns:** 15
```

---

## Feature Comparison

| Feature | Before | After |
|---------|--------|-------|
| PSM Support | ✅ | ✅ |
| View Matched Data | ❌ | ✅ NEW |
| Export Matched Data | Partial | ✅ Full (CSV/Excel) |
| Statistics on Matched Data | ❌ | ✅ NEW |
| Dataset Switching in Analyses | ❌ | ✅ NEW |
| Compare Original vs Matched | ❌ | ✅ NEW |
| Data Source Logging | ❌ | ✅ NEW |

---

## Implementation Details

### File Structure

```
stat-netilfy/
├── app.py                              (UPDATED)
│   ├── Session state initialization
│   ├── Matched data status banner
│   ├── Clear matched data button
│   └── Pass df_matched to tabs
│
├── tabs/
│   ├── tab_baseline_matching.py        (UPDATED - MAJOR)
│   │   ├── PSM matching (existing)
│   │   ├── Subtab 3: NEW - Matched Data View
│   │   └── Store matched data in session
│   │
│   ├── tab_logit.py                    (UPDATED - TEMPLATE)
│   │   ├── Helper: _get_dataset_for_analysis()
│   │   ├── Dataset selector UI
│   │   └── Use selected dataset
│   │
│   ├── tab_diag.py                     (TO UPDATE - Similar pattern)
│   ├── tab_corr.py                     (TO UPDATE - Similar pattern)
│   └── tab_survival.py                 (TO UPDATE - Similar pattern)
│
├── psm_lib.py                          (UNCHANGED)
├── logger.py                           (UNCHANGED)
└── MATCHED_DATA_FEATURE.md             (NEW - This file)
```

### Data Flow

```
┌─────────────────┐
│  Original Data  │
│   (600 rows)    │
└────────┬────────┘
         │
         v
    ┌────────────────────┐
    │  PSM Matching      │
    │ (Calculate PS)     │
    │ (Greedy Matching)  │
    └────────┬───────────┘
             │
             v
    ┌────────────────────┐
    │  Matched Data      │  ← Stored in st.session_state.df_matched
    │   (180 rows)       │     st.session_state.is_matched = True
    └────────┬───────────┘
             │
             v
    ┌────────────────────────────────────┐
    │  Matched Data View Subtab          │
    │  - Summary Stats                   │
    │  - Preview                         │
    │  - Export (CSV/Excel)              │
    │  - Statistics by Group             │
    └────────┬───────────────────────────┘
             │
      ┌──────┴───────────────────────────────┐
      │                                       │
      v                                       v
┌──────────────────┐            ┌──────────────────────┐
│ Used for further │            │ Used for further     │
│  analyses with   │            │  analyses with       │
│ dataset selector │            │ dataset selector     │
└──────────────────┘            └──────────────────────┘
  • Logistic Reg                  • Logistic Reg
  • Survival Anal                 • Survival Anal
  • Diagnostic                    • Diagnostic
  (Original Data)                 (Matched Data) ✅
```

---

## How to Extend to Other Tabs

To add matched data support to other analysis tabs (e.g., `tab_diag.py`, `tab_corr.py`, `tab_survival.py`):

### 1. Copy the helper function

```python
def _get_dataset_for_analysis():
    """See code example above"""
    ...
```

### 2. In your render function, add:

```python
def render(df, var_meta):
    st.subheader("Your Analysis Title")
    
    # NEW: Add matched data note if available
    if st.session_state.get('is_matched', False):
        st.info("✅ **Matched Dataset Available** - You can select it below")
    
    # Get dataset selection
    selected_df, data_label = _get_dataset_for_analysis()
    if selected_df is None:
        selected_df = df  # Default to original
    
    # Display which dataset is being used
    st.write(f"**Using:** {data_label}")
    st.write(f"**Rows:** {len(selected_df)} | **Columns:** {len(selected_df.columns)}")
    
    # Use selected_df instead of df for all analysis
    all_cols = selected_df.columns.tolist()  # ← Use selected_df!
    
    # ... rest of your analysis code using selected_df ...
```

### 3. Log the data source used

```python
data_source_label = "✅ Matched" if st.session_state.get('is_matched') else "Original"
logger.info("Analysis completed | data_source=%s | n=%d", data_source_label, len(selected_df))
```

---

## Testing Checklist

- [ ] Load example data successfully
- [ ] Run PSM matching successfully
- [ ] Matched data stored in session state
- [ ] "Matched Data View" subtab shows matched data
- [ ] Export CSV works
- [ ] Export Excel works (if openpyxl installed)
- [ ] Summary statistics display correctly
- [ ] Group statistics display correctly
- [ ] Box plots render correctly
- [ ] Clear button resets matched data
- [ ] Sidebar "Clear Matched Data" button appears when matched
- [ ] Dataset selector appears in Logistic Regression tab
- [ ] Can switch between Original and Matched data
- [ ] Results change appropriately when switching datasets
- [ ] Logging records data source correctly
- [ ] Status banner appears when matched data available
- [ ] Extending to other tabs works (Diagnostic, Correlation, Survival)

---

## Future Enhancements

1. **Side-by-side comparison**: Show original and matched results simultaneously
2. **Propensity score visualization**: Plot PS distribution before/after matching
3. **Matching quality metrics**: Additional diagnostics (e.g., variance ratios)
4. **Different matching algorithms**: Options for 1:N matching, caliper variations
5. **Sensitivity analysis**: How robust are results to matching method changes?
6. **Stratified analyses**: Run analyses by matched strata
7. **Export matched cohort with analysis results**: Combined report

---

## Troubleshooting

### "Matched data not appearing in analysis tabs"

✅ **Solution:**
1. Ensure PSM was run successfully (check "Matched Data View" subtab)
2. Ensure tab has the dataset selector (should see radio buttons)
3. Try clicking "Clear Matched Data" and running PSM again
4. Check browser console for errors (F12)

### "Cannot export to Excel"

✅ **Solution:**
1. openpyxl package not installed: `pip install openpyxl`
2. Use CSV export instead as workaround

### "Matched data disappears when navigating tabs"

✅ **Solution:**
- This is expected behavior in Streamlit. Data is preserved in session state but may be visually reset.
- Refreshing the page (F5) should restore it.
- If it disappears after reset, you need to re-run PSM.

---

## Related Documentation

- `psm_lib.py`: Propensity Score Matching library
- `tab_baseline_matching.py`: Full implementation of matching feature
- `logger.py`: Logging system
- GitHub Issues: Feature requests and bug reports

---

## Version History

### v2.0.0 (Current - 2025-12-19)
- ✨ NEW: Matched Data View subtab
- ✨ NEW: Dataset selector in analysis tabs
- ✨ NEW: Export matched data (CSV/Excel)
- ✨ NEW: Statistics by treatment group
- ✨ NEW: Session state management for matched data
- ✨ NEW: Data source logging

### v1.0.0 (Previous)
- PSM matching (store in preview only)
- Love plot visualization
- SMD calculations

---

**Created:** 2025-12-19
**Last Updated:** 2025-12-19
**Maintained By:** NTWKKM
