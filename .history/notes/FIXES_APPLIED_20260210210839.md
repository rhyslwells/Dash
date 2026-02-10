# Dash Application Fixes - Documentation

## Overview
Fixed deprecation issues and setup problems in the Dash logistic regression app (`1_example.py`).

## Issues Found and Fixed

### 1. **Deprecated Dash Imports** ❌ → ✅
**Problem:**
- Code used old import syntax: `dash_html_components`, `dash_core_components`
- These packages were deprecated after Dash 2.0

**Old Code:**
```python
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
```

**Fixed Code:**
```python
import dash.html as html
import dash.dcc as dcc
from dash import Output, Input, State, callback
```

**Why:** Modern Dash (v2.14+) integrates these directly into the main package, reducing dependencies and improving consistency.

---

### 2. **Missing Dependencies in pyproject.toml** ❌ → ✅
**Problem:**
- `scikit-learn` and `numpy` were not listed in dependencies
- Code imports these but they weren't declared, causing potential installation failures

**Old:**
```toml
dependencies = [
    "dash>=2.14.0",
    "dash-bootstrap-components>=1.4.0",
    "plotly",
    "pandas",
    "nbformat>=5.10.4",
]
```

**Fixed:**
```toml
dependencies = [
    "dash>=2.14.0",
    "dash-bootstrap-components>=1.4.0",
    "plotly>=5.0.0",
    "pandas>=1.0.0",
    "scikit-learn>=1.0.0",
    "numpy>=1.20.0",
    "nbformat>=5.10.4",
]
```

**Why:** Ensures `uv` installs all required packages and makes dependencies explicit.

---

### 3. **Hard-coded Relative File Paths** ❌ → ✅
**Problem:**
- Code assumed execution from specific directory: `../data/poverty.csv`
- Would fail if script run from different location
- Not portable across systems

**Old Code:**
```python
poverty_data = pd.read_csv('../data/PovStatsData.csv')
poverty = pd.read_csv('../data/poverty.csv', low_memory=False)
```

**Fixed Code:**
```python
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent
data_dir = project_root / 'data'

poverty_data = pd.read_csv(data_dir / 'PovStatsData.csv')
poverty = pd.read_csv(data_dir / 'poverty.csv', low_memory=False)
```

**Why:** 
- Uses `Path` to dynamically locate data directory relative to script location
- Works regardless of where the script is executed from
- Cross-platform compatible (Windows, Mac, Linux)
- More robust and professional

---

### 4. **Removed Unused Imports** ⚙️
- Removed `pickle` (not used in code)
- Removed `classification_report` (imported but not used)
- Kept `os` import for compatibility

---

## What the Application Does

### Purpose
Demonstrates a complete machine learning pipeline in a Dash web app:

1. **Data Preparation**: Loads World Bank poverty data
2. **Model Training**: Trains logistic regression on GINI index to predict high/low poverty
3. **Interactive Dashboard** with 5 tabs:
   - **Model Performance**: Accuracy, ROC AUC, Confusion Matrix, ROC Curve
   - **Feature Importance**: Displays logistic regression coefficients
   - **Make Predictions**: Let users input values and get predictions
   - **Data Explorer**: Distribution charts and dataset statistics
   - **Project Info**: Links to book, GitHub repo, and data source

### Key Components
- **Model**: Logistic Regression classifier
- **Scaling**: StandardScaler for normalization
- **Data**: World Bank poverty statistics
- **UI Framework**: Dash with Bootstrap styling

---

## Setup Instructions with `uv`

### Prerequisites
- Python 3.8+ installed
- `uv` package installer (`pip install uv` or use standalone binary)

### Installation Steps

1. **Navigate to project directory:**
   ```bash
   cd c:\Users\RhysL\Desktop\Dash
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   uv sync
   ```

3. **Activate virtual environment:**
   - Windows (CMD):
     ```bash
     .venv\Scripts\activate
     ```
   - Windows (PowerShell):
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - Mac/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Run the application:**
   ```bash
   python 1_example.py
   ```

5. **Access the dashboard:**
   - Open browser to `http://127.0.0.1:8050/`
   - `Ctrl+C` to stop the server

---

## Why Use `uv`?

- **Fast**: Written in Rust, significantly faster than pip
- **Reliable**: Better dependency resolution
- **Simple**: Drop-in replacement for pip
- **Project-aware**: Works seamlessly with `pyproject.toml`

---

## Testing the Fix

Run these commands to verify everything works:

```python
# Test imports
python -c "import dash; import dash.html; from sklearn.linear_model import LogisticRegression; print('✅ All imports successful!')"

# Test file paths
python -c "from pathlib import Path; print('Data directory:', Path(__file__).resolve().parent / 'data')"

# Run application
python 1_example.py
```

---

## Files Modified

1. `pyproject.toml` - Added missing dependencies with version specs
2. `1_example.py` - Updated imports and file path handling

## Summary of Changes
- ✅ Updated to modern Dash syntax (v2.14+)
- ✅ Added missing scikit-learn and numpy dependencies
- ✅ Fixed file path handling for cross-platform compatibility
- ✅ Cleaned up unused imports
- ✅ Code now ready for production use
