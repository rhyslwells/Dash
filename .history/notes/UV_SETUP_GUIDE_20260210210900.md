# Environment Setup Guide with `uv`

## Quick Start

### Step 1: Install `uv` (if not already installed)
```bash
pip install uv
```

Or download standalone from: https://github.com/astral-sh/uv/releases

### Step 2: Set up the project environment

Navigate to the project directory and run:

```bash
cd c:\Users\RhysL\Desktop\Dash

# Create virtual environment and sync dependencies
uv venv
uv sync
```

### Step 3: Activate the environment

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

### Step 4: Run the application

```bash
python 1_example.py
```

The application will start on `http://127.0.0.1:8050/`

---

## What `uv` Does

| Command | Purpose |
|---------|---------|
| `uv venv` | Creates a virtual environment in `.venv/` folder |
| `uv sync` | Installs all dependencies from `pyproject.toml` |
| `uv pip install <package>` | Install additional packages |
| `uv pip freeze` | List installed packages |

---

## Troubleshooting

### If `uv` command not found:
```bash
pip install uv
```

### If PowerShell execution policy blocks activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the activation script again.

### If dependencies don't install:
```bash
uv sync --force  # Force reinstall all dependencies
```

### Test the installation:
```python
# Open Python and test
python -c "import dash; import sklearn; import pandas; print('✅ All packages installed!')"
```

---

## Project Dependencies (from pyproject.toml)

- **dash** ≥ 2.14.0 - Web framework
- **dash-bootstrap-components** ≥ 1.4.0 - Bootstrap styling
- **plotly** ≥ 5.0.0 - Interactive visualizations
- **pandas** ≥ 1.0.0 - Data manipulation
- **scikit-learn** ≥ 1.0.0 - Machine learning
- **numpy** ≥ 1.20.0 - Numerical computing
- **nbformat** ≥ 5.10.4 - Jupyter notebook support

---

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Dash Documentation](https://dash.plotly.com/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
