# Running Python Modules with Proper PYTHONPATH Setup

## Problem
When running Python scripts directly (e.g., `python src/components/data_transformation.py`), you may encounter import errors like:
```
ModuleNotFoundError: No module named 'src'
```
This happens because Python does not treat the current directory as a package root, so relative imports fail.

## Solutions

### 1. Run as a Module from Project Root
Run the script using the `-m` flag from the project root directory (`d:/new_ml_project`):
```bash
python -m src.components.data_transformation
python -m src.components.data_ingestion
```
This tells Python to treat `src` as a package and resolves imports correctly.

### 2. Set PYTHONPATH Environment Variable
Set the `PYTHONPATH` environment variable to the project root directory before running the script:
- On Windows CMD:
```cmd
set PYTHONPATH=d:\new_ml_project
python src\components\data_transformation.py
```
- On PowerShell:
```powershell
$env:PYTHONPATH = "d:\new_ml_project"
python src\components\data_transformation.py
```

### 3. Use Helper Scripts
You can create helper scripts (like `src/run_data_transformation.py`) that set the `PYTHONPATH` internally and run the modules.

## Summary
The recommended approach is to run your scripts as modules using the `-m` flag from the project root directory.

If you need help creating or using helper scripts, please ask.

---

If you have any questions or need further assistance, feel free to ask.
