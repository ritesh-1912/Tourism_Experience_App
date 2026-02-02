# 🚀 Quick Start Guide - Tourism Analytics Project

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Project](#running-the-project)
4. [Project Structure](#project-structure)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
  - Check version: `python --version` or `python3 --version`
  - Download from: https://www.python.org/downloads/

- **pip** (Python package installer)
  - Usually comes with Python
  - Check version: `pip --version` or `pip3 --version`

---

## Installation

### Step 1: Navigate to Project Directory

```bash
cd Tourism_Analytics_Project
```

### Step 2: Install Required Packages

**Option A: Using pip (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Using pip3**
```bash
pip3 install -r requirements.txt
```

**Option C: Install packages individually if needed**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly openpyxl joblib
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, streamlit; print('All packages installed successfully!')"
```

---

## Running the Project

### Option 1: Run Complete Pipeline (Recommended for First Time)

#### Step 1: Data Preprocessing
```bash
python src/data_preprocessing.py
```

This will:
- Load all 9 Excel files from `data/` directory
- Clean and merge datasets
- Engineer features
- Save processed data to `data/processed_data.csv`

**Expected Output**: 
```
Loading data files...
Data loaded successfully!
================================================================================
CLEANING DATA
================================================================================
...
Processed data saved to data/processed_data.csv
```

#### Step 2: Run Streamlit Application
```bash
streamlit run app.py
```

This will:
- Load processed data and trained models
- Launch web application
- Open browser automatically at `http://localhost:8501`

**Expected Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Option 2: Run Only the Streamlit App (If Models Already Trained)

```bash
streamlit run app.py
```

### Option 3: Train Models Separately

If you want to retrain models:

```bash
cd src
python models.py
```

Or use Python interactively:

```python
import sys
sys.path.append('src')
from models import RatingPredictor, VisitModeClassifier, AttractionRecommender
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Train models (see models.py for details)
# ...
```

---

## Project Structure

```
Tourism_Analytics_Project/
│
├── 📁 data/                       # Data files
│   ├── City.xlsx
│   ├── Continent.xlsx
│   ├── Country.xlsx
│   ├── Item.xlsx
│   ├── Mode.xlsx
│   ├── Region.xlsx
│   ├── Transaction.xlsx
│   ├── Type.xlsx
│   ├── User.xlsx
│   └── processed_data.csv        # Generated after preprocessing
│
├── 📁 src/                        # Source code
│   ├── data_preprocessing.py     # Data cleaning & feature engineering
│   └── models.py                 # ML models
│
├── 📁 models/                     # Trained models (generated)
│   ├── rating_predictor.pkl
│   ├── visitmode_classifier.pkl
│   ├── recommender.pkl
│   └── model_performance.json
│
├── 📁 visualizations/            # Generated charts (after running preprocessing)
│   ├── rating_distribution.png
│   ├── visit_mode_distribution.png
│   └── ...
│
├── 📁 notebooks/                 # Jupyter notebooks for EDA
│   └── 01_EDA_and_Analysis.ipynb
│
├── 📄 app.py                     # Main Streamlit application
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Comprehensive documentation
├── 📄 PROJECT_REPORT.md          # Detailed project report
└── 📄 QUICK_START.md            # This file
```

---

## Using the Streamlit Application

Once the app is running, you'll see 5 main pages:

### 1. 📊 Dashboard
- View overall statistics
- Explore data visualizations
- Understand key metrics

### 2. 🔮 Rating Prediction
1. Select existing user or create new user profile
2. Choose an attraction
3. Select visit details (year, month, mode)
4. Click "Predict Rating"
5. View predicted rating and interpretation

### 3. 🎯 Visit Mode Classification
1. Select a user
2. Choose an attraction
3. Select visit timing
4. Click "Predict Visit Mode"
5. See predicted travel mode and confidence scores

### 4. 🌟 Recommendations
1. Select a user
2. Choose recommendation method:
   - Collaborative Filtering
   - Content-Based
   - Hybrid
3. Set number of recommendations
4. Click "Get Recommendations"
5. View personalized attraction suggestions

### 5. 📈 Insights
- Statistical summaries
- Geographic analysis
- Temporal patterns
- Attraction analytics

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError"

**Problem**: Package not installed
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**:
```bash
pip install pandas
# or install all packages
pip install -r requirements.txt
```

### Issue 2: "FileNotFoundError"

**Problem**: Data files not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/Transaction.xlsx'
```

**Solution**:
- Ensure all 9 Excel files are in the `data/` directory
- Check file names match exactly (case-sensitive)
- Verify you're running from the project root directory

### Issue 3: Streamlit Not Opening in Browser

**Problem**: App starts but browser doesn't open

**Solution**:
- Manually open: `http://localhost:8501` in your browser
- Try a different port: `streamlit run app.py --server.port 8502`
- Check firewall settings

### Issue 4: Port Already in Use

**Problem**:
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the process using port 8501
# On Mac/Linux:
lsof -ti:8501 | xargs kill -9
# On Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue 5: Models Not Found

**Problem**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/rating_predictor.pkl'
```

**Solution**:
Run preprocessing first to generate models:
```bash
python src/data_preprocessing.py
```

Then the models will be created in the `models/` directory.

### Issue 6: Memory Error

**Problem**: Not enough RAM

**Solution**:
- Close other applications
- Use data sampling (already implemented in training script)
- Increase system swap space

### Issue 7: Slow Performance

**Problem**: App is slow

**Solution**:
- The first load caches data and models (one-time delay)
- Subsequent operations should be fast
- Consider using a smaller data sample for testing

---

## Additional Commands

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/01_EDA_and_Analysis.ipynb
```

### Generate Visualizations
```bash
python -c "
import sys
sys.path.append('src')
from data_preprocessing import TourismDataProcessor
processor = TourismDataProcessor()
processor.load_data()
# Generate visualizations
"
```

### Check Model Performance
```bash
python -c "
import json
with open('models/model_performance.json', 'r') as f:
    perf = json.load(f)
    print(json.dumps(perf, indent=2))
"
```

---

## Tips for Best Experience

1. **First Run**: Always run preprocessing before launching the app
2. **Data Updates**: If you update Excel files, re-run preprocessing
3. **Browser**: Use Chrome, Firefox, or Edge for best compatibility
4. **Screen Size**: Use full screen for optimal visualization
5. **Explore Features**: Try all 5 pages to understand full capabilities

---

## Getting Help

If you encounter issues not covered here:

1. Check the main README.md for detailed documentation
2. Review PROJECT_REPORT.md for technical details
3. Examine error messages carefully
4. Verify all prerequisites are installed
5. Check Python version compatibility

---

## Quick Reference

### Essential Commands
```bash
# Install packages
pip install -r requirements.txt

# Run preprocessing
python src/data_preprocessing.py

# Launch app
streamlit run app.py

# Run notebook
jupyter notebook

# Check versions
python --version
pip list
```

### File Locations
- **Data**: `data/*.xlsx`
- **Models**: `models/*.pkl`
- **Source**: `src/*.py`
- **App**: `app.py`
- **Visualizations**: `visualizations/*.png`

---

## Next Steps

After running the application:

1. ✅ Explore the Dashboard to understand the data
2. ✅ Try Rating Prediction with different user profiles
3. ✅ Test Visit Mode Classification
4. ✅ Get personalized Recommendations
5. ✅ Analyze Insights for business understanding

---

## Support

For additional support or questions:
- Review the comprehensive README.md
- Check the detailed PROJECT_REPORT.md
- Examine code comments in source files

---

**Happy Analyzing! 🎉**

Last Updated: January 2026
