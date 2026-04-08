# Tourism Experience Analytics

**A Complete Machine Learning Project for Tourism Data Analysis, Prediction, and Recommendation**

## 📋 Project Overview

This project analyzes tourism data to provide personalized recommendations, predict user satisfaction, and classify travel behavior patterns. It addresses three main objectives:

1. **Regression**: Predict attraction ratings based on user and attraction features
2. **Classification**: Predict visit modes (Business, Family, Couples, Friends, Solo)
3. **Recommendation**: Suggest personalized attractions using collaborative and content-based filtering

## 🎯 Business Use Cases

- **Personalized Recommendations**: Suggest attractions based on user preferences and history
- **Tourism Analytics**: Provide insights into popular attractions and regional trends
- **Customer Segmentation**: Classify users for targeted marketing campaigns
- **Customer Retention**: Boost loyalty through personalized experiences

## 📊 Dataset

The project uses 9 datasets containing information about:

- **Transaction Data**: User visits, ratings, and visit details (52,930 records)
- **User Data**: User demographics and location (33,530 users)
- **Attraction Data**: Tourist attractions and their attributes (30 attractions)
- **Geographic Data**: Cities, countries, regions, and continents
- **Metadata**: Visit modes and attraction types

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd Tourism_Analytics_Project
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files**
   Ensure all 9 Excel files are in the `data/` directory:
   - Transaction.xlsx
   - User.xlsx
   - City.xlsx
   - Continent.xlsx
   - Country.xlsx
   - Region.xlsx
   - Item.xlsx
   - Mode.xlsx
   - Type.xlsx

## 📁 Project Structure

```
Tourism_Analytics_Project/
│
├── data/                          # Data files
│   ├── *.xlsx                     # Original Excel files
│   └── processed_data.csv         # Processed dataset
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning and feature engineering
│   └── models.py                  # ML models (Regression, Classification, Recommendation)
│
├── notebooks/                     # Jupyter notebooks
│   └── 01_EDA_and_Analysis.ipynb # Exploratory Data Analysis
│
├── models/                        # Trained models
│   ├── rating_predictor.pkl       # Rating prediction model
│   ├── visitmode_classifier.pkl   # Visit mode classification model
│   ├── recommender.pkl            # Recommendation system
│   └── model_performance.json     # Model performance metrics
│
├── visualizations/                # Generated plots and charts
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Usage

### 1. Data Preprocessing

Run the preprocessing pipeline to clean and prepare data:

```bash
python src/data_preprocessing.py
```

This will:
- Load all 9 datasets
- Clean and handle missing values
- Merge datasets
- Engineer new features
- Save processed data to `data/processed_data.csv`

### 2. Model Training

Train all machine learning models:

```bash
python -c "
import sys
sys.path.append('src')
from models import RatingPredictor, VisitModeClassifier, AttractionRecommender
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Train models (code continues...)
"
```

Or use the training script (if created separately).

### 3. Run the Streamlit Application

Launch the interactive web dashboard:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🎨 Features

### Streamlit Application

The web application includes 5 main pages:

1. **📊 Dashboard**: Overview of key metrics and visualizations
   - Total visits, users, attractions statistics
   - Rating distribution
   - Visit mode distribution
   - Temporal trends
   - Top attractions

2. **🔮 Rating Prediction**: Predict attraction ratings
   - Select existing users or create new user profiles
   - Choose attractions and visit details
   - Get predicted ratings (1-5 scale)
   - View rating interpretation

3. **🎯 Visit Mode Classification**: Predict travel modes
   - Select user and attraction
   - Predict whether user will travel as Business, Family, Couples, Friends, or Solo
   - View prediction confidence scores
   - Get mode-specific recommendations

4. **🌟 Recommendations**: Personalized attraction suggestions
   - Collaborative filtering (based on similar users)
   - Content-based filtering (based on attraction features)
   - Hybrid recommendations (combination of both)
   - View user's visit history

5. **📈 Insights**: Data analytics and insights
   - Statistical summaries
   - Geographic analysis
   - Temporal patterns
   - Attraction analysis

## 🤖 Machine Learning Models

### 1. Rating Prediction (Regression)

**Models Used:**
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor

**Best Model:** Ridge Regression
- R² Score: 0.7457
- RMSE: 0.4766
- MAE: 0.2858

**Features:**
- User demographics (continent, region, country, city)
- User statistics (average rating, total visits, unique attractions)
- Attraction features (type, location, average rating)
- Temporal features (year, month, quarter, season)
- Visit mode
- Mode-type interaction features

### 2. Visit Mode Classification

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

**Best Model:** Gradient Boosting Classifier
- Accuracy: 99.91%
- Precision: 99.91%
- Recall: 99.91%
- F1-Score: 99.91%

**Features:**
- Same as regression (excluding visit mode as it's the target)

### 3. Recommendation System

**Methods:**
- **Collaborative Filtering**: User-user similarity using cosine similarity
- **Content-Based Filtering**: Attraction features and user preferences
- **Hybrid Approach**: Combination of both methods

**Metrics:**
- User-Item Matrix: 9,288 users × 30 attractions
- Recommendation Accuracy: Based on user ratings

## 📊 Key Insights

1. **Rating Patterns**: Most ratings are concentrated at 4-5 stars, indicating high user satisfaction
2. **Popular Visit Modes**: Couples and Family are the most common visit modes
3. **Seasonal Trends**: Summer months show increased tourism activity
4. **Geographic Distribution**: Users are distributed across multiple continents with varying preferences
5. **Attraction Preferences**: Different user segments prefer different types of attractions

## 🔍 Data Preprocessing Steps

1. **Data Loading**: Load 9 Excel files
2. **Data Cleaning**:
   - Handle missing values in city names and user city IDs
   - Remove placeholder entries (ID = 0 or '-')
3. **Data Merging**: Join all datasets on appropriate keys
4. **Feature Engineering**:
   - User-level aggregations (average rating, total visits, unique attractions)
   - Attraction-level aggregations (average rating, total visits, unique visitors)
   - Temporal features (quarter, season indicators)
   - Interaction features (visit mode × attraction type)
5. **Encoding**: Label encode categorical variables
6. **Scaling**: StandardScaler for numerical features

## 📈 Model Evaluation

### Regression Model Performance
```
Model: Ridge Regression
- R² Score:  0.7457 (74.57% variance explained)
- RMSE:      0.4766 (average error of 0.48 rating points)
- MAE:       0.2858 (average absolute error)
```

### Classification Model Performance
```
Model: Gradient Boosting Classifier
- Accuracy:  99.91%
- Precision: 99.91%
- Recall:    99.91%
- F1-Score:  99.91%

Class-wise Performance:
- Business: 100% accuracy
- Couples:  100% accuracy
- Family:   100% accuracy
- Friends:  100% accuracy
- Solo:     99% accuracy
```

### Recommendation System
- Successfully generates personalized recommendations
- Uses cosine similarity for user-user matching
- Considers user preferences and attraction features
- Provides diverse recommendations across attraction types

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Joblib**: Model serialization
- **OpenPyXL**: Excel file handling

## 📝 Project Deliverables

1. ✅ **Cleaned Dataset**: `data/processed_data.csv`
2. ✅ **Source Code**: All Python scripts in `src/`
3. ✅ **Trained Models**: Saved in `models/` directory
4. ✅ **Streamlit Application**: Interactive web dashboard
5. ✅ **Documentation**: This comprehensive README
6. ✅ **Jupyter Notebook**: EDA and analysis

## 🎓 Skills Demonstrated

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning (Regression, Classification)
- Recommendation Systems (Collaborative & Content-Based Filtering)
- Data Visualization
- Web Application Development (Streamlit)
- Model Evaluation and Selection
- End-to-End ML Project Development

## 🚦 Future Enhancements

1. **Deep Learning Models**: Implement neural networks for better predictions
2. **Real-time Recommendations**: Add online learning capabilities
3. **More Features**: Incorporate user reviews, social data, weather data
4. **A/B Testing**: Test different recommendation strategies
5. **Mobile Application**: Develop mobile version of the dashboard
6. **API Development**: Create REST API for model serving
7. **Advanced Filtering**: Add more sophisticated content-based filtering
8. **Explainability**: Add SHAP or LIME for model interpretability

## 📞 Support

For questions or issues:
1. Check the code comments and docstrings
2. Review the Jupyter notebook for detailed analysis
3. Examine model performance metrics in `models/model_performance.json`

## 📄 License

This project is created for educational and portfolio purposes.

## 🙏 Acknowledgments

- Dataset provided as part of Tourism Analytics project
- Built with open-source libraries and frameworks
- Inspired by real-world tourism recommendation systems

---

**Project Status**: ✅ Complete

**Last Updated**: January 2026
