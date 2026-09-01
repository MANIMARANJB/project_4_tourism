# 🌍 Tourism Experience Analytics

## 📌 Project Overview

Tourism Experience Analytics is an end-to-end Data Science and Machine Learning project designed to analyze tourist behaviour, predict attraction ratings, classify tourist visit modes, and recommend suitable tourist attractions.

The project covers the complete Data Science workflow:

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- SQL Database Integration
- Regression
- Classification
- Recommendation System
- Streamlit Dashboard


## 🎯 Project Objectives

The project focuses on three major Machine Learning tasks.

### 1. ⭐ Rating Prediction — Regression

The objective is to predict the rating a tourist is likely to give to an attraction.

**Target Variable:** `Rating`

**Algorithm Used:** `HistGradientBoostingRegressor`

Features include:

- Visit Mode
- Attraction
- Attraction Type
- Attraction City
- Tourist Country
- Visit Year
- Visit Month

### 2. 👥 Visit Mode Prediction — Classification

The objective is to predict how a tourist is likely to travel.

**Target Variable:** `VisitMode`

Possible classes:

- Business
- Couples
- Family
- Friends
- Solo

**Algorithm Used:** `RandomForestClassifier`

This can help tourism businesses design targeted packages and promotional campaigns for different types of travellers.

### 3. 🎯 Attraction Recommendation System

The project uses a **Hybrid Recommendation System**.

It combines:

- Content-Based Filtering
- One-Hot Encoding
- Cosine Similarity
- Attraction Popularity
- Average Rating
- Weighted Recommendation Score

The final recommendation score is:

```text
Hybrid Score =
70% Content Similarity
+
30% Popularity & Rating Score
```

The system recommends attractions that are similar to the tourist's selected attraction while also considering their overall popularity and ratings.


## 📂 Dataset

The project uses nine related datasets:

```text
City.xlsx
Continent.xlsx
Country.xlsx
Item.xlsx
Mode.xlsx
Region.xlsx
Transaction.xlsx
Type.xlsx
User.xlsx
```

The main transaction dataset contains tourist visits, attraction information, visit mode and ratings.


## 🔗 Dataset Relationships

The major relationships between the datasets are:

```text
Transaction
   │
   ├── UserId ────────────> User
   │
   ├── AttractionId ──────> Item
   │
   └── VisitMode ─────────> Mode

Item
   │
   └── AttractionTypeId ──> Type

User
   │
   └── CityId ────────────> City

City
   │
   └── CountryId ─────────> Country

Country
   │
   └── RegionId ──────────> Region

Region
   │
   └── ContinentId ───────> Continent
```

The attraction location identifier is handled separately from the tourist City table to maintain the correct dataset relationship.


## 🧹 Data Cleaning and Preprocessing

The preprocessing stage includes:

- Dataset structure validation
- Missing value analysis
- Duplicate checking
- Data type validation
- Primary key validation
- Foreign key validation
- Text cleaning
- Handling unknown values
- Merging related datasets
- Categorical encoding
- Feature selection
- Train-test splitting

A consolidated master dataset was created for EDA, Machine Learning and dashboard development.

```text
Master Dataset
Rows    : 52,930
Columns : 23
```


## 📊 Exploratory Data Analysis

EDA was performed to understand important tourism patterns such as:

- Rating distribution
- Visit mode distribution
- Most visited attractions
- Attraction popularity
- Attraction type performance
- Attraction city performance
- Tourist country distribution
- Tourist continent distribution
- Yearly tourism trends
- Monthly tourism patterns
- Repeat visitors
- Average rating by attraction
- Average rating by attraction type
- Average rating by visit mode


## 📈 Key Insights

Some important observations from the analysis include:

- Ratings of 4 and 5 represent the majority of tourist ratings.
- Couples are the most common visit mode, followed by Family and Friends.
- Bali accounts for the majority of attraction visits in the dataset.
- Sacred Monkey Forest Sanctuary is one of the most frequently visited attractions.
- Waterbom Bali combines strong visitor volume with a high average rating.
- Tourists mainly originate from Asia, Australia/Oceania, Europe and America.
- Repeat visitors form an important customer segment.


## 🤖 Machine Learning Models

### ⭐ Regression Model

**Model:** HistGradientBoostingRegressor

**Target:** Rating

Evaluation metrics:

- MAE
- MSE
- RMSE
- R² Score

Final performance:

```text
MAE  : 0.71
RMSE : 0.91
R²   : 0.13
```

Saved model files:

```text
tourism_rating_model.pkl
tourism_rating_columns.pkl
```


### 👥 Classification Model

**Model:** RandomForestClassifier

**Target:** VisitMode

Because the VisitMode classes are imbalanced, controlled oversampling was used during model training.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Weighted F1 Score
- Confusion Matrix
- Classification Report

Final performance:

```text
Accuracy          : 45.00%
Macro F1 Score    : 0.33
Weighted F1 Score : 0.44
```

Saved model files:

```text
tourism_visitmode_model.pkl
tourism_visitmode_columns.pkl
```


## 🎯 Recommendation System

The recommendation engine uses a hybrid approach.

### Content-Based Recommendation

Attractions are represented using:

- Attraction Type
- Attraction City

Categorical information is transformed using `OneHotEncoder`.

`Cosine Similarity` is then used to measure similarity between attractions.

### Popularity-Based Recommendation

Attraction popularity is calculated using:

- Visit Count
- Unique Visitors
- Average Rating
- Weighted Rating Score

### Hybrid Recommendation

The final system combines content similarity and popularity:

```text
HybridScore =
(0.70 × SimilarityScore)
+
(0.30 × NormalizedRecommendationScore)
```

Saved recommendation files:

```text
tourism_attraction_profiles.pkl
tourism_similarity_matrix.pkl
tourism_recommendation_ranked.pkl
tourism_content_encoder.pkl
```


## 🗄️ MySQL Database

The cleaned data is stored in a normalized MySQL database.

**Database Name:** `tourism_db`

Tables:

```text
continent_table
region_table
country_table
city_table
mode_table
type_table
user_table
item_table
transaction_table
master_data_table
```

The normalized tables preserve the original dataset relationships, while `master_data_table` is used for analytics and Streamlit visualization.


## 📊 Streamlit Application

The project includes an interactive Streamlit application with four modules:

```text
📊 Business Dashboard
⭐ Rating Prediction
👥 Visit Mode Prediction
🎯 Recommendation System
```

### Business Dashboard

The dashboard provides:

- Total visits
- Unique tourists
- Number of attractions
- Average rating
- Repeat visitor percentage
- Top attractions
- Visit mode distribution
- Tourism trends
- Attraction type analysis
- Tourist country analysis
- Tourist continent analysis
- Attraction city analysis
- Rating distribution
- Monthly tourism patterns
- Interactive filters and sliders


### Rating Prediction

Users provide travel information and the regression model predicts the expected attraction rating.


### Visit Mode Prediction

The classification model predicts whether a tourist is most likely travelling for:

```text
Business
Couples
Family
Friends
Solo
```

Related geographical and attraction information is automatically linked to prevent invalid input combinations.


### Recommendation System

Users select an attraction and the hybrid recommendation engine suggests similar and highly ranked attractions.


## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Imbalanced-learn
- PyMySQL
- MySQL
- Joblib
- Streamlit
- Jupyter Notebook
- VS Code


## 📁 Project Structure

```text
Tourism-Experience-Analytics/
│
├── Data/
│   ├── City.xlsx
│   ├── Continent.xlsx
│   ├── Country.xlsx
│   ├── Item.xlsx
│   ├── Mode.xlsx
│   ├── Region.xlsx
│   ├── Transaction.xlsx
│   ├── Type.xlsx
│   └── User.xlsx
│
├── tourism_master_clean.csv
├── Tourism_Experience_Analytics.ipynb
├── tourism.py
│
├── tourism_rating_model.pkl
├── tourism_rating_columns.pkl
├── tourism_visitmode_model.pkl
├── tourism_visitmode_columns.pkl
│
├── tourism_attraction_profiles.pkl
├── tourism_similarity_matrix.pkl
├── tourism_recommendation_ranked.pkl
├── tourism_content_encoder.pkl
│
├── requirements.txt
└── README.md
```


## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Tourism-Experience-Analytics
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install pandas numpy matplotlib plotly scikit-learn imbalanced-learn pymysql streamlit joblib openpyxl
```


## ▶️ Run the Streamlit Application

Make sure the MySQL database is running and the required tables have been created.

Run:

```bash
python3 -m streamlit run tourism.py
```

The application will normally be available at:

```text
http://localhost:8501
```


## 💼 Business Use Cases

The project can support tourism businesses with:

- Tourist behaviour analysis
- Customer segmentation
- Attraction performance analysis
- Personalized attraction recommendations
- Tourist satisfaction prediction
- Visit mode prediction
- Targeted travel packages
- Marketing campaign planning
- Popular destination identification
- Tourism business decision support


## 🚀 Future Improvements

Possible future enhancements include:

- Collaborative filtering
- User-specific recommendations
- Matrix factorization
- More tourist demographic features
- Hotel recommendations
- Complete travel package recommendations
- Real-time tourism data
- Geographical map visualization
- Cloud deployment
- Automated model retraining


## ✅ Conclusion

The **Tourism Experience Analytics** project demonstrates an end-to-end Data Science workflow covering:

```text
Raw Tourism Data
        ↓
Data Cleaning
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Data Visualization
        ↓
MySQL Database
        ↓
Machine Learning
   ↙       ↓        ↘
Regression Classification Recommendation
   ↘       ↓        ↙
      Streamlit App
```

The final application combines business analytics with Machine Learning to predict attraction ratings, classify tourist visit modes, and provide attraction recommendations.
