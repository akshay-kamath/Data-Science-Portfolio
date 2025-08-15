# Automobile Price Prediction

This repository contains a **Jupyter Notebook** documenting a complete machine learning project aimed at predicting automobile prices. It demonstrates the end-to-end **data science workflow**, from raw data ingestion and preprocessing to model training, evaluation, and refinement.

## Objective
The primary objective of this project is to **develop and evaluate multiple prediction models** capable of estimating the selling price of cars based on their technical and categorical features. By analyzing historical automobile data, the project identifies key attributes that significantly influence market value.

## Project Workflow

### 1. Data Loading & Preprocessing
- The dataset used is the [Automobile Dataset](https://www.kaggle.com/datasets/premptk/automobile-data-changed) from Kaggle
- Handled missing values through imputation strategies.
- Converted features to correct data types (e.g., categorical, numeric).
- Applied normalization and standardization where appropriate.

### 2. Exploratory Data Analysis (EDA)
- Visualized relationships between car attributes and price.
- Identified correlation patterns between features.
- Highlighted high-impact predictors such as engine size, horsepower and fuel efficiency.

### 3. Model Development
- Implemented multiple regression models, including:
  - **Linear Regression**
  - **Decision Tree Regressor**
  - **Random Forest Regressor**
- Tuned hyperparameters for improved accuracy.

### 4. Model Evaluation
- Used performance metrics such as **R² score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.
- Compared results to select the most accurate and generalizable model.

---

## Dataset Details

- **Size:** 205 entries, 26 features
- **Attributes include:**
  - **Categorical:** Make, fuel type, aspiration, body style, drive wheels
  - **Numerical:** Engine size, horsepower, city-mpg, highway-mpg, curb weight

---

## Technologies & Libraries

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## Outcomes & Insights
- Identified engine size, curb weight, and horsepower as the strongest price determinants.
- Demonstrated the trade-off between interpretability (Linear Regression) and predictive power (Random Forest).
- Established a baseline for expanding into advanced models such as Gradient Boosting or Neural Networks.

---
