# 🏠 House Price Prediction – Task 1  
**Machine Learning Internship @ SkillCraft Technologies**

---

## 📌 Problem Statement

Predict the sale prices of houses in Ames, Iowa using a simple **Linear Regression** model based on a few key numerical features. The goal is to build a predictive model using structured tabular data from a Kaggle competition.

---

## 📂 Dataset

Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- `train.csv`: Contains features and `SalePrice` (target).
- `test.csv`: Contains features only; used for submission prediction.

---

## 🧠 Features Used

| Feature        | Description                               |
|----------------|-------------------------------------------|
| `GrLivArea`    | Above-ground living area (sq ft)          |
| `BedroomAbvGr` | Bedrooms above basement level             |
| `FullBath`     | Full bathrooms above grade                |

**Target Variable:** `SalePrice` (in USD)

---

## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- scikit-learn
- Seaborn / Matplotlib

---

## 📈 Model & Evaluation

### 🔧 Model
- Linear Regression (`sklearn.linear_model.LinearRegression`)
- Missing data handled with `SimpleImputer(strategy='mean')`

### 🧪 Evaluation Metrics (on validation set)
| Metric | Value |
|--------|--------|
| R² Score | `0.6341` |
| RMSE     | `52,975.72` |

> The model explains ~63% of variance in sale prices using only 3 features.

---

## 📊 Visualizations

All plots are saved as `.png` files automatically:

- `GrLivArea_vs_SalePrice.png`
- `BedroomAbvGr_vs_SalePrice.png`
- `FullBath_vs_SalePrice.png`
- `residuals_distribution.png`
- `actual_vs_predicted.png`

---


