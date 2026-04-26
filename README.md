# 🧑‍💼 Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave a company, using the IBM HR Analytics dataset. The project covers the full ML pipeline — from EDA to model comparison, explainability, and an interactive Gradio demo.

---

## 📌 Project Overview

Employee attrition is a costly problem for organizations. This project builds a predictive model to identify at-risk employees using structured HR data, enabling proactive retention strategies.

**Target Variable:** `Attrition` (1 = Left, 0 = Stayed)

---

## 📂 Dataset

- **Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — Kaggle
- **Size:** 1,470 rows × 35 columns
- **Class Distribution:** ~84% Stay, ~16% Leave (imbalanced)

---

## 🔧 Tech Stack

| Category | Libraries |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Explainability | `shap` |
| Demo | `gradio` |

---

## 📊 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Class balance check
- Attrition by Department and Age
- Correlation heatmap

### 2. Data Preprocessing
- Dropped irrelevant columns (`EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`)
- Checked for null values, duplicates, and unique values
- Label encoded categorical features
- Standardized features with `StandardScaler`

### 3. Modeling
Three models were trained and compared:

| Model | ROC-AUC |
|---|---|
| 🥇 Logistic Regression | 0.807 |
| 🥈 XGBoost | 0.771 |
| 🥉 Random Forest | 0.760 |

- Used `scale_pos_weight=5` (XGBoost) and `class_weight='balanced'` (RF & LR) to handle class imbalance
- Evaluated using classification report, ROC-AUC, and confusion matrix

### 4. Model Evaluation
- ROC Curve comparison for all 3 models
- Confusion matrix side-by-side comparison
- 5-Fold Cross Validation for robust performance estimation

### 5. Explainability (SHAP)
- SHAP summary plot — feature impact on individual predictions
- SHAP bar plot — global feature importance ranking
- Key drivers: `OverTime`, `MonthlyIncome`, `Age`, `JobSatisfaction`

### 6. Gradio Demo
Interactive web app where you can input employee details and get a real-time attrition risk score.

---

## 🚀 How to Run

### Option 1 — Kaggle (Recommended)
1. Open the notebook directly on Kaggle
2. Enable internet access in notebook settings
3. Run all cells

### Option 2 — Google Colab
```bash
# Install dependencies
pip install kagglehub xgboost shap gradio
```
Then upload and run the notebook.

### Option 3 — Local
```bash
git clone https://github.com/Berserker268/employee-attrition-prediction
cd employee-attrition-prediction
pip install -r requirements.txt
jupyter notebook Employee_Attrition_Prediction.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
gradio
kagglehub
```

---

## 📈 Key Findings

- **OverTime** is the strongest predictor of attrition — employees working overtime are significantly more likely to leave
- **Low MonthlyIncome** and **Single marital status** are also strong attrition indicators
- **High JobSatisfaction** and **StockOptionLevel** are strong retention factors
- Logistic Regression outperformed tree-based models on ROC-AUC, likely due to the small, well-structured dataset size

---

## 🙋 Author Mahdi Rahman

Made with ❤️ as part of a personal ML portfolio project.

Feel free to fork, star ⭐, and contribute!
