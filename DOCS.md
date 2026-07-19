# Churn Prediction — Full Project Documentation

> **Author:** Keroles Salah | **Date:** 2026-07-18 | **Last Updated:** 2026-07-18
> **Stack:** Python 3 · Scikit‑learn · Pandas · Streamlit · Joblib
> **Dataset:** Kaggle — Bank Customer Churn (10,000 customers × 14 columns)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis-eda)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Selection & Training](#5-model-selection--training)
6. [Model Evaluation](#6-model-evaluation)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Feature Importance](#8-feature-importance)
9. [Deployment — Streamlit Dashboard](#9-deployment--streamlit-dashboard)
10. [How to Run](#10-how-to-run)
11. [File Structure](#11-file-structure)

---

## 1. Project Overview

### Business Problem
A European bank wants to identify customers who are likely to **leave the bank (churn)**. By predicting churn in advance, the bank can:
- Offer retention incentives to at-risk customers
- Reduce customer acquisition costs
- Increase long-term revenue

### Goal
Build a **machine learning classifier** that predicts whether a customer will exit the bank based on their demographic, financial, and behavioural data.

### What We Delivered
| Deliverable | Description |
|---|---|
| **ML Model** | Random Forest Classifier — 86.1% accuracy, 0.87 ROC-AUC |
| **Interactive Dashboard** | Streamlit web app — dark modern UI with Space Grotesk font |
| **ML Model** | Random Forest Classifier (saved separately from scaler) |
| **Knowledge Base** | 2,700+ line Arabic reference guide explaining every concept |

---

## 2. Dataset

### Source
[Kaggle — Churn Modelling](https://www.kaggle.com/datasets/shubh0799/churn-modelling)

### Shape
**10,000 rows × 14 columns**

### Column Descriptions

| # | Column | Type | Range / Values | Meaning |
|---|--------|------|----------------|---------|
| 1 | `RowNumber` | int | 1 – 10,000 | Row index — **dropped** |
| 2 | `CustomerId` | int | 15,565,702 – 15,815,690 | Unique ID — **dropped** |
| 3 | `Surname` | str | — | Last name — **dropped** |
| 4 | `CreditScore` | int | 350 – 850 | Financial credit rating |
| 5 | `Geography` | str | France, Germany, Spain | Customer country |
| 6 | `Gender` | str | Male, Female | Customer sex |
| 7 | `Age` | int | 18 – 92 | Age in years |
| 8 | `Tenure` | int | 0 – 10 | Years with the bank |
| 9 | `Balance` | float | 0 – 250,898 | Current account balance |
| 10 | `NumOfProducts` | int | 1 – 4 | Number of bank products owned |
| 11 | `HasCrCard` | int | 0 / 1 | Owns a credit card |
| 12 | `IsActiveMember` | int | 0 / 1 | Active user of bank services |
| 13 | `EstimatedSalary` | float | 11 – 199,992 | Estimated annual income |
| 14 | **`Exited`** | int | 0 / 1 | **Target** — did the customer leave? |

### Target Distribution
```
No Churn (0):  7,963 (79.63%)
Churn (1):     2,037 (20.37%)
```
**Mild class imbalance** (~80/20) — handled naturally; no class weighting used.

### Data Quality
- **Null values:** 0
- **Duplicate rows:** 0
- **Data types:** 11 numeric (int/float) + 3 object (string)

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Quick Inspection
```python
df.head()           # First 5 rows
df.tail()           # Last 5 rows
df.shape            # (10000, 14)
df.info()           # Column types, non-null counts
df.describe()       # Summary statistics for numeric columns
df.describe(include='object')  # Summary for categorical columns
df.isnull().sum()   # All zeros — clean data
df.duplicated().sum()  # 0 duplicates
```

### 3.2 Target Distribution
```python
df['Exited'].value_counts().plot(kind='bar')
```
**Finding:** 79.6% stay / 20.4% churn — mildly imbalanced, but manageable without oversampling.

### 3.3 Histograms
```python
plt.hist(df['Age'], bins=20)
plt.hist(df['Balance'], bins=20)
plt.hist(df['EstimatedSalary'], bins=20)
```
**Findings:**
- **Age:** Right‑skewed — most customers are 25–45. Older customers show higher churn.
- **Balance:** Bimodal — large spike at 0 (inactive or new accounts), then spread across 50k–150k.
- **EstimatedSalary:** Nearly uniform distribution — weak predictor.

### 3.4 Categorical Distributions
```python
df['Gender'].value_counts().plot(kind='bar')     # ~50/50 Male/Female
df['Geography'].value_counts().plot(kind='bar')  # France > Germany > Spain
```

### 3.5 Boxplots (Feature vs Target)
```python
sns.boxplot(x='Exited', y='Age', data=df)
sns.boxplot(x='Exited', y='Balance', data=df)
```
**Findings:**
- **Age:** Churners are noticeably older (median ~45 vs ~35 for non-churners).
- **Balance:** Churners have slightly higher median balance, but many non-churners have zero balance.

### 3.6 Correlation Heatmap
```python
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
```
**Top correlations with Exited:**
| Feature | Correlation | Direction |
|---------|-------------|-----------|
| Age | +0.29 | Positive |
| Balance | +0.12 | Positive |
| IsActiveMember | -0.16 | Negative |
| NumOfProducts | -0.048 | Positive |
| EstimatedSalary | +0.01 | ~None |

---

## 4. Data Preprocessing

### 4.1 Drop Irrelevant Columns
```python
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```
These columns have no predictive value — they would only introduce noise.

### 4.2 Split Features & Target
```python
X = df.drop('Exited', axis=1)
y = df['Exited']
```

### 4.3 Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- 80/20 split with `random_state=42` for reproducibility (no stratification)

### 4.4 Encoding Categorical Variables
```python
X = pd.get_dummies(X, drop_first=True)
```
- **Geography** (3 values) → 2 binary columns (`Geography_Germany`, `Geography_Spain`)
- **Gender** (2 values) → 1 binary column (`Gender_Male`)
- `drop_first=True` avoids the **dummy variable trap** (multicollinearity)

### 4.5 Feature Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Why StandardScaler?**
- KNN, SVM, and Logistic Regression are distance-based — they require standardized features
- Formula: `z = (x - μ) / σ` → result has mean = 0, std = 1

**⚠️ Never fit on test data — only transform.**
Fitting on test data would leak information from the test set into the model.

---

## 5. Model Selection & Training

### 5.1 Models Tested
| Model | Key Hyperparameter | Scaling Required | Speed |
|-------|-------------------|------------------|-------|
| Logistic Regression | `C` (regularization strength) | ✅ Yes | ⚡ Very fast |
| Decision Tree | `max_depth` | ❌ No | ⚡ Fast |
| Random Forest | `n_estimators`, `max_depth` | ❌ No | 🐢 Moderate |
| KNN | `n_neighbors` | ✅ Yes | 🐢 Slow predict |
| SVM | `C`, `kernel`, `gamma` | ✅ Yes | 🐢 Slow on large data |

### 5.2 Training Loop (With Scaling via Pipeline)
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary']
categorical_cols = ['Geography', 'Gender']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

models = {
    "Logistic Regression": Pipeline([('prep', preprocessor), ('clf', LogisticRegression(random_state=42, max_iter=1000))]),
    "Decision Tree":       Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),
    "Random Forest":       Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    "KNN":                 Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier())]),
    "SVM":                 Pipeline([('prep', preprocessor), ('clf', SVC(random_state=42, probability=True))]),
}

for name, model in models.items():
    model.fit(X_train, y_train)
```

### 5.3 Why Pipeline? (for model comparison)
- **Fair comparison:** Same preprocessing is applied identically to every model during training
- **No feature mismatch:** Pipeline guarantees the same encoding and scaling across all models
- **Clean training code:** Each model is trained with identical preprocessing in one `fit()` call

> **Note:** For deployment, preprocessing is done manually in `app_dashboard.py` — one-hot encoding in Python + `scaler.pkl` + `bank_churn_model.pkl`. The training Pipeline was used only during experimentation, not in production.

---

## 6. Model Evaluation

### 6.1 Evaluation Metrics
| Metric | Formula | Best Use |
|--------|---------|----------|
| **Accuracy** | (TP + TN) / Total | Balanced classes only |
| **Precision** | TP / (TP + FP) | When false positives are expensive |
| **Recall** | TP / (TP + FN) | When false negatives are expensive |
| **F1‑Score** | 2 × (P × R) / (P + R) | Single balanced number |
| **ROC-AUC** | Area under ROC curve | **Imbalanced classes** — our primary metric |

### 6.2 Results — All Models

```
                Model   Accuracy  Precision  Recall  F1-Score  ROC-AUC
        Random Forest     0.8665     0.7625  0.4656    0.5782   0.8653
                  SVM     0.8560     0.7692  0.3817    0.5102   0.8248
                  KNN     0.8300     0.6109  0.3715    0.4620   0.7604
  Logistic Regression     0.8110     0.5524  0.2010    0.2948   0.7789
        Decision Tree     0.7810     0.4490  0.5038    0.4748   0.6763
```

### 6.3 Analysis
- **🏆 Random Forest wins** on accuracy and ROC-AUC.
- **SVM** is a strong second — highest precision (0.7692).
- **Logistic Regression** underfits — 20.1% recall means it misses 80% of churners.
- **Decision Tree** overfits severely — 100% train accuracy, only 78.1% test.

### 6.4 Overfitting Check

```
                Model   Train Accuracy  Test Accuracy  Difference
        Decision Tree          1.0000         0.7810      0.2190 🔴
        Random Forest          1.0000         0.8665      0.1335 🟠
                  KNN          0.8741         0.8300      0.0441 🟡
  Logistic Regression          0.8114         0.8110      0.0004 ✅
                  SVM          0.8654         0.8560      0.0094 ⭐
```

**Key Insights:**
- **SVM:** Near-perfect generalization — only 0.94% gap.
- **Random Forest:** 13.4% gap — needs hyperparameter tuning
- **Decision Tree:** 21.9% gap — dangerously overfit

---

## 7. Hyperparameter Tuning

### 7.1 Tuning Strategy
We selected **Random Forest** (best ROC-AUC) and tuned to reduce overfitting while maintaining accuracy.

### 7.2 Before vs After

| Parameter | Default (Before) | Tuned (After) | Effect |
|-----------|-----------------|---------------|--------|
| `n_estimators` | 100 | **200** | More trees = more stable = lower variance |
| `max_depth` | None (unlimited) | **10** | Caps tree depth — prevents memorization |
| `min_samples_split` | 2 | **5** | Requires at least 5 samples to split a node |
| `min_samples_leaf` | 1 | **2** | Minimum 2 samples per leaf — smoother boundaries |

```python
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_tuned.fit(X_train, y_train)
```

### 7.3 Results After Tuning
| Metric | Default RF | Tuned RF |
|--------|-----------|----------|
| Train Accuracy | 100% | **89.4%** |
| Test Accuracy | 86.7% | **86.1%** ✅ |
| Overfitting Gap | 13.4% | **3.3%** ✅ |

**The gap dropped from 13.4% to 3.3%** — the model now generalizes much better.

### 7.4 Confusion Matrix
```
                     Predicted No   Predicted Yes
     Actual No         1548              59
     Actual Yes         219             174
```

- **True Negatives:** 1,548 (correctly identified stayers)
- **True Positives:** 174 (correctly identified churners)
- **False Positives:** 59 (predicted churn, but stayed)
- **False Negatives:** 219 (predicted stay, but churned — **missed churners**)

### 7.5 Classification Report
```
              precision    recall  f1-score   support
           0       0.88      0.96      0.92      1607
           1       0.75      0.44      0.56       393
    accuracy                           0.86      2000
```

### 7.6 ROC Curve
The ROC-AUC of the tuned model is **0.87**, confirming strong separation between the two classes.

---

## 8. Feature Importance

```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_tuned.feature_importances_
}).sort_values(by='Importance', ascending=False)
```

### Top Features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **Age** | 32.3% | Older customers are much more likely to churn |
| 2 | **NumOfProducts** | 23.6% | Customers with 3-4 products show higher churn |
| 3 | **Balance** | 10.3% | High balance + inactivity = churn signal |
| 4 | **EstimatedSalary** | 7.5% | Moderate influence through interactions |
| 5 | **CreditScore** | 7.4% | Lower scores correlate with higher churn |
| 6 | **IsActiveMember** | 6.9% | Active members are more loyal |
| 7 | **Geography (Germany)** | 4.5% | German customers churn more than French/Spanish |
| 8 | **Tenure** | 4.0% | Newer customers are slightly more likely to churn |

**Bottom line:** Age × Balance × Activity are the three pillars of churn prediction for this dataset.

---

## 9. Deployment — Streamlit Dashboard

### 9.1 Architecture
```
User Browser (Port 8501)
        │
        ▼
┌──────────────────────────────────┐
│   app_dashboard.py               │  Streamlit frontend (dark modern UI)
│   ├── Manual one-hot encoding     │  Geography → 2 cols, Gender → 1 col
│   └── scaler.transform()         │  StandardScaler on 11 features
└──────┬───────────────────────────┘
       │ model.predict() + scaler.transform()
       ▼
┌─────────────────────┬─────────────────┐
│  bank_churn_        │  scaler.pkl     │
│  model.pkl          │  (StandardScaler)│
│  (RandomForest)     │                 │
└─────────────────────┴─────────────────┘
```

### 9.2 Dashboard Features
| Feature | Implementation |
|---------|---------------|
| **Font** | Space Grotesk (Google Fonts) |
| **Color Palette** | `#462C7D` / `#831C91` / `#D552A3` / `#FF70BF` |
| **Theme** | Dark glassmorphism — deep purple/black background |
| **Input Fields** | 3‑column grid: sliders, number inputs, select boxes, radio, checkboxes |
| **Predict Button** | Full‑width gradient button with neon glow |
| **Results** | Custom HTML card — green/pink based on prediction |
| **Probability** | Two metric cards: Stay % + Churn % |
| **Responsive** | Mobile‑first media queries (`max-width: 640px`) |

### 9.3 Key Code Pattern — Manual Preprocessing + Prediction
```python
# Encode categorical features manually, then scale, then predict
import pandas as pd

raw_data = {
    'CreditScore': 650, 'Age': 35, 'Tenure': 5,
    'Balance': 50000.0, 'NumOfProducts': 2,
    'HasCrCard': 1, 'IsActiveMember': 1,
    'EstimatedSalary': 100000.0,
    'Geography': 'France', 'Gender': 'Female'
}

# Manual one-hot encoding (must match training order!)
geo_germany = 1 if raw_data['Geography'] == 'Germany' else 0
geo_spain  = 1 if raw_data['Geography'] == 'Spain' else 0
gender_male = 1 if raw_data['Gender'] == 'Male' else 0

input_df = pd.DataFrame([[raw_data['CreditScore'], raw_data['Age'],
    raw_data['Tenure'], raw_data['Balance'], raw_data['NumOfProducts'],
    raw_data['HasCrCard'], raw_data['IsActiveMember'],
    raw_data['EstimatedSalary'],
    geo_germany, geo_spain, gender_male]],
    columns=['CreditScore','Age','Tenure','Balance','NumOfProducts',
             'HasCrCard','IsActiveMember','EstimatedSalary',
             'Geography_Germany','Geography_Spain','Gender_Male'])

scaled = scaler.transform(input_df)
prediction = model.predict(scaled)[0]           # 0 or 1
probability = model.predict_proba(scaled)[0]    # [P(stay), P(churn)]
```

### 9.4 Model Serialization
```python
import joblib

# Save model and scaler separately
joblib.dump(model, "bank_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load both for predictions
model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
```

**Why two files?**
- Model and scaler are saved separately for clarity
- Manual one-hot encoding is done in Python before scaling
- Feature order is explicitly defined in the app code

---

## 10. How to Run

### 10.1 Prerequisites
- **Python** 3.10+
- **pip** (Python package manager)

### 10.2 Setup
```bash
# Clone or copy the project folder
cd churn-predictor

# Install dependencies
pip install -r requirements.txt
```

### 10.3 Run the Dashboard
```bash
streamlit run app_dashboard.py

# Opens at: http://localhost:8501
```

### 10.4 Dependencies (`requirements.txt`)
```
streamlit>=1.50
scikit-learn>=1.5
pandas>=2.0
numpy>=1.24
joblib>=1.4
```

### 10.5 Test a Prediction
1. Enter a customer profile in the input fields
2. Click **"Predict Churn"**
3. View the result card showing "LIKELY TO STAY" or "LIKELY TO CHURN"
4. Two probability cards show exact percentages

---

## 11. File Structure

```
churn-predictor/
│
├── app_dashboard.py              # Dark modern dashboard (MAIN)
│
├── bank_churn_model.pkl          # Trained Random Forest Classifier
├── scaler.pkl                    # StandardScaler (fitted on 11 features)
│
├── Churn_Modelling.csv           # Dataset (10,000 rows × 14 columns)
├── Churn_Modelling_Solution.ipynb # EDA + training notebook
├── test_cases.xlsx               # Sample test cases for validation
│
├── screenshots/
│   └── dashboard.png             # App screenshot
│
├── requirements.txt              # Python dependencies
├── README.md                     # Quick-start guide
│
└── DOCS.md                       # ← This file (full documentation)
```

---

## Appendix A: Model Performance Summary

| Stage | Model | ROC-AUC | Train Acc | Test Acc | Overfit Gap |
|-------|-------|---------|-----------|----------|-------------|
| Baseline | Logistic Regression | 0.7789 | 0.8114 | 0.8110 | 0.0004 |
| Baseline | Decision Tree | 0.6763 | 1.0000 | 0.7810 | 0.2190 |
| Baseline | Random Forest | 0.8653 | 1.0000 | 0.8665 | 0.1335 |
| Baseline | KNN | 0.7604 | 0.8741 | 0.8300 | 0.0441 |
| Baseline | SVM | 0.8248 | 0.8654 | 0.8560 | 0.0094 |
| **Tuned** | **Random Forest** | **0.8703** | **0.8944** | **0.8610** | **0.0334** ✅ |

## Appendix B: Key Lessons Learned

1. **Always do EDA first** — the correlation heatmap revealed Age, Balance, and Activity as the top predictors
2. **Try multiple models** — Logistic Regression was the worst; Random Forest was the best. No one can predict this without experimentation.
3. **Overfitting is real** — the default Decision Tree memorized the training set (100% accuracy) and failed on the test set (78.1%)
4. **Hyperparameter tuning matters** — reducing the Random Forest overfitting gap from 13.4% → 3.3% with 4 parameter changes
5. **Document your preprocessing exactly** — the model can't predict without knowing how features were encoded. Manual encoding in code is more transparent than a pipeline `.pkl`.
6. **ROC-AUC > Accuracy** — the imbalance (80/20) means a "predict all zero" model gets 80% accuracy but 0.5 AUC. AUC catches this.

---

<p align="center"><strong>End of Documentation</strong></p>
