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
| **ML Model** | Random Forest Classifier — 86.8% accuracy, 0.85 ROC-AUC |
| **Interactive Dashboard** | Streamlit web app — dark modern UI with Space Grotesk font |
| **Full Pipeline** | Scikit‑learn Pipeline (preprocessing + model) saved as a single file |
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
**Mild class imbalance** — ROC-AUC is preferred over accuracy for evaluation.

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
| NumOfProducts | +0.05 | Positive |
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
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- **80/20 split** with stratification to preserve the 20% churn ratio in both sets
- `random_state=42` for reproducibility

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

### 5.3 Why Pipeline?
- **Single file:** One `.pkl` contains preprocessing + model — no separate scaler needed
- **No feature mismatch:** Pipeline guarantees the same encoding and scaling every time
- **No manual dummies:** Send raw text (`"France"`, `"Male"`) — Pipeline handles OneHotEncoding automatically

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
        Random Forest     0.8640     0.7824  0.4595    0.5789   0.8522
                  SVM     0.8625     0.8474  0.3956    0.5394   0.8231
                  KNN     0.8390     0.6680  0.4152    0.5121   0.7828
  Logistic Regression     0.8080     0.5891  0.1867    0.2836   0.7748
        Decision Tree     0.7825     0.4685  0.5111    0.4888   0.6815
```

### 6.3 Analysis
- **🏆 Random Forest wins** across accuracy, F1, and ROC-AUC.
- **SVM** is a strong second — zero overfitting, highest precision.
- **Logistic Regression** underfits — 18.7% recall means it misses 81% of churners.
- **Decision Tree** overfits severely — 100% train accuracy, only 78% test.

### 6.4 Overfitting Check

```
                Model   Train Accuracy  Test Accuracy  Difference
        Decision Tree          1.0000         0.7825      0.2175 🔴
        Random Forest          1.0000         0.8640      0.1360 🟠
                  KNN          0.8806         0.8390      0.0416 🟡
  Logistic Regression          0.8106         0.8080      0.0026 ✅
                  SVM          0.8625         0.8625      0.0000 ⭐
```

**Key Insights:**
- **SVM:** Perfect generalization — train = test accuracy. Overfitting = 0.
- **Random Forest:** 13.6% gap — needs hyperparameter tuning
- **Decision Tree:** 21.7% gap — dangerously overfit

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
| Train Accuracy | 100% | **89.3%** |
| Test Accuracy | 86.4% | **86.8%** ✅ |
| Overfitting Gap | 13.6% | **2.5%** ✅ |

**The gap dropped from 13.6% to 2.5%** — the model now generalizes much better.

### 7.4 Confusion Matrix
```
                     Predicted No   Predicted Yes
     Actual No         1521              71
     Actual Yes         205             203
```

- **True Negatives:** 1,521 (correctly identified stayers)
- **True Positives:** 203 (correctly identified churners)
- **False Positives:** 71 (predicted churn, but stayed)
- **False Negatives:** 205 (predicted stay, but churned — **missed churners**)

### 7.5 Classification Report
```
              precision    recall  f1-score   support
           0       0.88      0.96      0.92      1592
           1       0.74      0.50      0.60       408
    accuracy                           0.87      2000
```

### 7.6 ROC Curve
The ROC-AUC of the tuned model is **~0.85**, confirming strong separation between the two classes.

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
| 1 | **Age** | ~25% | Older customers are much more likely to churn |
| 2 | **NumOfProducts** | ~18% | Customers with 3-4 products show higher churn |
| 3 | **Balance** | ~15% | High balance + inactivity = churn signal |
| 4 | **EstimatedSalary** | ~12% | Moderate influence through interactions |
| 5 | **CreditScore** | ~10% | Lower scores correlate with higher churn |
| 6 | **Geography (Germany)** | ~7% | German customers churn more than French/Spanish |
| 7 | **IsActiveMember** | ~6% | Active members are more loyal |
| 8 | **Tenure** | ~4% | Newer customers are slightly more likely to churn |

**Bottom line:** Age × Balance × Activity are the three pillars of churn prediction for this dataset.

---

## 9. Deployment — Streamlit Dashboard

### 9.1 Architecture
```
User Browser (Port 8501)
        │
        ▼
┌──────────────────┐
│   app.py          │  Streamlit frontend
│   app_dashboard.py│  (dark modern variant)
└──────┬───────────┘
       │ joblib.load()
       ▼
┌──────────────────┐
│  bank_churn_     │  Scikit‑learn Pipeline
│  model_full.pkl  │  ├── ColumnTransformer
│                  │  │   ├── StandardScaler
│                  │  │   └── OneHotEncoder
│                  │  └── RandomForestClassifier
└──────────────────┘
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

### 9.3 Key Code Pattern — Pipeline Predictions
```python
# No need to manually encode or scale — the Pipeline handles everything
input_data = pd.DataFrame([{
    'CreditScore': 650, 'Age': 35, 'Tenure': 5,
    'Balance': 50000.0, 'NumOfProducts': 2,
    'HasCrCard': 1, 'IsActiveMember': 1,
    'EstimatedSalary': 100000.0,
    'Geography': 'France', 'Gender': 'Female'
}])

prediction = model.predict(input_data)[0]        # 0 or 1
probability = model.predict_proba(input_data)[0]  # [P(stay), P(churn)]
```

### 9.4 Model Serialization
```python
import joblib

# Save the complete Pipeline (preprocessor + classifier)
joblib.dump(model, "bank_churn_model_full.pkl")

# Load anywhere — same interface, same preprocessing
model = joblib.load("bank_churn_model_full.pkl")
```

**Why one file?**
- No separate scaler/encoder files to manage
- Zero risk of feature-order mismatch
- Identical preprocessing guarantees reproducibility

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
├── app.py                        # Original Streamlit app (Arabic comments)
├── app_clean.py                  # Clean version — no comments
├── app_dashboard.py              # Dark modern dashboard (MAIN)
│
├── bank_churn_model_full.pkl     # Trained Pipeline — 8.3 MB
│
├── requirements.txt              # Python dependencies
├── README.md                     # Quick-start guide
│
└── DOCS.md                       # ← This file (full documentation)
```

### Supporting Files (in parent Vault)
```
KSF Vault/
├── AI-شروحات.md                  # 2,700+ line Arabic knowledge base
└── assets/
    ├── logistic_regression.png   # Model diagrams (5)
    ├── decision_tree.png
    ├── random_forest.png
    ├── knn.png
    ├── svm.png
    ├── correlation_types.png     # Correlation diagrams (3)
    ├── positive_correlation.png
    ├── negative_correlation.png
    └── dash_final_*.png          # Dashboard screenshots
```

---

## Appendix A: Model Performance Summary

| Stage | Model | ROC-AUC | Train Acc | Test Acc | Overfit Gap |
|-------|-------|---------|-----------|----------|-------------|
| Baseline | Logistic Regression | 0.7748 | 0.8106 | 0.8080 | 0.0026 |
| Baseline | Decision Tree | 0.6815 | 1.0000 | 0.7825 | 0.2175 |
| Baseline | Random Forest | 0.8522 | 1.0000 | 0.8640 | 0.1360 |
| Baseline | KNN | 0.7828 | 0.8806 | 0.8390 | 0.0416 |
| Baseline | SVM | 0.8231 | 0.8625 | 0.8625 | 0.0000 |
| **Tuned** | **Random Forest** | **0.8522** | **0.8931** | **0.8680** | **0.0251** ✅ |

## Appendix B: Key Lessons Learned

1. **Always do EDA first** — the correlation heatmap revealed Age, Balance, and Activity as the top predictors
2. **Try multiple models** — Logistic Regression was the worst; Random Forest was the best. No one can predict this without experimentation.
3. **Overfitting is real** — the default Decision Tree memorized the training set (100% accuracy) and failed on the test set (78%)
4. **Hyperparameter tuning matters** — reducing the Random Forest overfitting gap from 13.6% → 2.5% with 4 parameter changes
5. **Pipeline is king for deployment** — one `.pkl` file ensures identical preprocessing every time. No feature-name warnings, no mismatch errors.
6. **ROC-AUC > Accuracy** — the imbalance (80/20) means a "predict all zero" model gets 80% accuracy but 0.5 AUC. AUC catches this.

---

<p align="center"><strong>End of Documentation</strong></p>
