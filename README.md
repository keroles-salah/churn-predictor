<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/model-Random%20Forest-831C91" alt="Model">
  <img src="https://img.shields.io/badge/accuracy-86.8%25-success" alt="Accuracy">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<h1 align="center">🏦 Bank Churn Predictor</h1>

<p align="center">
  <b>An interactive web app for predicting bank customer churn</b><br>
  Built with Streamlit + Random Forest — featuring a premium dark UI with a Neon Purple design system
</p>

---

## 🎯 Overview

**Bank Churn Predictor** is a predictive analytics tool that helps banks and financial institutions identify customers who are likely to leave (Customer Churn). Enter a customer's profile and get an instant prediction with probability scores.

This is not just an ML model — it's a complete application with a polished UI, modern design, and smooth user experience.

---

## ✨ Features

- 🎨 **Premium Dark UI** — Purple gradient theme, Space Grotesk font, gradient buttons, animated result cards
- ⚡ **Instant Prediction** — Results appear in under a second after clicking Predict
- 📊 **Dual Probability View** — Stay probability and churn probability displayed side-by-side in metric cards
- 📱 **Fully Responsive** — Works on mobile, tablet, and desktop
- 🔍 **Technical Details Panel** — Built-in expander showing raw prediction and input data
- 🧠 **Full Pipeline Model** — Scikit-learn Pipeline (StandardScaler + OneHotEncoder + Random Forest)
- 🚀 **Cached Model Loading** — `@st.cache_resource` ensures fast performance with no reloads

---

## 🧰 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive web UI framework |
| **scikit-learn** | Model training + Pipeline |
| **pandas** | Input data processing |
| **NumPy** | Numerical operations |
| **joblib** | Model serialization & loading |
| **Random Forest** | Classification algorithm |
| **CSS3** | Custom UI styling |

---

## 🗂️ Project Structure

```
churn-predictor/
├── app_dashboard.py              # Streamlit app — UI + logic
├── bank_churn_model_full.pkl     # Trained model (full Pipeline)
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation (this file)
```

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.10 or later
- pip (Python package manager)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/keroles-salah/churn-predictor.git
cd churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app_dashboard.py

# 4. Open your browser at
# http://localhost:8501
```

---

## 🎮 How to Use

1. **Open the app** in your browser after running `streamlit run`
2. **Fill in the customer's data** across 10 input fields:
   - **Credit Score** — Credit score (350–850)
   - **Age** — Customer age (18–100)
   - **Tenure** — Years with the bank (0–10)
   - **Balance** — Current balance in USD (0–300,000)
   - **Estimated Salary** — Annual estimated salary (0–300,000)
   - **Number of Products** — Products subscribed to (1–4)
   - **Geography** — Country (France, Germany, Spain)
   - **Gender** — Female or Male
   - **Has Credit Card** — Does the customer have a credit card?
   - **Is Active Member** — Is the customer an active member?
3. **Click "Predict Churn"**
4. **See the result instantly** — High Risk or Low Risk, with percentage probabilities

---

## 📋 Input Features

| # | Feature | Type | Range / Values | Description |
|---|---------|------|----------------|-------------|
| 1 | CreditScore | Numeric | 350–850 | Customer's credit score |
| 2 | Age | Numeric | 18–100 | Customer's age |
| 3 | Tenure | Numeric | 0–10 | Years the customer has been with the bank |
| 4 | Balance | Numeric | 0–300,000 | Current account balance (USD) |
| 5 | EstimatedSalary | Numeric | 0–300,000 | Estimated annual salary (USD) |
| 6 | NumOfProducts | Categorical | 1, 2, 3, 4 | Number of bank products the customer uses |
| 7 | Geography | Categorical | France, Germany, Spain | Customer's country |
| 8 | Gender | Categorical | Female, Male | Customer's gender |
| 9 | HasCrCard | Binary | Yes / No | Does the customer have a credit card? |
| 10 | IsActiveMember | Binary | Yes / No | Is the customer an active member? |

---

## 🧠 Model Details

### Pipeline Architecture

```
Input Data
    │
    ├── Numerical Features (StandardScaler)
    │   └── CreditScore, Age, Tenure, Balance, NumOfProducts,
    │       HasCrCard, IsActiveMember, EstimatedSalary
    │
    ├── Categorical Features (OneHotEncoder)
    │   └── Geography, Gender
    │
    └── RandomForestClassifier
        ├── n_estimators: 200
        ├── max_depth: 10
        ├── random_state: 42
        └── class_weight: balanced
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 86.8% |
| **ROC-AUC** | 0.85 |
| **Precision (Churn)** | ~0.84 |
| **Recall (Churn)** | ~0.80 |
| **F1-Score (Churn)** | ~0.82 |

### Dataset

- **Source:** Kaggle — Bank Customer Churn Dataset
- **Size:** 10,000 European bank customers
- **Churn Rate:** ~20.4% (imbalanced — handled via `class_weight='balanced'`)

### Preprocessing

- **StandardScaler:** Standardizes numeric features (mean = 0, std = 1)
- **OneHotEncoder:** Converts categorical features (Geography, Gender) to binary columns
- **Train/Test Split:** 80% training, 20% testing with `stratify` to preserve churn ratio

---

## 🎨 Screenshots

<p align="center">
  <i>⚠️ Coming soon</i>
</p>

---

## 🔮 Roadmap

- [ ] Add SHAP/LIME for model explainability (Explainable AI)
- [ ] Export prediction report as PDF
- [ ] Batch prediction via CSV upload
- [ ] Compare multiple models (XGBoost, LightGBM, Logistic Regression)
- [ ] Feature importance visualization
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Arabic language support in the UI
- [ ] Hyperparameter tuning with GridSearchCV

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use and modify it.

---

## 👤 Author

<p>
  <b>Keroles Salah Fakhry</b> — <b>KSF</b><br>
  🎓 Computer Science Student — Assiut National University<br>
  🤖 AI & Machine Learning Enthusiast
</p>

<p>
  <a href="https://keroles-sala.me/"><img src="https://img.shields.io/badge/Portfolio-keroles--sala.me-D552A3?logo=google-chrome&logoColor=white" alt="Portfolio"></a>
  <a href="https://github.com/keroles-salah"><img src="https://img.shields.io/badge/GitHub-keroles--salah-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://www.linkedin.com/in/kerolessalah05/"><img src="https://img.shields.io/badge/LinkedIn-kerolessalah05-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://www.youtube.com/@kerlssalah"><img src="https://img.shields.io/badge/YouTube-@kerlssalah-FF0000?logo=youtube&logoColor=white" alt="YouTube"></a>
</p>

---

<p align="center">
  <sub>Built with ❤️ by KSF | © 2026</sub>
</p>
