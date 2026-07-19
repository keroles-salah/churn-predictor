import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings


st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# Feature columns — MUST match the order the scaler was trained on
FEATURE_COLUMNS = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

@st.cache_resource
def load_artifacts():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(script_dir, "bank_churn_model.pkl"))
    scaler = joblib.load(os.path.join(script_dir, "scaler.pkl"))
    return model, scaler

model, scaler = load_artifacts()

def prepare_input(credit_score, age, tenure, balance, num_products,
                  has_crcard, is_active, estimated_salary,
                  geography, gender):
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    data = pd.DataFrame([[
        credit_score, age, tenure, balance, num_products,
        int(has_crcard), int(is_active), estimated_salary,
        geo_germany, geo_spain, gender_male
    ]], columns=FEATURE_COLUMNS)

    scaled = scaler.transform(data)
    return scaled

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ── ROOT ── */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
}
.stApp {
    background: linear-gradient(160deg, #07050f 0%, #120922 35%, #200a31 65%, #1a0a2e 100%);
}

/* ── CONTAINER ── */
.main .block-container {
    max-width: 840px !important;
    padding: 2.5rem 2.2rem !important;
    margin: 2rem auto !important;
    background: rgba(18,9,34,0.88);
    border: 1px solid rgba(213,82,163,0.2);
    border-radius: 28px;
    backdrop-filter: blur(30px);
    box-shadow: 0 12px 48px rgba(70,44,125,0.25);
}

[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── HEADINGS ── */
h1 {
    background: linear-gradient(135deg, #D552A3 20%, #FF70BF 80%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem !important; font-weight: 700 !important;
    letter-spacing: -1px; margin-bottom: 0.1rem !important;
}
.st-caption { color: rgba(255,255,255,0.55) !important; font-size: 0.95rem; }
.stDivider { border-color: rgba(255,255,255,0.06) !important; }

.badge {
    display: inline-block;
    background: rgba(213,82,163,0.14);
    color: #FF70BF;
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 3px; padding: 0.35rem 0.9rem;
    border-radius: 20px; margin-bottom: 1rem;
}

/* ═══════════════════════════════════════════════
   INPUT FIELDS — ALL DARK — HIGH CONTRAST
   ═══════════════════════════════════════════════ */
/* Labels */
.stNumberInput label, .stSlider label, .stSelectbox label,
.stCheckbox label, .stRadio label, .stTextInput label {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
}

/* Number input — OVERRIDE the default light bg */
/* Streamlit number_input uses a web component; force dark */
input[type="number"], .stNumberInput input {
    background-color: #1e1035 !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(213,82,163,0.3) !important;
    border-radius: 12px !important;
    font-size: 1rem !important; font-weight: 600 !important;
    padding: 0.6rem 0.8rem !important;
    -webkit-text-fill-color: #ffffff !important;
}
input[type="number"]:focus, .stNumberInput input:focus {
    border-color: #FF70BF !important;
    box-shadow: 0 0 0 3px rgba(255,112,191,0.2) !important;
    background-color: #261545 !important;
}
/* Number input up/down arrows */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    filter: invert(0.6);
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: rgba(213,82,163,0.35) !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: #FF70BF !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: #fff !important; background: #831C91 !important;
    border-radius: 8px !important; font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Selectbox */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #200a31 !important;
    border: 1.5px solid rgba(213,82,163,0.3) !important;
    border-radius: 12px !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: #200a31 !important;
    border: 1.5px solid rgba(213,82,163,0.3) !important;
    border-radius: 12px !important;
}
.stSelectbox [data-baseweb="select"] div[class*="ValueContainer"],
.stSelectbox [data-baseweb="select"] [class*="singleValue"],
.stSelectbox [data-baseweb="select"] span {
    color: #ffffff !important; font-weight: 500 !important;
    font-size: 0.95rem !important;
}
.stSelectbox [data-baseweb="select"] svg { color: rgba(255,255,255,0.5) !important; }

/* Dropdown menu */
div[data-baseweb="popover"] {
    background: #1e1035 !important;
    border: 1px solid rgba(213,82,163,0.4) !important;
    border-radius: 10px !important;
}
div[data-baseweb="popover"] li {
    color: rgba(255,255,255,0.95) !important; font-size: 0.9rem !important;
}
div[data-baseweb="popover"] li:hover {
    background: rgba(213,82,163,0.2) !important;
}

/* Radio */
.stRadio label p {
    color: rgba(255,255,255,0.85) !important; font-weight: 600 !important;
}
.stRadio label[data-baseweb="radio"] > div:first-child {
    border-color: rgba(255,255,255,0.25) !important;
}
.stRadio label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
    background: #D552A3 !important; border-color: #FF70BF !important;
}

/* Checkbox */
.stCheckbox label p {
    color: rgba(255,255,255,0.85) !important; font-weight: 600 !important;
}
.stCheckbox label[data-baseweb="checkbox"] > div:first-child {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.25) !important;
}
.stCheckbox label[data-baseweb="checkbox"][aria-checked="true"] > div:first-child {
    background: #D552A3 !important; border-color: #FF70BF !important;
}

/* ════════════════════════════════════════════
   BUTTON
   ════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #831C91, #D552A3, #FF70BF) !important;
    border: none !important; border-radius: 16px !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: 1.15rem !important; padding: 1rem 2rem !important;
    letter-spacing: 0.5px; transition: all 0.3s ease;
    box-shadow: 0 6px 28px rgba(213,82,163,0.4);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(255,112,191,0.55);
}

/* ════════════════════════════════════════════
   RESULT CARDS
   ════════════════════════════════════════════ */
.result-box {
    border-radius: 20px; padding: 2rem; text-align: center;
    margin: 0.8rem 0; border: 2px solid;
}
.danger {
    background: linear-gradient(135deg, rgba(255,112,191,0.18), rgba(213,82,163,0.1));
    border-color: rgba(255,112,191,0.5);
    box-shadow: 0 0 40px rgba(255,112,191,0.1);
}
.safe {
    background: linear-gradient(135deg, rgba(70,44,125,0.3), rgba(131,28,145,0.15));
    border-color: rgba(131,28,145,0.5);
    box-shadow: 0 0 40px rgba(131,28,145,0.1);
}
.result-rank { font-size: 0.78rem; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 0.6rem; }
.rank-danger { color: #FF70BF; }
.rank-safe { color: #D552A3; }
.result-headline { font-size: 2.6rem; font-weight: 700; letter-spacing: -1px; margin-bottom: 0.3rem; }
.headline-danger { color: #FF70BF; }
.headline-safe { color: #D552A3; }
.result-sub { font-size: 0.95rem; color: rgba(255,255,255,0.45); }

/* Metric mini-cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 1.2rem; text-align: center;
}
.metric-card .label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 2px; color: rgba(255,255,255,0.4); margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2.4rem; font-weight: 700; letter-spacing: -1px;
}
.stay-val { color: #D552A3; }
.leave-val { color: #FF70BF; }

/* Expander */
.stExpander {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important; margin-top: 1rem !important;
}
.stExpander summary p {
    color: rgba(255,255,255,0.45) !important; font-size: 0.82rem !important;
}
.stExpander p, .stExpander span, .stExpander .stText {
    color: rgba(255,255,255,0.55) !important; font-size: 0.82rem !important;
}

/* ── Mobile ── */
@media (max-width: 640px) {
    .main .block-container { padding: 1.2rem 0.9rem !important; margin: 0.5rem !important; border-radius: 18px; }
    h1 { font-size: 2rem !important; }
    .result-headline { font-size: 1.7rem; }
    .metric-card .value { font-size: 1.8rem; }
    div[data-testid="stButton"] > button { font-size: 1rem !important; padding: 0.9rem 1.5rem !important; }
}
            .st-ae.st-bu.st-cf.st-cg.st-ch.st-af.st-ci.st-cj.st-ag.st-bm.st-bn.st-bo.st-bp {
    color: white;
}

p {
    color: white;
}
            ul.st-f1.st-bu.st-cs.st-ct.st-cl.st-cv.st-cw.st-cx.st-b8.st-b9.st-bz.st-c0.st-c1.st-c2.st-cr.st-f8.st-f9 {
    background-color: #1f0a30;
}

</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════
st.markdown('<p class="badge">PREDICTIVE ANALYTICS</p>', unsafe_allow_html=True)
st.title("Churn Predictor")
st.caption("Enter customer details to predict the likelihood of churn")
st.divider()

st.markdown('<p class="badge">CUSTOMER PROFILE</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=650)
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=1)

with col3:
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    gender = st.radio("Gender", options=["Female", "Male"], horizontal=True)
    has_crcard = st.checkbox("Has Credit Card", value=True)
    is_active = st.checkbox("Is Active Member", value=True)

st.write("")
predict_btn = st.button("Predict Churn", type="primary", width='stretch')

# ── Results ──
if predict_btn:
    scaled_input = prepare_input(
        credit_score, age, tenure, balance, num_products,
        has_crcard, is_active, estimated_salary,
        geography, gender
    )

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0]

    input_data = pd.DataFrame([{
        'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': num_products,
        'HasCrCard': int(has_crcard), 'IsActiveMember': int(is_active),
        'EstimatedSalary': estimated_salary,
        'Geography': geography, 'Gender': gender,
    }])

    st.divider()
    st.markdown('<p class="badge">RESULT</p>', unsafe_allow_html=True)

    is_churn = prediction == 1
    css_box = "danger" if is_churn else "safe"
    css_rank = "rank-danger" if is_churn else "rank-safe"
    css_head = "headline-danger" if is_churn else "headline-safe"
    rank = "High Risk" if is_churn else "Low Risk"
    headline = "LIKELY TO CHURN" if is_churn else "LIKELY TO STAY"
    sub = "This customer is at risk of leaving the bank" if is_churn else "This customer is expected to remain with the bank"

    st.markdown(f"""
    <div class="result-box {css_box}">
        <div class="result-rank {css_rank}">{rank}</div>
        <div class="result-headline {css_head}">{headline}</div>
        <div class="result-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Stay Probability</div>
            <div class="value stay-val">{prob[0]:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Churn Probability</div>
            <div class="value leave-val">{prob[1]:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Technical Details"):
        st.text(f"Raw Prediction: {prediction}")
        st.text(f"Probabilities: [{prob[0]:.6f}  {prob[1]:.6f}]")
        st.dataframe(input_data, width='stretch')
