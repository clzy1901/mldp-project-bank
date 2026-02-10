import os
import base64
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --------------------------------------------------
# Page config (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    layout="wide"
)

# --------------------------------------------------
# Background helper
# --------------------------------------------------
def set_background(image_path: str):
    """Set a local image as the Streamlit app background (works on deploy too)."""
    if not os.path.exists(image_path):
        st.warning(f"Background image not found: {image_path} (skipping background)")
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Your background file name (place background.jpg in the SAME folder as this .py file)
set_background("background.jpg")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_artifacts(model_path: str, cols_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not os.path.exists(cols_path):
        st.error(f"Feature columns file not found: {cols_path}")
        st.stop()

    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    return model, feature_cols

# --------------------------------------------------
# Load trained model and feature columns
# --------------------------------------------------
BEST_MODEL_PATH = "bank_model_v2_rf_best.pkl"
BEST_COLS_PATH  = "bank_feature_cols_v2.pkl"

model, feature_cols = load_artifacts(BEST_MODEL_PATH, BEST_COLS_PATH)

# --------------------------------------------------
# Page Styling (polished + readable over background)
# --------------------------------------------------
st.markdown("""
<style>
/* Make the main content readable on top of a dark background image */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: rgba(255, 255, 255, 0.60);
    border-radius: 16px;
}

/* Headings/text */
h1, h2, h3, p, label, .stMarkdown { color: #111111 !important; }

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div { background-color: #1f2937 !important; }

div[data-baseweb="select"] span,
div[data-baseweb="input"] input { color: #ffffff !important; }

/* Button */
div.stButton > button {
    background-color: #1f2937 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
}
div.stButton > button, div.stButton > button * { color: #ffffff !important; }
div.stButton > button:hover { background-color: #374151 !important; }

/* Subheader spacing */
h3 { margin-top: 1.2rem; }

/* Result area – OPAQUE and high-contrast */
.result-box {
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    background: #f4f5f7;
    border: 1px solid rgba(0, 0, 0, 0.12);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

/* Prediction alert boxes (YES / NO) – opaque and readable */
div[data-testid="stAlert"] {
    background-color: #f3f4f6 !important;
    color: #111111 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15) !important;
    font-weight: 650 !important;
}

/* Make predicted probability text stronger */
div[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    font-weight: 650 !important;
    color: #111111 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    color: #111111 !important;
}

/* Caption / note text – BLACK and clear */
.stCaption {
    color: #111111 !important;
    font-size: 0.95rem !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
           
div[data-testid="stCaptionContainer"] p {
    color: #000000 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("Bank Term Deposit Subscription Prediction")
st.write(
    "This app estimates the likelihood that a customer will subscribe to a bank term deposit "
    "based on customer profile and recent campaign interaction."
)

# --------------------------------------------------
# Inputs (cleaner layout)
# --------------------------------------------------
st.subheader("Customer Profile")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", 18, 95, 35)
with col2:
    education = st.selectbox(
        "Highest education level",
        ["Primary", "Secondary", "Tertiary", "Unknown"]
    )

col3, col4 = st.columns(2)
with col3:
    job = st.selectbox(
        "Occupation / job category",
        [
            "Admin", "Blue-collar", "Entrepreneur", "Management",
            "Retired", "Self-employed", "Services", "Student",
            "Technician", "Unemployed", "Unknown"
        ]
    )
with col4:
    marital = st.selectbox("Marital status", ["Married", "Single", "Divorced"])

st.subheader("Campaign & Account Details")

col5, col6 = st.columns(2)
with col5:
    balance = st.number_input(
        "Account balance (SGD)",
        value=1000,
        min_value=-100000,
        max_value=1_000_000
    )
with col6:
    duration = st.number_input(
        "Call duration (seconds)",
        value=180,
        min_value=0,
        max_value=5000
    )

col7, col8 = st.columns(2)
with col7:
    campaign = st.number_input(
        "Contact attempts (this campaign)",
        min_value=1,
        value=1,
        help="How many times the bank contacted the customer during the current marketing campaign."
    )
with col8:
    previously_contacted = st.selectbox(
        "Contacted before (past campaigns)?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# --------------------------------------------------
# Hidden defaults (removed from UI for better UX)
# --------------------------------------------------
housing = "No"
loan = "No"
contact = "Cellular"
month = "May"

# --------------------------------------------------
# Fixed decision threshold (50%)
# --------------------------------------------------
threshold = 0.50

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Subscription", key="predict_btn"):

    df_input = pd.DataFrame({
        "age": [age],
        "balance": [balance],
        "duration": [duration],
        "campaign": [campaign],
        "previously_contacted": [previously_contacted],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month]
    })

    # Replicate training feature engineering (log features)
    df_input["log_duration"] = np.log1p(df_input["duration"])
    df_input["log_campaign"] = np.log1p(df_input["campaign"])
    df_input["log_balance"] = np.sign(df_input["balance"]) * np.log1p(np.abs(df_input["balance"]))

    # One-hot encode categorical variables
    df_input = pd.get_dummies(df_input, drop_first=True)

    # Align to training columns
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)

    # Probability for YES class safely
    classes = list(model.classes_)
    if "yes" in classes:
        pos_class = "yes"
    elif 1 in classes:
        pos_class = 1
    else:
        pos_class = classes[-1]

    pos_idx = classes.index(pos_class)
    probability = model.predict_proba(df_input)[0][pos_idx]

    st.markdown("---")
    st.subheader("Result")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.progress(float(probability))

    # ✅ Clear YES/NO with color + emoji
    if probability >= threshold:
        st.success("Prediction: YES — Customer likely to subscribe", icon="👍")
    else:
        st.error("Prediction: NO — Customer unlikely to subscribe", icon="❌")

    st.metric("Predicted probability of subscription", f"{probability:.2%}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Note in BLACK
    st.caption(
        "Note: Due to class imbalance, even high-potential customers may have probabilities below 50%."
    )
    

    # OPTIONAL debug (keep for testing, remove for submission)
    # with st.expander("Debug: active features (non-zero)"):
    #     nz = df_input.loc[0][df_input.loc[0] != 0].sort_values(ascending=False)
    #     st.write(nz)
