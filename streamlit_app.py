import os
import streamlit as st
import joblib
import pandas as pd

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
# Load trained model and feature columns (BEST CASE by default)
# --------------------------------------------------
BEST_MODEL_PATH = "bank_model_v2_rf_best.pkl"
BEST_COLS_PATH  = "bank_feature_cols_v2.pkl"

model, feature_cols = load_artifacts(BEST_MODEL_PATH, BEST_COLS_PATH)

# --------------------------------------------------
# Streamlit App UI
# --------------------------------------------------
st.title("Bank Term Deposit Subscription Prediction")

st.write(
    "This app predicts whether a customer is likely to subscribe "
    "to a term deposit based on customer and campaign information."
)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
age = st.slider("Age", 18, 95, 35)
balance = st.number_input("Account Balance", value=1000)
duration = st.number_input("Last Contact Duration (seconds)", value=180)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
previously_contacted = st.selectbox(
    "Previously Contacted Before?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

job = st.selectbox(
    "Job Type",
    ["admin.", "blue-collar", "entrepreneur", "management",
     "retired", "self-employed", "services", "student",
     "technician", "unemployed", "unknown"]
)

marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
month = st.selectbox(
    "Last Contact Month",
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
)

# --------------------------------------------------
# Predict Button
# --------------------------------------------------
if st.button("Predict Subscription"):

    # Create input dataframe
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

    # One-hot encode categorical variables
    df_input = pd.get_dummies(df_input, drop_first=True)

    # Align input features with training features
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)

    # Make prediction
    prediction = model.predict(df_input)[0]

    # Probability (only if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df_input)[0][1]
    else:
        probability = None

    # Display result (handles either "yes"/"no" or 1/0)
    pred_str = str(prediction).lower()
    if pred_str in ["yes", "1", "true"]:
        st.success("Prediction: YES — Customer likely to subscribe 💰")
    else:
        st.warning("Prediction: NO — Customer unlikely to subscribe")

    if probability is not None:
        st.write(f"Probability of subscription: **{probability:.2%}**")

# --------------------------------------------------
# Page Styling (Optional)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f9fc; }
    </style>
    """,
    unsafe_allow_html=True
)
