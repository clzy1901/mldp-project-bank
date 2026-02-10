import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load trained model and feature columns
# --------------------------------------------------
model = joblib.load("best_random_forest_model.pkl")
feature_cols = joblib.load("bank_feature_cols_v2.pkl")

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
    probability = model.predict_proba(df_input)[0][1]

    # Display result
    if prediction == "yes":
        st.success(f"Prediction: YES — Customer likely to subscribe 💰")
    else:
        st.warning(f"Prediction: NO — Customer unlikely to subscribe")

    st.write(f"Probability of subscription: **{probability:.2%}**")

# --------------------------------------------------
# Page Styling (Optional)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
