import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =========================
# Load Model Files
# =========================

model = pickle.load(open('model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))

features = pickle.load(open('features.pkl', 'rb'))

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>

    .stButton>button {
        width: 100%;
        background-color: #6C63FF;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Title
# =========================

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details below:")

# =========================
# User Inputs
# =========================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=10000.0,
    value=100.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=100000.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

# =========================
# Prediction Button
# =========================

if st.button("Predict Churn"):

    # Create all features with default 0
    input_dict = dict.fromkeys(features, 0)

    # =========================
    # Numerical Features
    # =========================

    if 'tenure' in input_dict:
        input_dict['tenure'] = tenure

    if 'MonthlyCharges' in input_dict:
        input_dict['MonthlyCharges'] = monthly_charges

    if 'TotalCharges' in input_dict:
        input_dict['TotalCharges'] = total_charges

    if 'SeniorCitizen' in input_dict:
        input_dict['SeniorCitizen'] = 0

    # =========================
    # Gender Encoding
    # =========================

    if 'gender_Male' in input_dict:
        input_dict['gender_Male'] = 1 if gender == "Male" else 0

    # =========================
    # Contract Encoding
    # =========================

    if 'Contract_One year' in input_dict:
        input_dict['Contract_One year'] = 1 if contract == "One year" else 0

    if 'Contract_Two year' in input_dict:
        input_dict['Contract_Two year'] = 1 if contract == "Two year" else 0

    # =========================
    # Internet Service Encoding
    # =========================

    if 'InternetService_Fiber optic' in input_dict:
        input_dict['InternetService_Fiber optic'] = (
            1 if internet_service == "Fiber optic" else 0
        )

    if 'InternetService_No' in input_dict:
        input_dict['InternetService_No'] = (
            1 if internet_service == "No" else 0
        )

    # =========================
    # Payment Method Encoding
    # =========================

    if 'PaymentMethod_Electronic check' in input_dict:
        input_dict['PaymentMethod_Electronic check'] = (
            1 if payment_method == "Electronic check" else 0
        )

    if 'PaymentMethod_Mailed check' in input_dict:
        input_dict['PaymentMethod_Mailed check'] = (
            1 if payment_method == "Mailed check" else 0
        )

    if 'PaymentMethod_Credit card (automatic)' in input_dict:
        input_dict['PaymentMethod_Credit card (automatic)'] = (
            1 if payment_method == "Credit card" else 0
        )

    # =========================
    # Tech Support Encoding
    # =========================

    if 'TechSupport_Yes' in input_dict:
        input_dict['TechSupport_Yes'] = (
            1 if tech_support == "Yes" else 0
        )

    # =========================
    # Online Security Encoding
    # =========================

    if 'OnlineSecurity_Yes' in input_dict:
        input_dict['OnlineSecurity_Yes'] = (
            1 if online_security == "Yes" else 0
        )

    # =========================
    # Convert to DataFrame
    # =========================

    input_df = pd.DataFrame([input_dict])

    # =========================
    # Scale Input
    # =========================

    input_scaled = scaler.transform(input_df)

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]

    # =========================
    # Output
    # =========================

    if prediction[0] == 1:

        st.error("❗ Customer is likely to CHURN")

    else:

        st.success("✅ Customer will NOT churn")

    st.info(
        f"ℹ️ Model Prediction (0 = Not Churn, 1 = Churn): {prediction[0]}"
    )

    st.warning(
        f"📉 Churn Probability: {probability:.2f}"
    )