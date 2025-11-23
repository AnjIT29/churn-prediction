import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Load model & feature order
# ---------------------------
model = joblib.load("final_model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.title("💳 Bank Customer Churn Prediction App")
st.write("Predict whether a customer will leave the bank (churn) using machine learning.")

# ---------------------------
# User Input Form
# ---------------------------
st.subheader("📌 Enter Customer Details")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=3)
Balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
NumOfProducts = st.number_input("Number of Bank Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card?", ["No", "Yes"])
IsActiveMember = st.selectbox("Is Active Member?", ["No", "Yes"])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# One-hot encodings
st.write("### 🌍 Geography")
geo = st.selectbox("Select Country", ["France", "Germany", "Spain"])

st.write("### 👤 Gender")
gender = st.selectbox("Select Gender", ["Male", "Female"])

# Convert categorical fields to numeric one-hot
France = 1 if geo == "France" else 0
Germany = 1 if geo == "Germany" else 0
Spain = 1 if geo == "Spain" else 0

Male = 1 if gender == "Male" else 0
Female = 1 if gender == "Female" else 0

HasCrCard = 1 if HasCrCard == "Yes" else 0
IsActiveMember = 1 if IsActiveMember == "Yes" else 0

# ---------------------------
# Create input vector in correct order
# ---------------------------
user_data = np.array([[
    CreditScore, Age, Tenure, Balance, NumOfProducts,
    HasCrCard, IsActiveMember, EstimatedSalary,
    France, Germany, Spain, Female, Male
]])

# ---------------------------
# Prediction button
# ---------------------------
if st.button("🔍 Predict Churn"):

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    st.subheader("🔎 Prediction Result:")

    if prediction == 1:
        st.error(f"⚠️ The customer is **LIKELY to CHURN** (leave the bank). Probability: {probability:.2f}")
    else:
        st.success(f"🟢 The customer will **STAY**. Probability of churn: {probability:.2f}")

