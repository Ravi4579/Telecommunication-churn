import pickle
import streamlit as st
import numpy as np

# -----------------------------
# Load model and scaler (from same folder)
# -----------------------------
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")
st.markdown("This app predicts whether a customer is likely to **Churn** or **Stay**.")

# -----------------------------
# Input fields using columns for compact layout
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)

# Convert SeniorCitizen to 0/1 internally
SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0

# -----------------------------
# Model feature order
# -----------------------------
model_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_input():
    # Scale numerical features
    num_features = np.array([[tenure, MonthlyCharges, TotalCharges]])
    num_features_scaled = scaler.transform(num_features)

    # Encode categorical features
    data_dict = {
        'gender_Male': 1 if gender=="Male" else 0,
        'Partner_Yes': 1 if Partner=="Yes" else 0,
        'Dependents_Yes': 1 if Dependents=="Yes" else 0,
        'PhoneService_Yes': 1 if PhoneService=="Yes" else 0,
        'MultipleLines_No phone service': 1 if MultipleLines=="No phone service" else 0,
        'MultipleLines_Yes': 1 if MultipleLines=="Yes" else 0,
        'InternetService_Fiber optic': 1 if InternetService=="Fiber optic" else 0,
        'InternetService_No': 1 if InternetService=="No" else 0,
        'OnlineSecurity_No internet service': 1 if OnlineSecurity=="No internet service" else 0,
        'OnlineSecurity_Yes': 1 if OnlineSecurity=="Yes" else 0,
        'OnlineBackup_No internet service': 1 if OnlineBackup=="No internet service" else 0,
        'OnlineBackup_Yes': 1 if OnlineBackup=="Yes" else 0,
        'DeviceProtection_No internet service': 1 if DeviceProtection=="No internet service" else 0,
        'DeviceProtection_Yes': 1 if DeviceProtection=="Yes" else 0,
        'TechSupport_No internet service': 1 if TechSupport=="No internet service" else 0,
        'TechSupport_Yes': 1 if TechSupport=="Yes" else 0,
        'StreamingTV_No internet service': 1 if StreamingTV=="No internet service" else 0,
        'StreamingTV_Yes': 1 if StreamingTV=="Yes" else 0,
        'StreamingMovies_No internet service': 1 if StreamingMovies=="No internet service" else 0,
        'StreamingMovies_Yes': 1 if StreamingMovies=="Yes" else 0,
        'Contract_One year': 1 if Contract=="One year" else 0,
        'Contract_Two year': 1 if Contract=="Two year" else 0,
        'PaperlessBilling_Yes': 1 if PaperlessBilling=="Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod=="Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod=="Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod=="Mailed check" else 0
    }

    # Combine features in exact order
    final_features = []
    for feat in model_features:
        if feat == 'tenure':
            final_features.append(num_features_scaled[0][0])
        elif feat == 'MonthlyCharges':
            final_features.append(num_features_scaled[0][1])
        elif feat == 'TotalCharges':
            final_features.append(num_features_scaled[0][2])
        elif feat == 'SeniorCitizen':
            final_features.append(SeniorCitizen)
        else:
            final_features.append(data_dict[feat])

    return np.array([final_features])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    features = preprocess_input()
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"Customer is likely to **Churn**. Probability: {prediction_proba:.2f}")
    else:
        st.success(f"Customer is likely to **Stay**. Probability: {prediction_proba:.2f}")
