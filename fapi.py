from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from pydantic import BaseModel
# -----------------------------
# Load model and scaler
# -----------------------------
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# FastAPI initialization
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API")

# -----------------------------
# Serve static HTML
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
# -----------------------------
# Pydantic model for input
# -----------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    PaperlessBilling: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

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
# Preprocessing function
# -----------------------------
def preprocess_input(data: CustomerData):
    # Convert SeniorCitizen Yes/No to 1/0
    SeniorCitizen_val = 1 if data.SeniorCitizen == "Yes" else 0

    # Scale numerical features
    num_features = np.array([[data.tenure, data.MonthlyCharges, data.TotalCharges]])
    num_features_scaled = scaler.transform(num_features)

    # Encode categorical features
    data_dict = {
        'gender_Male': 1 if data.gender=="Male" else 0,
        'Partner_Yes': 1 if data.Partner=="Yes" else 0,
        'Dependents_Yes': 1 if data.Dependents=="Yes" else 0,
        'PhoneService_Yes': 1 if data.PhoneService=="Yes" else 0,
        'MultipleLines_No phone service': 1 if data.MultipleLines=="No phone service" else 0,
        'MultipleLines_Yes': 1 if data.MultipleLines=="Yes" else 0,
        'InternetService_Fiber optic': 1 if data.InternetService=="Fiber optic" else 0,
        'InternetService_No': 1 if data.InternetService=="No" else 0,
        'OnlineSecurity_No internet service': 1 if data.OnlineSecurity=="No internet service" else 0,
        'OnlineSecurity_Yes': 1 if data.OnlineSecurity=="Yes" else 0,
        'OnlineBackup_No internet service': 1 if data.OnlineBackup=="No internet service" else 0,
        'OnlineBackup_Yes': 1 if data.OnlineBackup=="Yes" else 0,
        'DeviceProtection_No internet service': 1 if data.DeviceProtection=="No internet service" else 0,
        'DeviceProtection_Yes': 1 if data.DeviceProtection=="Yes" else 0,
        'TechSupport_No internet service': 1 if data.TechSupport=="No internet service" else 0,
        'TechSupport_Yes': 1 if data.TechSupport=="Yes" else 0,
        'StreamingTV_No internet service': 1 if data.StreamingTV=="No internet service" else 0,
        'StreamingTV_Yes': 1 if data.StreamingTV=="Yes" else 0,
        'StreamingMovies_No internet service': 1 if data.StreamingMovies=="No internet service" else 0,
        'StreamingMovies_Yes': 1 if data.StreamingMovies=="Yes" else 0,
        'Contract_One year': 1 if data.Contract=="One year" else 0,
        'Contract_Two year': 1 if data.Contract=="Two year" else 0,
        'PaperlessBilling_Yes': 1 if data.PaperlessBilling=="Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if data.PaymentMethod=="Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if data.PaymentMethod=="Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if data.PaymentMethod=="Mailed check" else 0
    }

    # Combine features in correct order
    final_features = []
    for feat in model_features:
        if feat == 'tenure':
            final_features.append(num_features_scaled[0][0])
        elif feat == 'MonthlyCharges':
            final_features.append(num_features_scaled[0][1])
        elif feat == 'TotalCharges':
            final_features.append(num_features_scaled[0][2])
        elif feat == 'SeniorCitizen':
            final_features.append(SeniorCitizen_val)
        else:
            final_features.append(data_dict[feat])

    return np.array([final_features])

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_churn(data: CustomerData):
    features = preprocess_input(data)
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0][1]

    result = {
        "prediction": "Churn" if prediction==1 else "Stay",
        "probability": round(float(prediction_proba), 2)
    }
    return result

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API. Use /predict endpoint with JSON data."}
