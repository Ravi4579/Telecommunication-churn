Customer Churn Prediction
📌 Project Overview

This project predicts whether a customer will churn (leave) or stay with a telecom company using machine learning. The goal is to help businesses identify at-risk customers and take proactive actions.

📊 Exploratory Data Analysis (EDA)

During EDA, we:

Checked and handled missing values, especially in TotalCharges.

Explored categorical features such as Contract, PaymentMethod, and InternetService.

Converted categorical features into numeric form using one-hot encoding.

Normalized numerical features like tenure, MonthlyCharges, and TotalCharges using MinMaxScaler.

Analyzed feature importance and the distribution of churn.

🤖 Machine Learning Models

We trained and evaluated multiple models:

XGBoost Classifier (XGB) – Selected for deployment due to high accuracy.

Extra models tested but not deployed:

LightGBM (LGB)

Decision Tree (DT)

Random Forest (RF)

Logistic Regression with L1 regularization (Lasso)

🔧 Features Used

Numerical: tenure, MonthlyCharges, TotalCharges, SeniorCitizen

Categorical (one-hot encoded): gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

🚀 Deployment
uvicorn fapi:app --reload


Streamlit App – Interactive UI to input customer data and get churn prediction.

FastAPI + HTML Frontend – API endpoint /predict for predictions and a responsive index.html UI.
