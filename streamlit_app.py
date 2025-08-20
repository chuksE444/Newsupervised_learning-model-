# streamlit_app_full.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("stroke_prediction_model.pkl")

st.title("Stroke Risk Prediction App")
st.write("Enter patient details to predict stroke risk.")

# ------------------------
# User Inputs
# ------------------------
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# ------------------------
# Feature Engineering
# ------------------------
# Age group
if age < 30:
    age_group = "Young"
elif age < 60:
    age_group = "Adult"
else:
    age_group = "Senior"

# BMI group
if bmi < 18.5:
    bmi_group = "Underweight"
elif bmi < 25:
    bmi_group = "Normal"
elif bmi < 30:
    bmi_group = "Overweight"
else:
    bmi_group = "Obese"

# ------------------------
# Convert categorical inputs to same format used in training
# ------------------------
def encode_features(df):
    # Map binary features
    df['ever_married'] = df['ever_married'].map({"Yes": 1, "No": 0})
    df['gender'] = df['gender'].map({"Male": 1, "Female": 0})
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["work_type", "Residence_type", "smoking_status", "age_group", "bmi_group"], drop_first=True)
    
    # Ensure all expected columns exist (fill missing dummy columns with 0)
    expected_cols = model.feature_names_in_  # uses model's training features
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]  # ensure column order matches training
    return df

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Stroke Risk"):
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status],
        "age_group": [age_group],
        "bmi_group": [bmi_group]
    })

    # Encode features
    input_df_encoded = encode_features(input_df)

    # Predict
    prediction = model.predict(input_df_encoded)
    probability = model.predict_proba(input_df_encoded)[0][1]

    risk = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"
    st.success(f"Predicted Stroke Risk: {risk}")
    st.info(f"Probability of Stroke: {probability:.2f}")
