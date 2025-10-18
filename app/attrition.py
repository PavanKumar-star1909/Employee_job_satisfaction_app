import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ======================
# Paths
# ======================
project_root = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction'
model_path = os.path.join(project_root, 'models', 'attrition_model.pkl')
features_path = os.path.join(project_root, 'models', 'model_features.pkl')

# ======================
# Load model and feature list
# ======================
model = joblib.load(model_path)
model_features = joblib.load(features_path)

st.title("Employee Attrition Prediction App")
st.write("Enter employee details to predict the likelihood of leaving the company.")

# ======================
# User Input
# ======================
OverTime = st.selectbox("OverTime", ["Yes", "No"])
JobSatisfaction = st.slider("Job Satisfaction (1=Low, 4=Very High)", 1, 4, 3)
WorkLifeBalance = st.slider("Work-Life Balance (1=Low, 4=Very High)", 1, 4, 3)
JobInvolvement = st.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=2)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)

# ======================
# Create input DataFrame
# ======================
input_data = pd.DataFrame({
    'OverTime': [1 if OverTime=='Yes' else 0],
    'JobSatisfaction': [JobSatisfaction],
    'WorkLifeBalance': [WorkLifeBalance],
    'JobInvolvement': [JobInvolvement],
    'YearsAtCompany': [YearsAtCompany],
    'MonthlyIncome': [MonthlyIncome]
})

# ======================
# Feature Engineering (match training)
# ======================
# EngagementScore
input_data['EngagementScore'] = (input_data['JobInvolvement'] + input_data['WorkLifeBalance']) / 2

# TenureCategory
tenure_bins = [-1, 2, 5, 10, 20, 50]
tenure_labels = ['New', 'Mid', 'Experienced', 'Senior', 'Veteran']
input_data['TenureCategory'] = pd.cut(input_data['YearsAtCompany'], bins=tenure_bins, labels=tenure_labels)

# One-hot encode TenureCategory like training
tenure_dummies = pd.get_dummies(input_data['TenureCategory'], drop_first=True)
input_data = pd.concat([input_data, tenure_dummies], axis=1)
input_data.drop('TenureCategory', axis=1, inplace=True)

# ======================
# Align columns with training data
# ======================
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_features]

# ======================
# Prediction
# ======================
if st.button("Predict Attrition"):
    pred = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"Employee is likely to leave! Probability: {pred_proba:.2f}")
    else:
        st.success(f"Employee is likely to stay. Probability of leaving: {pred_proba:.2f}")
