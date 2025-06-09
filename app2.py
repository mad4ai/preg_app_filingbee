import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pickle

# --- Load model and feature order ---
model = joblib.load("xgboost_high_risk_model.pkl")
with open("model_features.pkl", "rb") as f:
    feature_order = pickle.load(f)

# --- Define UI inputs ---
st.title("High-Risk Pregnancy Predictor")

st.header("Enter Pregnancy Details")

def user_input_features():
    d = {}

    # Demographics
    d['AGE'] = st.slider("Mother's Age", 12.0, 55.0, 25.0)
    d['HEIGHT'] = st.slider("Height (cm)", 100.0, 200.0, 155.0)
    d['WEIGHT'] = st.slider("Weight (kg)", 30.0, 150.0, 55.0)

    # Pregnancy history
    for field in ['GRAVIDA', 'PARITY', 'ABORTIONS', 'LIVE', 'DEATH']:
        d[field] = st.number_input(field, min_value=0, max_value=15, value=0)

    # Blood pressure and vitals
    d['BP'] = st.slider("BP (mmHg)", 70.0, 200.0, 120.0)
    d['BP1'] = st.slider("BP1 (second reading)", 70.0, 200.0, 80.0)
    d['HEART_RATE'] = st.slider("Heart Rate", 40, 200, 80)
    d['PULSE_RATE'] = st.slider("Pulse Rate", 40.0, 200.0, 80.0)
    d['RESPIRATORY_RATE'] = st.slider("Respiratory Rate", 10.0, 50.0, 18.0)
    d['FEVER'] = st.slider("Temperature (°C)", 35.0, 41.0, 36.5)

    # Blood work
    d['HEMOGLOBIN'] = st.slider("Hemoglobin (g/dL)", 2.0, 18.0, 11.0)
    d['BLOOD_SUGAR'] = st.slider("Random Blood Sugar (mg/dL)", 50.0, 500.0, 100.0)
    d['FASTING'] = st.slider("Fasting Sugar (mg/dL)", 50, 200, 90)
    d['POST_PRANDIAL'] = st.slider("Post Prandial (mg/dL)", 50, 300, 120)
    d['OGTT_2_HOURS'] = st.slider("OGTT after 2 hours (mg/dL)", 50.0, 300.0, 120.0)
    d['OGTT_GDM'] = st.selectbox("OGTT GDM Diagnosis", [0, 1])

    # Supplements
    for col in ['IFA_QUANTITY', 'CALC_QUANTITY', 'FOLIC_QUANTITY', 'ALB_QUANTITY']:
        d[col] = st.slider(f"{col.replace('_', ' ').title()} (tablets)", 0.0, 500.0, 100.0)

    # Infections & screening results
    for test in ['VDRL_RESULT', 'HIV_RESULT', 'HBSAG_RESULT', 'HEP_RESULT', 'THYROID', 'RH_NEGATIVE', 'SYPHYLIS']:
        d[test] = st.selectbox(f"{test.replace('_', ' ').title()}", [0, 1])

    # Urine tests
    d['URINE_SUGAR'] = st.selectbox("Urine Sugar", [0, 1])
    d['URINE_ALBUMIN'] = st.selectbox("Urine Albumin", [0, 1])

    # Scan/Clinical findings
    for col in ['USG_SCAN', 'MAL_PRESENT', 'PLACENTA', 'FOETAL_POSITION']:
        d[col] = st.selectbox(col.replace("_", " ").title(), [0, 1])

    d['UTERUS_SIZE'] = st.number_input("Uterus Size (weeks)", 0, 50, 20)

    # Risk scores
    d['PHQ_SCORE'] = st.slider("PHQ Score (Depression)", 0, 27, 5)
    d['GAD_SCORE'] = st.slider("GAD Score (Anxiety)", 0, 21, 5)

    # Treatment/Medication
    d['TT_GIVEN'] = st.selectbox("TT Given", [0, 1])
    d['IRON_SUCROSE_DOSE'] = st.slider("Iron Sucrose Dose (mg)", 0, 500, 100)

    # Systemic Disease
    d['SYS_DISEASE_encoded'] = st.selectbox("Any Systemic Disease", [0, 1])

   

    return pd.DataFrame([d])

# --- Collect input and make prediction ---
X_input = user_input_features()
X_input = X_input[feature_order]  # ensure correct order

if st.button("Predict Risk"):
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction")
    st.write("🚨 High-Risk Pregnancy" if pred == 1 else "✅ Low-Risk Pregnancy")
    st.write(f"Probability: `{prob:.2%}`")

    # --- SHAP Explainability ---
st.subheader("Model Explainability")

explainer = shap.Explainer(model)
shap_values = explainer(X_input)

# Create a matplotlib figure manually to avoid deprecated Streamlit behavior
import matplotlib.pyplot as plt

fig = plt.figure()
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
st.pyplot(fig, bbox_inches='tight')

