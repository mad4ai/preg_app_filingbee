import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load model and feature columns
model = joblib.load('xgboost_high_risk_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')  # list of features used during training

st.title("High-Risk Pregnancy Prediction App")

st.title("helps ANMs and  Govt")


input_data = {}

# Numerical inputs with sliders
input_data['GRAVIDA'] = st.slider('Gravida (Number of pregnancies)', min_value=0, max_value=15, value=1, step=1)
input_data['PARITY'] = st.slider('Parity (Number of births)', min_value=0, max_value=15, value=0, step=1)
input_data['ABORTIONS'] = st.slider('Abortions', min_value=0, max_value=10, value=0, step=1)
input_data['LIVE'] = st.slider('Live Births', min_value=0, max_value=15, value=0, step=1)
input_data['DEATH'] = st.slider('Previous Deaths', min_value=0, max_value=5, value=0, step=1)

input_data['AGE'] = st.slider('Age (years)', min_value=10, max_value=60, value=25, step=1)
input_data['HEIGHT'] = st.slider('Height (cm)', min_value=130, max_value=210, value=160, step=1)
input_data['WEIGHT'] = st.slider('Weight (kg)', min_value=30, max_value=150, value=60, step=1)

input_data['BP'] = st.slider('Systolic Blood Pressure', min_value=70, max_value=200, value=120, step=1)
input_data['BP1'] = st.slider('Diastolic Blood Pressure', min_value=40, max_value=130, value=80, step=1)

input_data['HEMOGLOBIN'] = st.slider('Hemoglobin (g/dL)', min_value=4.0, max_value=18.0, value=12.0, step=0.1)
input_data['BLOOD_SUGAR'] = st.slider('Blood Sugar (mg/dL)', min_value=40, max_value=500, value=90, step=1)
input_data['FASTING'] = st.slider('Fasting Blood Sugar (mg/dL)', min_value=40, max_value=250, value=85, step=1)
input_data['POST_PRANDIAL'] = st.slider('Post Prandial Blood Sugar (mg/dL)', min_value=60, max_value=350, value=120, step=1)
input_data['OGTT_2_HOURS'] = st.slider('OGTT 2 Hours (mg/dL)', min_value=60, max_value=300, value=140, step=1)

input_data['IFA_QUANTITY'] = st.slider('IFA Quantity (tablets)', min_value=0, max_value=200, value=30, step=1)
input_data['CALC_QUANTITY'] = st.slider('Calcium Quantity (tablets)', min_value=0, max_value=200, value=30, step=1)
input_data['FOLIC_QUANTITY'] = st.slider('Folic Acid Quantity (tablets)', min_value=0, max_value=200, value=30, step=1)
input_data['ALB_QUANTITY'] = st.slider('Alb Quantity (tablets)', min_value=0, max_value=200, value=0, step=1)
input_data['IRON_SUCROSE_DOSE'] = st.slider('Iron Sucrose Dose (mg)', min_value=0, max_value=500, value=0, step=10)

input_data['URINE_SUGAR'] = st.selectbox('Urine Sugar Present', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
input_data['URINE_ALBUMIN'] = st.selectbox('Urine Albumin Present', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

input_data['UTERUS_SIZE'] = st.slider('Uterus Size (weeks)', min_value=0, max_value=45, value=20, step=1)
input_data['FOETAL_POSITION'] = st.selectbox('Fetal Position', options=[0, 1], format_func=lambda x: 'Normal' if x == 0 else 'Abnormal')
input_data['MAL_PRESENT'] = st.selectbox('Malpresentation Present', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
input_data['PLACENTA'] = st.selectbox('Placenta Issues', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
input_data['USG_SCAN'] = st.selectbox('Ultrasound Scan Done', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
input_data['TT_GIVEN'] = st.selectbox('TT Vaccine Given', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

input_data['PHQ_SCORE'] = st.slider('PHQ (Patient Health Questionnaire) Score', min_value=0, max_value=27, value=5, step=1)
input_data['GAD_SCORE'] = st.slider('GAD (General Anxiety Disorder) Score', min_value=0, max_value=21, value=5, step=1)

input_data['PULSE_RATE'] = st.slider('Pulse Rate (beats per minute)', min_value=40, max_value=150, value=75, step=1)
input_data['RESPIRATORY_RATE'] = st.slider('Respiratory Rate (breaths per minute)', min_value=10, max_value=40, value=16, step=1)

input_data['FEVER'] = st.selectbox('Fever Present', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

input_data['Weeks_Pregnant'] = st.slider('Weeks Pregnant', min_value=0, max_value=45, value=20, step=1)

input_data['SYS_DISEASE_encoded'] = st.selectbox('Systemic Disease', options=[0, 1, 2, 3], format_func=lambda x: ['None', 'Hypertension', 'Diabetes', 'Other'][x])

# Fix spelling to match model training
input_data['SYPHILIS'] = st.selectbox('Syphilis', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Prediction button
if st.button("Predict High-Risk Pregnancy"):
    input_df = pd.DataFrame([input_data])
    
    # Ensure all feature columns present and ordered
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    label = "High Risk" if prediction == 1 else "Low Risk"
    st.success(f"Prediction: {label}")
