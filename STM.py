import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('STM.joblib')
le = joblib.load('STM_le.joblib')

st.title('Customer Churn Prediction')
st.header('Customer Information')

st.subheader('Enter the Following to Predict If the Customer is likely to Churn or not')

TENURE = st.number_input('Tenure', min_value=0, max_value=20, value=5)
MONTANT = st.number_input('Current Top-Up Amount', min_value=0.0, max_value=250000.0, value = 100.0)
FREQUENCE_RECH = st.number_input('Number of Top-Ups in the Last 30 Days', min_value=0, max_value=150, value=5)
REVENUE = st.number_input('Monthly Revenue', min_value=100.0, max_value=250000.0, value=1000.0)
ARPU_SEGMENT = st.number_input('Average 3 Months Revenue', min_value=0.0, max_value=250000.0, value=1000.0)
FREQUENCE = st.number_input('Number of Times the Customer Have Made a Call in Last 30 Days', min_value=0, max_value=100, value=5)
DATA_VOLUME = st.number_input('Rate of Internet Usage', min_value=0.0, max_value=250000.0, value=1000.0)
ON_NET = st.number_input('Total Number of Calls Made to and fro on the Same Network', min_value=0, max_value=25000, value=100)
ORANGE = st.number_input('Total Number of Calls Made to Orange Network', min_value=0, max_value=6000, value=100)
TIGO = st.number_input('Total Number of Calls Made to Tigo Network', min_value=0, max_value=2500, value=100)
ZONE1 = st.number_input('Total Number of Calls Made to Zone 1', min_value=0, max_value=2000, value=100)
ZONE2 = st.number_input('Total Number of Calls Made to Zone 2', min_value=0, max_value=2000, value=100)
REGULARITY = st.number_input('Number of Days the Customer is Active in the Last 90 Days', min_value=0, max_value=90, value=30)
FREQ_TOP_PACK = st.number_input('Number of Times the Customer has Activated the Top Packages', min_value= 0, max_value=1000, value=50)

input_data = {
    'TENURE': TENURE,
    'MONTANT': MONTANT,
    'FREQUENCE_RECH': FREQUENCE_RECH,
    'REVENUE': REVENUE,
    'ARPU_SEGMENT': ARPU_SEGMENT,
    'FREQUENCE': FREQUENCE,
    'DATA_VOLUME': DATA_VOLUME,
    'ON_NET': ON_NET,
    'ORANGE': ORANGE,
    'TIGO': TIGO,
    'ZONE1': ZONE1,
    'ZONE2': ZONE2,
    'REGULARITY': REGULARITY,
    'FREQ_TOP_PACK': FREQ_TOP_PACK
}

def churn_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    cols = []
    for col in cols:
        input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)
    return prediction

if st.button('Check Churn'):
    prediction = churn_prediction(input_data)
    if prediction == 0:
        st.success('This customer has an active relationship with the company')
    elif prediction == 1:
        st.error('This customer is at risk of leaving the company')
else:
    st.info('Click the button to check the churn status')