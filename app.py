import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("energy_model.pkl")

st.title("🔋 Energy Consumption Predictor")

st.sidebar.header("Building Details")

building_area = st.sidebar.number_input("Building Area", 500, 500000)
peak_demand = st.sidebar.number_input("Peak Electric Demand", 0.0, 10000.0)
gas_usage = st.sidebar.number_input("Natural Gas Usage", 0.0, 50000.0)
energy_intensity = st.sidebar.number_input("Energy Use Intensity", 0.0, 500.0)

input_df = pd.DataFrame({
    'Building Area': [building_area],
    'Peak Electric Demand': [peak_demand],
    'Natural Gas Usage': [gas_usage],
    'Energy Use Intensity': [energy_intensity]
})

if st.button("Predict Electricity Usage"):
    log_pred = model.predict(input_df)
    prediction = np.expm1(log_pred)
    st.success(f"Predicted Electricity Usage: {prediction[0]:,.0f}")