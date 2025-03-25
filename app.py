import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📈", layout="wide")

st.title("📊 AI-Powered Revenue Forecasting")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file (with 'Date' and 'Revenue' columns)", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Ensure required columns exist
    if "Date" in df.columns and "Revenue" in df.columns:
        df = df.rename(columns={"Date": "ds", "Revenue": "y"})
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Prophet model
        model = Prophet()
        model.fit(df)
        
        # Forecasting
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast, ax=ax)
        st.pyplot(fig)
        
        st.subheader("📅 Forecasted Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30])
    else:
        st.error("The uploaded file must contain 'Date' and 'Revenue' columns.")
else:
    st.info("Please upload an Excel file to generate forecasts.")
