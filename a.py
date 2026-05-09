import streamlit as st
import pandas as pd
from aq1 import predict_aqi

st.set_page_config(page_title="AQI Prediction", layout="centered")

st.title("🌫️ AQI Prediction using LSTM (Log PM2.5)")
st.write("Upload *last 24 hours* of air quality data")

st.markdown("""
*Required CSV format*
- Columns: O3, PM25
- Rows: *Exactly 24*
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if list(df.columns) != ["O3", "PM25"]:
        st.error("CSV must have columns: O3, PM25 (in this order)")
    elif df.shape != (24, 2):
        st.error("CSV must contain exactly 24 rows")
    else:
        st.subheader("📊 Input Data (Last 24 Hours)")
        st.dataframe(df)

        if st.button("Predict Next Hour AQI"):
            pm25, aqi = predict_aqi(df.values.tolist())

            st.success(f"✅ Predicted PM2.5: *{pm25} µg/m³*")
            st.success(f"🌫️ Predicted AQI: *{aqi}*")

            # AQI category
            if aqi <= 50:
                st.info("🟢 Good")
            elif aqi <= 100:
                st.warning("🟡 Moderate")
            elif aqi <= 150:
                st.warning("🟠 Unhealthy for Sensitive Groups")
            else:
                st.error("🔴 Unhealthy")