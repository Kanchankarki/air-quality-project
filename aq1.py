import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model & scaler
model = load_model("aqi_lstm_log_model.h5",compile=False
)
scaler = joblib.load("scaler1.pkl")

TIME_STEPS = 24
NUM_FEATURES = 2  # O3, PM25_log

# PM2.5 → AQI conversion
def pm25_to_aqi(pm25):
    if pm25 <= 12:
        return pm25 * 50 / 12
    elif pm25 <= 35.4:
        return 51 + (pm25 - 12.1) * 49 / 23.3
    elif pm25 <= 55.4:
        return 101 + (pm25 - 35.5) * 49 / 19.9
    elif pm25 <= 150.4:
        return 151 + (pm25 - 55.5) * 49 / 94.9
    else:
        return 300

def predict_aqi(last_24h):
    """
    last_24h: list of 24 rows → [[O3, PM25], ...]
    """
    data = np.array(last_24h)

    # Apply log transform to PM2.5
    data[:, 1] = np.log1p(data[:, 1])

    # Scale
    scaled = scaler.transform(data)

    # Reshape for LSTM
    X = scaled.reshape(1, TIME_STEPS, NUM_FEATURES)

    # Predict log(PM2.5)
    pm25_log_scaled = model.predict(X, verbose=0)[0][0]

    # Inverse scaling
    dummy = np.zeros((1, NUM_FEATURES))
    dummy[:, 1] = pm25_log_scaled
    pm25_log = scaler.inverse_transform(dummy)[0][1]

    # Inverse log
    pm25 = np.expm1(pm25_log)

    # Convert to AQI
    aqi = pm25_to_aqi(pm25)

    return round(pm25, 2), round(aqi, 2)