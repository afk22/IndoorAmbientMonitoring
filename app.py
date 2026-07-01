import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION AND SENSOR SETTINGS
# ==========================================

# 1. Sensor Positions - ÜÇÜNCÜ SENSÖRÜ EKLEDİK
SENSOR_1_POS = [13.0, 18.0]
SENSOR_2_POS = [9.0, 38.0]
SENSOR_3_POS = [19.0, 50.0] # <-- BURAYI KENDİ PLANINIZA GÖRE GÜNCELLEYİN

points = np.array([SENSOR_1_POS, SENSOR_2_POS, SENSOR_3_POS])

# 2. Live API Information
FROST_BASE_URL = "https://gi3.gis.lrg.tum.de/frost/v1.1"
SENSOR_1_IOT_ID = 1713  # Güncellenmiş ID'ler
SENSOR_2_IOT_ID = 1725
SENSOR_3_IOT_ID = 1881  # Üçüncü sensör ID

def fetch_history_from_frost(url, iot_id, count=20):
    try:
        api_url = f"{url}/Datastreams({iot_id})/Observations?$top={count}&$orderby=phenomenonTime%20desc"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("value"):
                return [(float(obs["result"]), obs["phenomenonTime"]) for obs in data["value"]]
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=5)
def get_all_sensor_data():
    default_time = datetime.now().isoformat()
    # 3 sensör için fallback
    s1_data = fetch_history_from_frost(FROST_BASE_URL, SENSOR_1_IOT_ID, count=20) or [(23.5, default_time)] * 20
    s2_data = fetch_history_from_frost(FROST_BASE_URL, SENSOR_2_IOT_ID, count=20) or [(24.2, default_time)] * 20
    s3_data = fetch_history_from_frost(FROST_BASE_URL, SENSOR_3_IOT_ID, count=20) or [(22.8, default_time)] * 20
    
    return s1_data, s2_data, s3_data

# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.set_page_config(page_title="Library Sensor Heatmap", layout="centered")
st.title("Library Spatial Temperature Distribution")

s1_history, s2_history, s3_history = get_all_sensor_data()
max_steps = min(len(s1_history), len(s2_history), len(s3_history))

st.sidebar.subheader("Timeline Control")
time_step = st.sidebar.slider("Go back in time (Steps)", 0, max_steps - 1, 0)

temp_1, time_1 = s1_history[time_step]
temp_2, time_2 = s2_history[time_step]
temp_3, time_3 = s3_history[time_step] # 3. veri
temperatures = np.array([temp_1, temp_2, temp_3])

# Sidebar Metrics
st.sidebar.subheader("Displayed Values")
st.sidebar.metric("Sensor 1", f"{round(temp_1, 2)} °C")
st.sidebar.metric("Sensor 2", f"{round(temp_2, 2)} °C")
st.sidebar.metric("Sensor 3", f"{round(temp_3, 2)} °C")

# ==========================================
# MASK AND HEATMAP (Aynı kalacak, sadece veriyi kullanır)
# ==========================================
# (Buradaki IDW ve görselleştirme bloğunu değiştirmeden bırakabilirsiniz)
# Not: Interpolasyon fonksiyonunuz otomatik olarak 'points' 
# değişkenindeki 3 noktayı işleyecektir.
