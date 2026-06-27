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

# 1. Sensor Positions - FIXED CORRECTION
SENSOR_1_POS = [13.0, 18.0]   # Thing 1 (ID: 1817) corrected position
SENSOR_2_POS = [9.0, 38.0]    # Thing 2 (ID: 1855) corrected position

points = np.array([SENSOR_1_POS, SENSOR_2_POS])

# 2. Live API (TUM FROST Server) Information
FROST_BASE_URL = "https://gi3.gis.lrg.tum.de/frost/v1.1"
SENSOR_1_IOT_ID = 1817  # Group 7: Thing 1 -> Temperature
SENSOR_2_IOT_ID = 1855  # Group 7: Thing 2 -> Temperature

def fetch_history_from_frost(url, iot_id, count=20):
    """Fetches the latest N observations for historical tracking."""
    try:
        # Requesting last N entries ordered by time descending
        api_url = f"{url}/Datastreams({iot_id})/Observations?$top={count}&$orderby=phenomenonTime%20desc"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("value"):
                # Returns pairs of (value, timestamp)
                return [(float(obs["result"]), obs["phenomenonTime"]) for obs in data["value"]]
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=5)
def get_all_sensor_data():
    """Pulls historical streams for both sensors."""
    # Fallback structure if API is down
    default_time = datetime.now().isoformat()
    fallback_1 = [(23.5, default_time)] * 20
    fallback_2 = [(24.2, default_time)] * 20
    
    s1_data = fetch_history_from_frost(FROST_BASE_URL, SENSOR_1_IOT_ID, count=20) or fallback_1
    s2_data = fetch_history_from_frost(FROST_BASE_URL, SENSOR_2_IOT_ID, count=20) or fallback_2
    
    return s1_data, s2_data

# ==========================================
# STREAMLIT INTERFACE AND SLIDER
# ==========================================

st.set_page_config(page_title="Library Sensor Heatmap", layout="centered")
st.title("Library Spatial Temperature Distribution")
st.write("Pulling real-time and historical data directly from the **TUM FROST Server**.")

# Fetch historical blocks
s1_history, s2_history = get_all_sensor_data()

# Ensure we map available data indices safely
max_steps = min(len(s1_history), len(s2_history))

# Sidebar: Historical Timeline Slider
st.sidebar.subheader("Timeline Control")
time_step = st.sidebar.slider(
    label="Go back in time (Steps)",
    min_value=0,
    max_value=max_steps - 1,
    value=0, # Default 0 means the most recent data (LIVE)
    help="0 is the latest live data. Higher numbers step back into past records."
)

# Extract temperatures based on slider position
# Index 0 is newest, index max_steps-1 is oldest
temp_1, time_1 = s1_history[time_step]
temp_2, time_2 = s2_history[time_step]
temperatures = np.array([temp_1, temp_2])

# Parse timestamp clean display
try:
    clean_time = datetime.strptime(time_1[:19], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
except:
    clean_time = time_1

# Display Mode Info
if time_step == 0:
    st.sidebar.success(f"🟢 Mode: LIVE DATA")
else:
    st.sidebar.warning(f"⏳ Mode: HISTORICAL (Step {time_step})")
    
st.sidebar.info(f"Data Timestamp:\n{clean_time}")

# Sidebar Metrics
st.sidebar.subheader("Displayed Values")
st.sidebar.metric(label="Sensor 1 (Thing 1)", value=f"{round(temp_1, 2)} °C")
st.sidebar.metric(label="Sensor 2 (Thing 2)", value=f"{round(temp_2, 2)} °C")

if st.button("🔄 Force Refresh API"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# MASK AND HEATMAP INTERPOLATION
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "library_plan.png")
mask_path = os.path.join(current_dir, "library_mask.png")
if not os.path.exists(mask_path):
    mask_path = os.path.join(current_dir, "library_mask.jpg")

aspect_ratio = 1.75 
img_loaded = False
mask_loaded = False

try:
    img = Image.open(image_path)
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    img_loaded = True
except FileNotFoundError:
    st.error(f"'{image_path}' not found.")
    img = np.ones((100, 100, 3)) 

try:
    mask_img = Image.open(mask_path).convert('L')
    if img_loaded:
        mask_img = mask_img.resize((img_width, img_height))
    mask_np = np.array(mask_img)
    mask_loaded = True
except FileNotFoundError:
    st.warning(f"'{mask_path}' not found! Showing unmasked heatmap.")

X_MAX = 30.0
Y_MAX = X_MAX * aspect_ratio  

def idw_interpolation(x, y, values, grid_x, grid_y, power=2):
    grid_z = np.zeros(grid_x.shape)
    if mask_loaded:
        m_height, m_width = mask_np.shape

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            point_x = grid_x[i, j]
            point_y = grid_y[i, j]

            if mask_loaded:
                pixel_x = int((point_x / X_MAX) * (m_width - 1))
                pixel_y = int(((Y_MAX - point_y) / Y_MAX) * (m_height - 1))
                pixel_x = max(0, min(pixel_x, m_width - 1))
                pixel_y = max(0, min(pixel_y, m_height - 1))

                if mask_np[pixel_y, pixel_x] < 128:
                    grid_z[i, j] = np.nan
                    continue

            distances = np.sqrt((x - point_x)**2 + (y - point_y)**2)
            if 0 in distances:
                grid_z[i, j] = values[np.where(distances == 0)[0][0]]
            else:
                weights = 1.0 / (distances**power)
                grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_z

resolution = 0.3 
grid_x, grid_y = np.mgrid[0:X_MAX:resolution, 0:Y_MAX:resolution]

with st.spinner('Generating heatmap layer...'):
    grid_z = idw_interpolation(points[:, 0], points[:, 1], temperatures, grid_x, grid_y)

# ==========================================
# VISUALIZATION (Fixed scale 20°C - 40°C)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 10)) 
ax.imshow(img, extent=[0, X_MAX, 0, Y_MAX], origin='upper', alpha=0.8)

vmin_val = 20.0
vmax_val = 40.0
color_levels = np.linspace(vmin_val, vmax_val, 41)

c = ax.contourf(grid_x, grid_y, grid_z, levels=color_levels, cmap='coolwarm', vmin=vmin_val, vmax=vmax_val, alpha=0.4)

cbar = fig.colorbar(c, ax=ax, shrink=0.7, ticks=np.arange(vmin_val, vmax_val + 1, 2))
cbar.set_label('Temperature (°C)', weight='bold')
cbar.ax.set_ylim(vmin_val, vmax_val)

ax.scatter(points[:, 0], points[:, 1], color='black', marker='x', s=100, linewidths=2, label='Active Sensors')
for i, txt in enumerate(np.round(temperatures, 1)):
    ax.annotate(f"{txt}°C", (points[i, 0], points[i, 1]), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=10, color='black', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="black", alpha=0.8))

ax.set_title(f"Library Heatmap - {clean_time}")
ax.set_xlabel("Width (meters)")
ax.set_ylabel("Length (meters)")
ax.legend(loc='lower left')
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.grid(True, linestyle='--', alpha=0.1)

st.pyplot(fig)
