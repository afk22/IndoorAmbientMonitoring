import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION AND SENSOR SETTINGS
# ==========================================

# 1. Sensor Positions
SENSOR_1_POS = [9.0, 38.0]    # Thing 1
SENSOR_2_POS = [13.0, 18.0]   # Thing 2
SENSOR_3_POS = [19.0, 50.0]   # Thing 3

points = np.array([SENSOR_1_POS, SENSOR_2_POS, SENSOR_3_POS])

# 2. Live API (TUM FROST Server) Base
FROST_BASE_URL = "https://gi3.gis.lrg.tum.de/frost/v1.1"

# 3. Metrics Configuration (Removed hardcoded vmin/vmax)
METRICS = {
    "Temperature": {
        "unit": "°C",
        "cmap": "coolwarm",
        "sensor_ids": [1817, 1855, 1713],
        "fallbacks": [23.5, 24.2, 23.8]
    },
    "Humidity": {
        "unit": "%",
        "cmap": "YlGnBu",
        "sensor_ids": [1818, 1856, 1725],
        "fallbacks": [45.0, 50.0, 48.0]
    },
    "CO2 Concentration": {
        "unit": "ppm",
        "cmap": "Reds",
        "sensor_ids": [1819, 1857, 1880], 
        "fallbacks": [450.0, 600.0, 550.0]
    },
    "Particulates": {
        "unit": "µg/m³",
        "cmap": "Purples",
        "sensor_ids": [1820, 1858, 1881], 
        "fallbacks": [12.0, 15.0, 10.0]
    }
}

def fetch_history_from_frost(url, iot_id, count=20):
    """Fetches the latest N observations for historical tracking."""
    try:
        api_url = f"{url}/Datastreams({iot_id})/Observations?$top={count}&$orderby=phenomenonTime%20desc"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("value"):
                return [(float(obs["result"]), obs["phenomenonTime"]) for obs in data["value"]]
    except Exception:
        pass
    return None

@st.cache_data(ttl=5)
def get_sensor_data_for_metric(metric_name):
    """Pulls historical streams for the selected metric."""
    config = METRICS[metric_name]
    ids = config["sensor_ids"]
    fb_vals = config["fallbacks"]
    
    default_time = datetime.now().isoformat()
    
    s1_data = fetch_history_from_frost(FROST_BASE_URL, ids[0]) or [(fb_vals[0], default_time)] * 20
    s2_data = fetch_history_from_frost(FROST_BASE_URL, ids[1]) or [(fb_vals[1], default_time)] * 20
    s3_data = fetch_history_from_frost(FROST_BASE_URL, ids[2]) or [(fb_vals[2], default_time)] * 20
    
    return s1_data, s2_data, s3_data

# ==========================================
# STREAMLIT INTERFACE AND SLIDER
# ==========================================

st.set_page_config(page_title="Library Environmental Heatmaps", layout="centered")
st.title("Library Spatial Environment Distribution")
st.write("Pulling real-time and historical data directly from the **TUM FROST Server**.")

# Sidebar: Metric Selection
st.sidebar.subheader("Measurement Type")
selected_metric = st.sidebar.selectbox(
    "Select Map Layer:",
    options=list(METRICS.keys())
)

# Fetch historical blocks for the selected metric
s1_history, s2_history, s3_history = get_sensor_data_for_metric(selected_metric)
metric_config = METRICS[selected_metric]

# Ensure we map available data indices safely across all sensors
max_steps = min(len(s1_history), len(s2_history), len(s3_history))

# Sidebar: Historical Timeline Slider
st.sidebar.subheader("Timeline Control")
time_step = st.sidebar.slider(
    label="Go back in time (Steps)",
    min_value=0,
    max_value=max_steps - 1 if max_steps > 0 else 0,
    value=0, 
    help="0 is the latest live data. Higher numbers step back into past records."
)

# Extract values based on slider position
val_1, time_1 = s1_history[time_step]
val_2, time_2 = s2_history[time_step]
val_3, time_3 = s3_history[time_step]
sensor_values = np.array([val_1, val_2, val_3])

# Parse timestamp and adjust for Munich Time (+2 Hours)
try:
    utc_time = datetime.strptime(time_1[:19], "%Y-%m-%dT%H:%M:%S")
    munich_time = utc_time + timedelta(hours=2) # Add 2 hours for local time
    clean_time = munich_time.strftime("%Y-%m-%d %H:%M:%S")
except:
    clean_time = time_1

# Display Mode Info
if time_step == 0:
    st.sidebar.success("🟢 Mode: LIVE DATA")
else:
    st.sidebar.warning(f"⏳ Mode: HISTORICAL (Step {time_step})")
    
st.sidebar.info(f"Data Timestamp (Munich):\n{clean_time}")

# Sidebar Metrics Display
unit = metric_config["unit"]
st.sidebar.subheader(f"Current {selected_metric} Values")
st.sidebar.metric(label="Sensor 1 (Thing 1)", value=f"{round(val_1, 2)} {unit}")
st.sidebar.metric(label="Sensor 2 (Thing 2)", value=f"{round(val_2, 2)} {unit}")
st.sidebar.metric(label="Sensor 3 (Thing 3)", value=f"{round(val_3, 2)} {unit}")

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

with st.spinner(f'Generating {selected_metric} heatmap layer...'):
    grid_z = idw_interpolation(points[:, 0], points[:, 1], sensor_values, grid_x, grid_y)

# ==========================================
# VISUALIZATION WITH DYNAMIC COLOR SCALING
# ==========================================
fig, ax = plt.subplots(figsize=(8, 10)) 
ax.imshow(img, extent=[0, X_MAX, 0, Y_MAX], origin='upper', alpha=0.8)

cmap_choice = metric_config["cmap"]

# DYNAMIC SCALE CALCULATION
# Find min and max of current sensor values
min_val = np.min(sensor_values)
max_val = np.max(sensor_values)

# Calculate a margin to ensure markers aren't exactly on the edge of the color map
if max_val == min_val:
    # Fallback if all sensors read the exact same value (prevents math errors)
    vmin_val = min_val - 1.0
    vmax_val = max_val + 1.0
else:
    # Add a 10% dynamic padding to top and bottom
    margin = (max_val - min_val) * 0.1
    vmin_val = min_val - margin
    vmax_val = max_val + margin

# Determine dynamic contour levels based on new dynamic limits
color_levels = np.linspace(vmin_val, vmax_val, 41)

c = ax.contourf(grid_x, grid_y, grid_z, levels=color_levels, cmap=cmap_choice, vmin=vmin_val, vmax=vmax_val, alpha=0.4)

# Create colorbar based on dynamic data range
ticks_count = 10
tick_levels = np.linspace(vmin_val, vmax_val, ticks_count)
cbar = fig.colorbar(c, ax=ax, shrink=0.7, ticks=tick_levels)
cbar.set_label(f'{selected_metric} ({unit})', weight='bold')

# Format ticks so they don't show unnecessarily long decimal numbers
cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in tick_levels])
cbar.ax.set_ylim(vmin_val, vmax_val)

ax.scatter(points[:, 0], points[:, 1], color='black', marker='x', s=100, linewidths=2, label='Active Sensors')
for i, txt in enumerate(np.round(sensor_values, 1)):
    ax.annotate(f"{txt} {unit}", (points[i, 0], points[i, 1]), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=9, color='black', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="black", alpha=0.8))

ax.set_title(f"Library {selected_metric} Heatmap - {clean_time}")
ax.set_xlabel("Width (meters)")
ax.set_ylabel("Length (meters)")
ax.legend(loc='lower left')
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.grid(True, linestyle='--', alpha=0.1)

st.pyplot(fig)
