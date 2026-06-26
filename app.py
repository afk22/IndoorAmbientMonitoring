import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests

# ==========================================
# CONFIGURATION AND LIVE SENSOR SETTINGS
# ==========================================

# 1. Sensor Positions (Coordinates in meters: [Width (X), Length (Y)])
SENSOR_1_POS = [9.0, 38.0]   # Thing 1 (ID: 1817) position
SENSOR_2_POS = [13.0, 18.0]  # Thing 2 (ID: 1855) position

points = np.array([SENSOR_1_POS, SENSOR_2_POS])

# 2. Live API (TUM FROST Server) Information
FROST_BASE_URL = "https://gi3.gis.lrg.tum.de/frost/v1.1"
SENSOR_1_IOT_ID = 1817  # Group 7: Thing 1 -> lpp_id: 1 (Temperature)
SENSOR_2_IOT_ID = 1855  # Group 7: Thing 2 -> lpp_id: 1 (Temperature)

def fetch_from_frost_server(url, iot_id):
    """Fetches the LATEST observation value for the given Datastream from FROST Server API."""
    try:
        # OGC SensorThings API standard query to get the most recent measurement
        api_url = f"{url}/Datastreams({iot_id})/Observations?$top=1&$orderby=phenomenonTime%20desc"
        response = requests.get(api_url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            if data.get("value") and len(data["value"]) > 0:
                return float(data["value"][0]["result"])
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=5)  # Checks for new data in the background every 5 seconds
def get_sensor_data_realtime():
    """Pulls real-time temperature data directly from the TUM FROST Server."""
    # Fallback default values in case of network issues or API downtime
    temperatures = [23.5, 24.2] 
    
    # Fetch data for Sensor 1
    val1 = fetch_from_frost_server(FROST_BASE_URL, SENSOR_1_IOT_ID)
    if val1 is not None:
        temperatures[0] = val1

    # Fetch data for Sensor 2
    val2 = fetch_from_frost_server(FROST_BASE_URL, SENSOR_2_IOT_ID)
    if val2 is not None:
        temperatures[1] = val2
                
    return np.array(temperatures)

# ==========================================
# STREAMLIT INTERFACE AND INTERPOLATION
# ==========================================

st.set_page_config(page_title="Library Sensor Heatmap", layout="centered")
st.title("Library Spatial Temperature Distribution")
st.write("This application pulls real-time data directly from the **TUM FROST Server** and generates an IDW spatial heatmap.")

# Fetch real-time data
temperatures = get_sensor_data_realtime()

# Sidebar for real-time metrics
st.sidebar.subheader("Live Sensor Values")
st.sidebar.metric(label="Sensor 1 (Thing 1)", value=f"{round(temperatures[0], 2)} °C")
st.sidebar.metric(label="Sensor 2 (Thing 2)", value=f"{round(temperatures[1], 2)} °C")

if st.button("🔄 Refresh Data"):
    st.rerun()

# 1. Floor Plan and Mask Image Setup
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
    st.error(f"'{image_path}' not found. Please ensure the image is in the same directory as 'app.py'.")
    img = np.ones((100, 100, 3)) 

try:
    mask_img = Image.open(mask_path).convert('L')
    if img_loaded:
        mask_img = mask_img.resize((img_width, img_height))
    mask_np = np.array(mask_img)
    mask_loaded = True
except FileNotFoundError:
    st.warning(f"'{mask_path}' not found! Showing flat heatmap without masking.")

X_MAX = 30.0
Y_MAX = X_MAX * aspect_ratio  

# 3. Inverse Distance Weighting (IDW) Function with Image Masking
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

# 4. Create Grid matching the layout size
resolution = 0.3 
grid_x, grid_y = np.mgrid[0:X_MAX:resolution, 0:Y_MAX:resolution]

with st.spinner('Calculating masked heatmap from live API data...'):
    grid_z = idw_interpolation(points[:, 0], points[:, 1], temperatures, grid_x, grid_y)

# 5. Visualization
fig, ax = plt.subplots(figsize=(8, 10)) 
ax.imshow(img, extent=[0, X_MAX, 0, Y_MAX], origin='upper', alpha=0.8)

# Fixed color scale boundaries (20°C to 40°C)
vmin_val = 20.0
vmax_val = 40.0

# Generate 41 levels for smooth 0.5°C color transitions between 20 and 40
color_levels = np.linspace(vmin_val, vmax_val, 41)

# Display the transparent heatmap contour
c = ax.contourf(grid_x, grid_y, grid_z, levels=color_levels, cmap='coolwarm', vmin=vmin_val, vmax=vmax_val, alpha=0.4)

# Colorbar configuration (Fixed scale from 20 to 40, ticks every 2 degrees)
cbar = fig.colorbar(c, ax=ax, shrink=0.7, ticks=np.arange(vmin_val, vmax_val + 1, 2))
cbar.set_label('Temperature (°C)', weight='bold')
cbar.ax.set_ylim(vmin_val, vmax_val)

# Scatter plot for active sensors
ax.scatter(points[:, 0], points[:, 1], color='black', marker='x', s=100, linewidths=2, label='Active Sensors')
for i, txt in enumerate(np.round(temperatures, 1)):
    ax.annotate(f"{txt}°C", (points[i, 0], points[i, 1]), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=10, color='black', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="black", alpha=0.8))

ax.set_title("Library - Masked Real-time Temperature Map")
ax.set_xlabel("Width (meters)")
ax.set_ylabel("Length (meters)")
ax.legend(loc='lower left')
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.grid(True, linestyle='--', alpha=0.1)

st.pyplot(fig)