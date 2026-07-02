import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
import time
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION AND SENSOR SETTINGS
# ==========================================

SENSOR_1_POS = [9.0, 38.0]    # Thing 1
SENSOR_2_POS = [13.0, 18.0]   # Thing 2
SENSOR_3_POS = [19.0, 50.0]   # Thing 3
points = np.array([SENSOR_1_POS, SENSOR_2_POS, SENSOR_3_POS])

FROST_BASE_URL = "https://gi3.gis.lrg.tum.de/frost/v1.1"
HISTORY_COUNT = 360 # 12 saat (2 dakikada bir = saatte 30 veri * 12 = 360 veri)

METRICS = {
    "Temperature": {"unit": "°C", "cmap": "coolwarm", "sensor_ids": [1817, 1855, 1713], "fallbacks": [23.5, 24.2, 23.8]},
    "Humidity": {"unit": "%", "cmap": "YlGnBu", "sensor_ids": [1818, 1856, 1725], "fallbacks": [45.0, 50.0, 48.0]},
    "CO2 Concentration": {"unit": "ppm", "cmap": "Reds", "sensor_ids": [1819, 1857, 1880], "fallbacks": [450.0, 600.0, 550.0]},
    "Particulates": {"unit": "µg/m³", "cmap": "Purples", "sensor_ids": [1820, 1858, 1881], "fallbacks": [12.0, 15.0, 10.0]}
}

def fetch_history_from_frost(url, iot_id, count=HISTORY_COUNT):
    try:
        api_url = f"{url}/Datastreams({iot_id})/Observations?$top={count}&$orderby=phenomenonTime%20desc"
        response = requests.get(api_url, timeout=10) # Timeout süresini 10 saniyeye çıkardık çünkü daha çok veri çekiyoruz
        if response.status_code == 200:
            data = response.json()
            if data.get("value"):
                return [(float(obs["result"]), obs["phenomenonTime"]) for obs in data["value"]]
    except Exception:
        pass
    return None

@st.cache_data(ttl=5)
def get_sensor_data_for_metric(metric_name):
    config = METRICS[metric_name]
    ids = config["sensor_ids"]
    fb_vals = config["fallbacks"]
    default_time = datetime.now().isoformat()
    
    s1_data = fetch_history_from_frost(FROST_BASE_URL, ids[0], HISTORY_COUNT) or [(fb_vals[0], default_time)] * HISTORY_COUNT
    s2_data = fetch_history_from_frost(FROST_BASE_URL, ids[1], HISTORY_COUNT) or [(fb_vals[1], default_time)] * HISTORY_COUNT
    s3_data = fetch_history_from_frost(FROST_BASE_URL, ids[2], HISTORY_COUNT) or [(fb_vals[2], default_time)] * HISTORY_COUNT
    return s1_data, s2_data, s3_data

# ==========================================
# PRE-LOAD IMAGE AND GRID
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "library_plan.png")
mask_path = os.path.join(current_dir, "library_mask.png")
if not os.path.exists(mask_path): mask_path = os.path.join(current_dir, "library_mask.jpg")

img_loaded, mask_loaded, aspect_ratio = False, False, 1.75
try:
    img = Image.open(image_path)
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    img_loaded = True
except FileNotFoundError:
    img = np.ones((100, 100, 3)) 

try:
    mask_img = Image.open(mask_path).convert('L')
    if img_loaded: mask_img = mask_img.resize((img_width, img_height))
    mask_np = np.array(mask_img)
    mask_loaded = True
except FileNotFoundError:
    pass

X_MAX = 30.0
Y_MAX = X_MAX * aspect_ratio  
resolution = 0.3 
grid_x, grid_y = np.mgrid[0:X_MAX:resolution, 0:Y_MAX:resolution]

# OPTIMIZED: Vektörel (Matris) IDW Interpolasyonu (Animasyon hızı için devasa performans artışı sağlar)
def idw_interpolation_fast(x, y, values, grid_x, grid_y, power=2):
    weights_sum = np.zeros(grid_x.shape)
    weighted_values_sum = np.zeros(grid_x.shape)
    
    for i in range(len(x)):
        dist = np.sqrt((grid_x - x[i])**2 + (grid_y - y[i])**2)
        dist = np.where(dist == 0, 1e-10, dist) # Sıfıra bölünme hatasını engeller
        weight = 1.0 / (dist ** power)
        
        weights_sum += weight
        weighted_values_sum += weight * values[i]
        
    grid_z = weighted_values_sum / weights_sum
    
    # Matris bazlı hızlı maskeleme işlemi
    if mask_loaded:
        m_height, m_width = mask_np.shape
        pixel_x = ((grid_x / X_MAX) * (m_width - 1)).astype(int)
        pixel_y = (((Y_MAX - grid_y) / Y_MAX) * (m_height - 1)).astype(int)
        
        pixel_x = np.clip(pixel_x, 0, m_width - 1)
        pixel_y = np.clip(pixel_y, 0, m_height - 1)
        
        invalid_mask = mask_np[pixel_y, pixel_x] < 128
        grid_z[invalid_mask] = np.nan
        
    return grid_z

def create_heatmap_plot(sensor_values, clean_time, selected_metric, metric_config):
    unit = metric_config["unit"]
    grid_z = idw_interpolation_fast(points[:, 0], points[:, 1], sensor_values, grid_x, grid_y)
    
    fig, ax = plt.subplots(figsize=(8, 10)) 
    ax.imshow(img, extent=[0, X_MAX, 0, Y_MAX], origin='upper', alpha=0.8)

    min_val, max_val = np.min(sensor_values), np.max(sensor_values)
    if max_val == min_val:
        vmin_val, vmax_val = min_val - 1.0, max_val + 1.0
    else:
        margin = (max_val - min_val) * 0.1
        vmin_val, vmax_val = min_val - margin, max_val + margin

    color_levels = np.linspace(vmin_val, vmax_val, 41)
    c = ax.contourf(grid_x, grid_y, grid_z, levels=color_levels, cmap=metric_config["cmap"], vmin=vmin_val, vmax=vmax_val, alpha=0.4)

    ticks_count = 10
    tick_levels = np.linspace(vmin_val, vmax_val, ticks_count)
    cbar = fig.colorbar(c, ax=ax, shrink=0.7, ticks=tick_levels)
    cbar.set_label(f'{selected_metric} ({unit})', weight='bold')
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
    
    return fig

# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.set_page_config(page_title="Library Environmental Dashboard", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Interactive Map", "Time-Lapse Animation"])

st.sidebar.subheader("Measurement Type")
selected_metric = st.sidebar.selectbox("Select Map Layer:", options=list(METRICS.keys()))

s1_history, s2_history, s3_history = get_sensor_data_for_metric(selected_metric)
metric_config = METRICS[selected_metric]
max_steps = min(len(s1_history), len(s2_history), len(s3_history))

def get_munich_time(time_str):
    try:
        utc_time = datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
        munich_time = utc_time + timedelta(hours=2)
        return munich_time.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return time_str

# ==========================================
# PAGE 1: INTERACTIVE MAP
# ==========================================
if page == "Interactive Map":
    st.title("Library Spatial Environment Distribution")
    
    st.sidebar.subheader("Timeline Control")
    time_step = st.sidebar.slider(
        label="Go back in time (Steps)",
        min_value=0,
        max_value=max_steps - 1 if max_steps > 0 else 0,
        value=0, 
        help=f"0 is the latest live data. Max goes back up to {HISTORY_COUNT} steps (approx. 12 hours)."
    )

    val_1, time_1 = s1_history[time_step]
    val_2, time_2 = s2_history[time_step]
    val_3, time_3 = s3_history[time_step]
    sensor_values = np.array([val_1, val_2, val_3])
    clean_time = get_munich_time(time_1)

    if time_step == 0: st.sidebar.success("🟢 Mode: LIVE DATA")
    else: st.sidebar.warning(f"⏳ Mode: HISTORICAL (Step {time_step})")
        
    st.sidebar.info(f"Data Timestamp (Munich):\n{clean_time}")
    unit = metric_config["unit"]
    st.sidebar.subheader("Current Values")
    st.sidebar.metric("Sensor 1 (Thing 1)", f"{round(val_1, 2)} {unit}")
    st.sidebar.metric("Sensor 2 (Thing 2)", f"{round(val_2, 2)} {unit}")
    st.sidebar.metric("Sensor 3 (Thing 3)", f"{round(val_3, 2)} {unit}")

    if st.button("🔄 Force Refresh API"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner(f'Generating {selected_metric} heatmap...'):
        fig = create_heatmap_plot(sensor_values, clean_time, selected_metric, metric_config)
        st.pyplot(fig)

# ==========================================
# PAGE 2: TIME-LAPSE ANIMATION
# ==========================================
elif page == "Time-Lapse Animation":
    st.title("12-Hour Historical Time-Lapse")
    st.write("Watch the environmental changes over the last ~12 hours of data.")
    
    st.sidebar.info("Animation Mode Active")
    
    # 0.0 saniye hızı eklendi. İşlemci gücüne göre sınırsız akış sağlar.
    anim_speed = st.slider("Animation Speed (seconds per frame)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, help="Set to 0.0 for maximum rendering speed.")
    
    start_animation = st.button("▶️ Play 12-Hour Animation (Oldest to Newest)")
    
    plot_placeholder = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    if not start_animation:
        val_1, time_1 = s1_history[0]
        val_2, time_2 = s2_history[0]
        val_3, time_3 = s3_history[0]
        clean_time = get_munich_time(time_1)
        fig = create_heatmap_plot(np.array([val_1, val_2, val_3]), clean_time, selected_metric, metric_config)
        plot_placeholder.pyplot(fig)
        status_text.info(f"Click 'Play Animation' to start the time-lapse of {max_steps} data points.")
        
    if start_animation:
        for step in range(max_steps - 1, -1, -1):
            val_1, time_1 = s1_history[step]
            val_2, time_2 = s2_history[step]
            val_3, time_3 = s3_history[step]
            clean_time = get_munich_time(time_1)
            
            sensor_values = np.array([val_1, val_2, val_3])
            
            fig = create_heatmap_plot(sensor_values, clean_time, selected_metric, metric_config)
            plot_placeholder.pyplot(fig)
            
            current_frame = max_steps - step
            status_text.warning(f"⏳ Animating... Frame {current_frame}/{max_steps} | Time: {clean_time}")
            
            # Progress bar güncellemesi
            progress_bar.progress(current_frame / max_steps)
            
            plt.close(fig)
            
            if anim_speed > 0:
                time.sleep(anim_speed)
            
        status_text.success("✅ Animation Complete! Displaying latest live data.")
