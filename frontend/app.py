import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Precision Crop Advisor (XAI)",
    page_icon="🌱",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_engine", "crop_pipeline.pkl")

@st.cache_resource
def load_local_pipeline():
    return joblib.load(MODEL_PATH)

# Check if FastAPI is live on Port 8000; otherwise use direct ML Engine (for Streamlit Cloud)
api_online = False
try:
    health = requests.get(f"{API_URL}/health", timeout=1.5).json()
    api_online = True
except Exception:
    api_online = False

st.title("🌱 Precision Agriculture & Crop Recommendation System")
st.markdown(
    "*Architecture:* Decoupled *FastAPI Backend* + *XGBoost Classifier* + *Real-Time SHAP Explainable AI*"
)

# Sidebar status badge
st.sidebar.header("🔌 System Engine Status")
if api_online:
    st.sidebar.success("FastAPI Backend: Online (Port 8000)")
else:
    st.sidebar.info("Cloud Standalone Mode: ML Pipeline Loaded Directly")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Soil Presets")
preset = st.sidebar.selectbox(
    "Load Sample Soil Profile:",
    ["Custom Manual Entry", "Rice (High Rain & Humidity)", "Chickpea (Dry / Low Rain)", "Cotton (High N, Warm)", "Coffee (Acidic Soil, High Rain)"]
)

defaults = {"N": 90.0, "P": 42.0, "K": 43.0, "temp": 20.8, "hum": 82.0, "ph": 6.5, "rain": 202.9}
if preset == "Rice (High Rain & Humidity)":
    defaults = {"N": 90.0, "P": 42.0, "K": 43.0, "temp": 20.8, "hum": 82.0, "ph": 6.5, "rain": 202.9}
elif preset == "Chickpea (Dry / Low Rain)":
    defaults = {"N": 40.0, "P": 67.0, "K": 79.0, "temp": 18.8, "hum": 16.8, "ph": 7.3, "rain": 80.0}
elif preset == "Cotton (High N, Warm)":
    defaults = {"N": 118.0, "P": 46.0, "K": 19.0, "temp": 24.0, "hum": 79.8, "ph": 6.8, "rain": 80.3}
elif preset == "Coffee (Acidic Soil, High Rain)":
    defaults = {"N": 101.0, "P": 28.0, "K": 30.0, "temp": 25.5, "hum": 58.8, "ph": 6.7, "rain": 158.0}

# Main Page Manual Number Input Form
st.subheader("🧪 Enter Soil & Climatic Parameters Manually")
c1, c2, c3, c4 = st.columns(4)

with c1:
    N = st.number_input("Nitrogen (N) [0 - 200 ppm]", min_value=0.0, max_value=200.0, value=defaults["N"], step=1.0)
    P = st.number_input("Phosphorus (P) [0 - 200 ppm]", min_value=0.0, max_value=200.0, value=defaults["P"], step=1.0)

with c2:
    K = st.number_input("Potassium (K) [0 - 250 ppm]", min_value=0.0, max_value=250.0, value=defaults["K"], step=1.0)
    temperature = st.number_input("Temperature (°C) [0 - 55]", min_value=0.0, max_value=55.0, value=defaults["temp"], step=0.5)

with c3:
    humidity = st.number_input("Relative Humidity (%) [5 - 100]", min_value=5.0, max_value=100.0, value=defaults["hum"], step=1.0)
    ph = st.number_input("Soil pH Level [3.0 - 10.0]", min_value=3.0, max_value=10.0, value=defaults["ph"], step=0.1)

with c4:
    rainfall = st.number_input("Rainfall (mm) [10 - 400]", min_value=10.0, max_value=400.0, value=defaults["rain"], step=5.0)
    st.write("")
    run_btn = st.button("🚀 Analyze Soil & Recommend Crop", type="primary", use_container_width=True)

payload = {
    "N": N,
    "P": P,
    "K": K,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall,
}

def compute_prediction(data_dict):
    if api_online:
        resp = requests.post(f"{API_URL}/predict", json=data_dict, timeout=5)
        return resp.json()
    else:
        pipe = load_local_pipeline()
        scaler = pipe["scaler"]
        model = pipe["model"]
        encoder = pipe["encoder"]
        explainer = pipe["explainer"]
        features = pipe["features"]

        df_in = pd.DataFrame([data_dict])[features]
        scaled = scaler.transform(df_in)
        pred_idx = int(model.predict(scaled)[0])
        probs = model.predict_proba(scaled)[0]
        conf = round(float(probs[pred_idx]) * 100, 2)
        crop = str(encoder.inverse_transform([pred_idx])[0])

        raw_shap = explainer.shap_values(scaled)
        exp_val = explainer.expected_value
        if isinstance(raw_shap, list):
            c_shap = np.array(raw_shap[pred_idx])[0]
        else:
            arr = np.array(raw_shap)
            c_shap = arr[0, :, pred_idx] if arr.ndim == 3 else arr[0]

        if isinstance(exp_val, (list, np.ndarray)) and len(np.array(exp_val).flatten()) > 1:
            base_v = float(np.array(exp_val).flatten()[pred_idx])
        else:
            base_v = float(np.array(exp_val).flatten()[0])

        shap_d = {f: round(float(v), 4) for f, v in zip(features, c_shap)}
        return {
            "recommended_crop": crop,
            "confidence": conf,
            "base_value": round(base_v, 4),
            "shap_contributions": shap_d,
        }

st.markdown("---")

# Run automatically on page load AND whenever user edits manual inputs or clicks the button
result = compute_prediction(payload)
crop = result["recommended_crop"]
conf = result["confidence"]
base_val = result["base_value"]
shap_dict = result["shap_contributions"]

col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.success(f"### 🏆 Recommended Crop: *{crop.upper()}*")
    st.metric(label="XGBoost Confidence Score", value=f"{conf}%")
    st.caption(f"SHAP Base Log-Odds: {base_val}")
    st.markdown("#### 📋 Current Field Telemetry")
    summary_df = pd.DataFrame(list(payload.items()), columns=["Parameter", "Manual Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("🔍 Explainable AI: Local SHAP Feature Attribution")
    st.caption("Green bars (+) pushed the model TOWARD this crop. Red bars (-) pushed AWAY from this crop.")

    shap_df = pd.DataFrame(list(shap_dict.items()), columns=["Feature", "SHAP Value"]).sort_values(by="SHAP Value", ascending=True)
    colors = ["#f43f5e" if v < 0 else "#10b981" for v in shap_df["SHAP Value"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("SHAP Contribution (Log-Odds Impact)")
    ax.set_title(f"Feature Attribution for {crop.upper()}")

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (0.04 if w >= 0 else -0.35),
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.2f}",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)