# 🌱 AgriSense XAI: Precision Crop Recommendation & Explainable AI System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%20Attribution-ff0066)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20Deployed-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io/)

**AgriSense XAI** is a production-grade, decoupled Machine Learning and Explainable AI (XAI) web application designed for precision agriculture. It recommends the optimal crop across **22 agricultural classes** (~99% validation accuracy) based on real-time soil and climatic telemetry, while providing mathematically transparent **local SHAP (SHapley Additive exPlanations)** feature attributions for every prediction.

📄 **[Read the Full Technical & Engineering Guide (PDF)](./agrisense_xai_complete_project_guide.pdf)**

---

## 🚀 Key Engineering Highlights

* **Decoupled Microservice Architecture:** Separates the inference backend (**FastAPI + Pydantic**) from the presentation layer (**Streamlit Analytics Dashboard** & **Native HTML/CSS/JS Web Portal**).
* **Zero Training-Serving Skew:** Bundles the `StandardScaler`, `XGBClassifier`, `LabelEncoder`, and `shap.TreeExplainer` into a single unified artifact (`crop_pipeline.pkl`).
* **Real-Time Explainable AI (SHAP):** Computes exact local log-odds contributions for each soil and weather parameter, illustrating which features pushed the model *toward* (`+` Green) or *against* (`-` Red) the recommended crop.
* **Smart Hybrid Cloud Routing:** Automatically connects to the FastAPI `/predict` microservice when running locally on port `8000`, and seamlessly falls back to cached direct pipeline inference (`@st.cache_resource`) when deployed on **Streamlit Community Cloud**.

---

## 📂 Project Architecture

~~~text
precision_crop_system/
├── backend/
│   ├── main.py                        # FastAPI server, /health & /predict endpoints, 3D SHAP slicing
│   └── schemas.py                     # Pydantic data validation models for soil/climate telemetry
├── dataset/
│   └── Crop_recommendation.csv        # 2,200 agronomic samples across 22 crop classes
├── frontend/
│   ├── app.py                         # Hybrid Streamlit dashboard (Manual inputs + Soil presets + SHAP)
│   ├── index.html                     # Standalone dark-mode HTML5/JS/CSS web interface
│   └── fix.py                         # Clean file generator utility for index.html
├── ml_engine/
│   ├── train_pipeline.py              # Model training, evaluation, and SHAP explainer serialization
│   └── crop_pipeline.pkl              # Bundled production artifact (Scaler + XGBoost + Encoder + SHAP)
├── agrisense_xai_complete_project_guide.pdf  # Complete 6-page step-by-step engineering documentation
├── requirements.txt                   # Root Python dependencies for local & Streamlit Cloud deployment
└── README.md                          # Project documentation
~~~

---

## 🧪 Agronomic Input Telemetry

The system validates and analyzes 7 core soil and environmental parameters:

| Parameter | Description | Typical Range | Unit |
| :--- | :--- | :--- | :--- |
| **Nitrogen (`N`)** | Soil nitrogen content | `0 – 140` | ppm (mg/kg) |
| **Phosphorus (`P`)** | Soil phosphorus content | `5 – 145` | ppm (mg/kg) |
| **Potassium (`K`)** | Soil potassium content | `5 – 205` | ppm (mg/kg) |
| **Temperature (`temperature`)** | Ambient air temperature | `8.0 – 45.0` | °C |
| **Relative Humidity (`humidity`)** | Atmospheric moisture level | `14.0 – 100.0` | % |
| **Soil pH (`ph`)** | Acidity or alkalinity of soil | `3.5 – 9.5` | pH scale |
| **Seasonal Rainfall (`rainfall`)** | Cumulative seasonal precipitation | `20.0 – 300.0` | mm |

---

## ⚙️ Local Installation & Usage

### 1. Clone the Repository & Install Dependencies
~~~bash
git clone https://github.com/singamalaomsai/precision-crop-system.git
cd precision-crop-system
pip install -r requirements.txt
~~~

### 2. (Optional) Retrain the ML & SHAP Pipeline
~~~bash
python ml_engine/train_pipeline.py
~~~

### 3. Launch the FastAPI Backend & Native Web UI
~~~bash
python -m uvicorn backend.main:app --reload --port 8000
~~~
* **Native Web App:** Open `http://127.0.0.1:8000`
* **Interactive Swagger API Docs:** Open `http://127.0.0.1:8000/docs`

### 4. Launch the Streamlit Analytics Dashboard
In a second terminal tab, run:
~~~bash
python -m streamlit run frontend/app.py
~~~
* **Streamlit UI:** Open `http://localhost:8501`

---

## 📊 Sample API Request & Response (`POST /predict`)

**Request Payload:**
~~~json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
~~~

**Response Payload:**
~~~json
{
  "recommended_crop": "rice",
  "confidence": 98.95,
  "base_value": 0.2267,
  "shap_contributions": {
    "rainfall": 3.412,
    "humidity": 1.845,
    "N": 0.612,
    "K": 0.194,
    "P": 0.088,
    "temperature": -0.142,
    "ph": 0.053
  }
}
~~~

---

## 👨‍💻 Author

**Singamala Om Sai**  
* GitHub: [@singamalaomsai](https://github.com/singamalaomsai)
