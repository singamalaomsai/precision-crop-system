import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.schemas import SoilWeatherInput, PredictionResponse

app = FastAPI(title="Precision Crop Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_engine", "crop_pipeline.pkl")
FRONTEND_HTML_PATH = os.path.join(BASE_DIR, "frontend", "index.html")

pipeline = joblib.load(MODEL_PATH)
scaler = pipeline["scaler"]
model = pipeline["model"]
encoder = pipeline["encoder"]
explainer = pipeline["explainer"]
features = pipeline["features"]

@app.get("/")
def serve_website():
    return FileResponse(FRONTEND_HTML_PATH)

@app.get("/health")
def health_check():
    return {"status": "healthy", "classes_loaded": len(encoder.classes_)}

@app.post("/predict", response_model=PredictionResponse)
def predict_crop(data: SoilWeatherInput):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])[features]
        scaled_array = scaler.transform(input_df)

        pred_idx = int(model.predict(scaled_array)[0])
        probabilities = model.predict_proba(scaled_array)[0]
        confidence = round(float(probabilities[pred_idx]) * 100, 2)
        crop_name = str(encoder.inverse_transform([pred_idx])[0])

        # Compute SHAP values safely across all SHAP library versions
        raw_shap = explainer.shap_values(scaled_array)
        expected_val = explainer.expected_value

        if isinstance(raw_shap, list):
            # List of arrays: one per class
            class_shap_values = np.array(raw_shap[pred_idx])[0]
        else:
            arr = np.array(raw_shap)
            if arr.ndim == 3:
                # Shape: (1, 7, 22)
                class_shap_values = arr[0, :, pred_idx]
            else:
                class_shap_values = arr[0]

        if isinstance(expected_val, (list, np.ndarray)) and len(np.array(expected_val).flatten()) > 1:
            base_val = float(np.array(expected_val).flatten()[pred_idx])
        else:
            base_val = float(np.array(expected_val).flatten()[0])

        shap_dict = {
            feat: round(float(val), 4)
            for feat, val in zip(features, class_shap_values)
        }

        return PredictionResponse(
            recommended_crop=crop_name,
            confidence=confidence,
            base_value=round(base_val, 4),
            shap_contributions=shap_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")