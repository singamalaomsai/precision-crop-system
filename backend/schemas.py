from pydantic import BaseModel, Field
from typing import Dict

class SoilWeatherInput(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen content in soil (0 to 200)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus content in soil (0 to 200)")
    K: float = Field(..., ge=0, le=250, description="Potassium content in soil (0 to 250)")
    temperature: float = Field(..., ge=0.0, le=55.0, description="Temperature in Celsius")
    humidity: float = Field(..., ge=5.0, le=100.0, description="Relative Humidity in %")
    ph: float = Field(..., ge=3.0, le=10.0, description="Soil pH level (3.0 to 10.0)")
    rainfall: float = Field(..., ge=10.0, le=400.0, description="Rainfall in mm")

class PredictionResponse(BaseModel):
    recommended_crop: str
    confidence: float
    base_value: float
    shap_contributions: Dict[str, float]