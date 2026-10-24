import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# 1. Ensure dataset exists (auto-download if missing)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")

if not os.path.exists(DATA_PATH):
    print("Downloading Crop_recommendation.csv into dataset/ folder...")
    url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
    urllib.request.urlretrieve(url, DATA_PATH)
    print("Download complete!")

# 2. Load dataset and select features
df = pd.read_csv(DATA_PATH)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]
y = df["label"]

# 3. Encode string crop names into integers (0 to 21)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Stratified 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# 5. Fit StandardScaler strictly on X_train to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize & Train XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    random_state=42,
    eval_metric="mlogloss",
)

print("Training XGBoost model...")
xgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False,
)

# 7. Evaluate accuracy
y_pred = xgb_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Validation Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Build SHAP TreeExplainer
explainer = shap.TreeExplainer(xgb_model)

# 9. Save complete artifact inside ml_engine/crop_pipeline.pkl
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_pipeline.pkl")
artifact = {
    "scaler": scaler,
    "model": xgb_model,
    "encoder": label_encoder,
    "explainer": explainer,
    "features": FEATURES,
}

joblib.dump(artifact, MODEL_SAVE_PATH)
print(f"Saved model artifact successfully at: {MODEL_SAVE_PATH}")