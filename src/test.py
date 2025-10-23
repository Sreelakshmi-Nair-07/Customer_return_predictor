import pandas as pd
import yaml
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load test data
X_test = pd.read_csv(params["paths"]["processed_test"])
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

os.makedirs("experiments", exist_ok=True)

# Load registry of trained models
registry_path = "models/models_registry.json"
if not os.path.exists(registry_path):
    raise FileNotFoundError("models/models_registry.json not found. Run train stage first.")

with open(registry_path, "r") as f:
    models_saved = json.load(f)

results = {}
best_model = None
best_auc = -1

for name, path in models_saved.items():
    model = joblib.load(path)
    print(f"Loaded model '{name}' from {path} -> class: {type(model).__name__}")
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    if auc > best_auc:
        best_auc = auc
        best_model = (name, path)

# Save metrics
with open(params["paths"]["results"], "w") as f:
    json.dump(results, f, indent=4)

with open("metrics.json", "w") as f:
    json.dump(results.get(best_model[0], {}), f, indent=4)

# Copy best model to models/model.pkl for downstream steps
if best_model:
    best_model_obj = joblib.load(best_model[1])
    joblib.dump(best_model_obj, params["paths"]["model"]) 

print("✅ Evaluation complete. Metrics saved. Best model:", best_model[0] if best_model else None)
