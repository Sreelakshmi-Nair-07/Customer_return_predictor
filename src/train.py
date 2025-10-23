import pandas as pd
import yaml
import joblib
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from imblearn.ensemble import BalancedBaggingClassifier
    IMBLEARN_BB_AVAILABLE = True
except Exception:
    IMBLEARN_BB_AVAILABLE = False

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load training data
X_train = pd.read_csv(params["paths"]["processed_train"])
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

model_type = params["model"]["type"]

os.makedirs("models", exist_ok=True)

# Determine imbalance strategy
imbalance = params.get("imbalance", {}).get("strategy", "auto")
print("Imbalance strategy requested:", imbalance)

# compute class distribution
classes, counts = np.unique(y_train, return_counts=True)
class_ratio = {int(c): int(n) for c, n in zip(classes, counts)}
print("Class distribution (train):", class_ratio)

# Helper: compute class_weight if needed
def get_class_weight(y):
    classes = np.unique(y)
    cw = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, cw))

use_class_weight = None
use_scale_pos_weight = None

if imbalance == "auto":
    # if minority < 0.4 of data, consider imbalance
    minority_frac = counts.min() / counts.sum()
    if minority_frac < 0.4:
        imbalance = "class_weight"
    else:
        imbalance = "none"

if imbalance == "class_weight":
    use_class_weight = get_class_weight(y_train)
    print("Using class_weight:", use_class_weight)
elif imbalance == "scale_pos_weight":
    # useful for xgboost
    neg = counts[classes == 0][0] if 0 in classes else 1
    pos = counts[classes == 1][0] if 1 in classes else 1
    use_scale_pos_weight = neg / pos
    print("Using scale_pos_weight:", use_scale_pos_weight)
elif imbalance == "balanced_bagging":
    if not IMBLEARN_BB_AVAILABLE:
        print("imbalanced-learn BalancedBaggingClassifier not available; falling back to class_weight")
        use_class_weight = get_class_weight(y_train)
    else:
        print("Using BalancedBaggingClassifier from imblearn")

# Instantiate the chosen model, applying weights where supported
if model_type == "logistic_regression":
    if use_class_weight is not None:
        model = LogisticRegression(class_weight=use_class_weight, **params["model"]["logistic_regression"]["params"])
    else:
        model = LogisticRegression(**params["model"]["logistic_regression"]["params"])
elif model_type == "random_forest":
    if imbalance == "balanced_bagging" and IMBLEARN_BB_AVAILABLE:
        base_clf = RandomForestClassifier(**params["model"]["random_forest"]["params"])
        bb_params = params.get("imbalance", {}).get("balanced_bagging", {})
        model = BalancedBaggingClassifier(base_estimator=base_clf, n_estimators=bb_params.get("n_estimators", 10))
    else:
        if use_class_weight is not None:
            model = RandomForestClassifier(class_weight=use_class_weight, **params["model"]["random_forest"]["params"])
        else:
            model = RandomForestClassifier(**params["model"]["random_forest"]["params"])
elif model_type == "xgboost":
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed but params.yaml requests xgboost. Install xgboost or change params.")
    xgb_params = params["model"]["xgboost"]["params"].copy()
    if use_scale_pos_weight is not None:
        xgb_params["scale_pos_weight"] = use_scale_pos_weight
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)  # type: ignore
else:
    raise ValueError(f"Unsupported model type in params.yaml: {model_type}")

print(f"Training {model_type} on the training set...")
model.fit(X_train, y_train)

# Save model to configured path
os.makedirs(os.path.dirname(params["paths"]["model"]), exist_ok=True)
joblib.dump(model, params["paths"]["model"]) 

# Also save a small registry for test.py compatibility
registry = {model_type: params["paths"]["model"]}
with open("models/models_registry.json", "w") as f:
    json.dump(registry, f, indent=2)

print("Training finished. Model saved to:", params["paths"]["model"]) 
