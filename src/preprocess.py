import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib

# Optionally import imblearn SMOTE if needed
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load raw CSV
df = pd.read_csv(params["dataset"]["raw_csv_path"])

# --- Data Quality Checks ---
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSummary statistics for numeric columns:\n", df.describe())

# Ensure column names mapping from dataset
# Expected columns in new dataset: InvoiceDate, UnitPrice, Quantity, CustomerID, Category, PaymentMethod, ShippingCost, Discount, ReturnStatus
target = params["dataset"]["target_col"]

# Normalize column names used in the script
if "InvoiceDate" in df.columns:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
else:
    # try other names
    df.columns = [c.strip() for c in df.columns]

# Compute TotalPurchase = UnitPrice * Quantity * (1 - Discount) + ShippingCost
df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice", df.get("Unit Price", pd.Series(0))), errors="coerce").fillna(0)
df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)
df["Discount"] = pd.to_numeric(df.get("Discount", 0), errors="coerce").fillna(0).clip(0, 1)
df["ShippingCost"] = pd.to_numeric(df.get("ShippingCost", 0), errors="coerce").fillna(0)

df["TotalPurchase"] = df["UnitPrice"] * df["Quantity"] * (1 - df["Discount"]) + df["ShippingCost"]

# Map target (ReturnStatus) to binary 1=Returned, 0=Not Returned
df[target] = df[target].astype(str).apply(lambda x: 1 if x.strip().lower().startswith("returned") else 0)
print("\nTarget distribution (binary):\n", df[target].value_counts(normalize=True))

# Analyze correlations of some original numeric features with target
orig_numeric = ["UnitPrice", "Quantity", "ShippingCost", "TotalPurchase"]
tmp = df[orig_numeric].copy()
tmp[target] = df[target]
numeric_corr = tmp.corr()[target].abs().sort_values(ascending=False)
print("\nNumeric feature correlations with target:\n", numeric_corr)

# For categorical features, use mean encoding difference
categorical_features = [c for c in ["Category", "PaymentMethod", "Country", "SalesChannel"] if c in df.columns]
print("\nCategorical feature mean-return differences:")
for cat in categorical_features:
    vals = df.groupby(cat)[target].mean()
    diff = (vals - df[target].mean()).abs().mean()
    print(f"{cat}: {diff:.4f}")

# --- Feature engineering (focused) ---
# Seasonal feature from InvoiceDate
if "InvoiceDate" in df.columns:
    df["Month"] = df["InvoiceDate"].dt.month
    df["Season"] = pd.cut(df["Month"], bins=[0,3,6,9,12], labels=["Winter","Spring","Summer","Fall"])

# Category return propensity
if "Category" in df.columns:
    cat_return = df.groupby("Category")[target].mean()
    df["Category_Return_Propensity"] = df["Category"].map(cat_return).fillna(0)

# Price per item is UnitPrice
df["Price_per_Item"] = df["UnitPrice"]

# Customer-level features
if "CustomerID" in df.columns:
    cust = df.groupby("CustomerID").agg({target: "mean", "TotalPurchase": ["mean","count"]})
    cust.columns = ["Customer_Return_Rate","Customer_Avg_Purchase","Customer_Purchase_Count"]
    df = df.merge(cust, left_on="CustomerID", right_index=True, how="left")
else:
    df["Customer_Return_Rate"] = 0
    df["Customer_Avg_Purchase"] = df["TotalPurchase"].mean()
    df["Customer_Purchase_Count"] = 1

# Risk score (simple): combine category propensity, customer return rate, and high price indicator
df["High_Price_Flag"] = (df["Price_per_Item"] > df["Price_per_Item"].median()).astype(int)
df["Risk_Score"] = (df.get("Category_Return_Propensity", 0) * 0.5 + df["Customer_Return_Rate"] * 0.4 + df["High_Price_Flag"] * 0.1)

# --- Additional features to boost signal (low-risk) ---
# Time features: hour of day and weekend flag
if "InvoiceDate" in df.columns:
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Is_Weekend"] = df["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)

# Recency: days since previous purchase for same customer
if "CustomerID" in df.columns and "InvoiceDate" in df.columns:
    df = df.sort_values(["CustomerID", "InvoiceDate"])
    df["Days_Since_Prev"] = df.groupby("CustomerID")["InvoiceDate"].diff().dt.days.fillna(-1)
    # fill -1 (first purchase) with a large value so it's distinguishable
    df["Days_Since_Prev"] = df["Days_Since_Prev"].replace(-1, df["Days_Since_Prev"].median())
else:
    df["Days_Since_Prev"] = df.get("Days_Since_Prev", 999)

# Smoothed category encoding (target encoding with smoothing)
global_mean = df[target].mean()
if "Category" in df.columns:
    cat_stats = df.groupby("Category")[target].agg(["mean", "count"]).rename(columns={"mean": "cat_mean", "count": "cat_count"})
    m = 10  # smoothing parameter
    cat_stats["cat_smooth"] = (cat_stats["cat_mean"] * cat_stats["cat_count"] + global_mean * m) / (cat_stats["cat_count"] + m)
    df["Category_Smooth_Return"] = df["Category"].map(cat_stats["cat_smooth"]).fillna(global_mean)
else:
    df["Category_Smooth_Return"] = global_mean

# Shipping to order value ratio
df["Shipping_to_Value"] = (df["ShippingCost"] / df["TotalPurchase"].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)

# Log transform of total purchase (stabilize skew)
df["Log_TotalPurchase"] = np.log1p(df["TotalPurchase"])

# Payment method popularity / counts
if "PaymentMethod" in df.columns:
    pay_counts = df["PaymentMethod"].value_counts()
    df["PaymentMethod_Count"] = df["PaymentMethod"].map(pay_counts).fillna(0)
else:
    df["PaymentMethod_Count"] = 0

# Price quantile bucket (4 bins)
df["Price_Bin"] = pd.qcut(df["Price_per_Item"].rank(method="first"), q=4, labels=False, duplicates='drop')

# Interaction features
df["Risk_x_LogPrice"] = df["Risk_Score"] * df["Log_TotalPurchase"]

# Final feature set (keep small and meaningful)
features = [
    col for col in [
        "Category", "Price_per_Item", "Quantity", "PaymentMethod", "Season",
        "Customer_Return_Rate", "Customer_Avg_Purchase", "Category_Return_Propensity", "Risk_Score",
        # new ones
        "Hour", "Is_Weekend", "Days_Since_Prev", "Category_Smooth_Return", "Shipping_to_Value",
        "Log_TotalPurchase", "PaymentMethod_Count", "Price_Bin", "Risk_x_LogPrice"
    ] if col in df.columns
]

# Prepare X,y and split
X = df[features]
y = df[target]

# Simple feature correlation analysis
correlations = []
for feature in features:
    if pd.api.types.is_numeric_dtype(df[feature]):
        corr = abs(df[feature].corr(df[target]))
    else:
        # For categorical features, use mean target encoding
        cat_corr = abs(df.groupby(feature)[target].mean() - df[target].mean()).mean()
        corr = cat_corr
    correlations.append((feature, corr))

# Sort features by correlation and keep top features
correlations.sort(key=lambda x: x[1], reverse=True)
top_features = [x[0] for x in correlations[:12]]  # Keep top 12 features
print("\nTop features by correlation with target:")
for feature, corr in correlations[:12]:
    print(f"{feature}: {corr:.4f}")

# Update features list with selected features
features = top_features

# Prepare final X and y
X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["split"]["test_size"],
    random_state=params["split"]["random_state"],
    stratify=y if params["split"]["stratify"] else None
)

# Preprocessing
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy=params["preprocess"]["numeric_imputer"])),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy=params["preprocess"]["categorical_imputer"])),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first' if params["preprocess"]["drop_first"] else None))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


# Fit the preprocessor on training data and transform both train and test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Optionally apply SMOTE for class imbalance
if params["preprocess"].get("use_smote", False):
    if SMOTE is None:
        raise ImportError("imblearn is required for SMOTE. Please install it with 'pip install imbalanced-learn'.")
    sm = SMOTE(random_state=params["split"]["random_state"])
    X_train_processed, y_train = sm.fit_resample(X_train_processed, y_train)

os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_train_processed).to_csv(params["paths"]["processed_train"], index=False)
pd.DataFrame(X_test_processed).to_csv(params["paths"]["processed_test"], index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

if params["preprocess"]["save_preprocessor"]:
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, params["paths"]["preprocessor"])
