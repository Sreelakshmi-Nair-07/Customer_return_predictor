# Customer Return Prediction System

A comprehensive machine learning pipeline for predicting customer return behavior with novel features and robust evaluation metrics.

## Features

- **Automatic Data Handling**: Handles missing values and data preprocessing
- **Novel Features**: 
  - Category Return Rate (learns which product categories have higher return rates)
  - Seasonal Effects (captures seasonal buying patterns)
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
- **Comprehensive Evaluation**: ROC-AUC, confusion matrix, feature importance
- **Risk Scoring**: Classifies customers as Low/Moderate/High risk
- **Visualizations**: ROC curves, confusion matrices, and risk distributions

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run with your own dataset
```python
from customer_return_predictor import CustomerReturnPredictor

# Initialize predictor
predictor = CustomerReturnPredictor()

# Run with your CSV file
results = predictor.run_complete_pipeline('your_dataset.csv')
```

### Option 2: Run with sample data (for testing)
```python
from customer_return_predictor import CustomerReturnPredictor

# Initialize predictor
predictor = CustomerReturnPredictor()

# Run with sample data
results = predictor.run_complete_pipeline()
```

### Option 3: Run directly
```bash
python customer_return_predictor.py
```

## Expected Dataset Format

Your CSV file should contain the following columns:
- `Customer_ID`: Unique customer identifier
- `Age`: Customer age
- `Gender`: Customer gender (Male/Female)
- `Purchase_Amount`: Amount of purchase
- `Product_Category`: Category of product (Electronics, Fashion, etc.)
- `Payment_Method`: Payment method used
- `Purchase_Date`: Date of purchase
- `Previous_Returns`: Number of previous returns
- `Customer_Satisfaction`: Satisfaction rating (1-5)
- `Shipping_Time`: Days taken for shipping
- `Returns`: Target variable (0 or 1)

## Output

The system generates:
1. **Model Evaluation**: Classification reports, confusion matrices, AUC scores
2. **Visualizations**: Saved as `customer_return_evaluation.png`
3. **Predictions**: Saved as `customer_return_predictions.csv`
4. **Risk Scores**: Low (0-0.3), Moderate (0.3-0.7), High (>0.7)

## Novel Features Explained

### 1. Category Return Rate
- Learns historical return rates by product category
- Captures product-type-level behavioral bias
- Example: Fashion items may have higher return rates than Electronics

### 2. Seasonal Effects
- Extracts month and season from purchase date
- Captures seasonal buying patterns
- Example: Holiday seasons often lead to more impulse buying and returns

## Model Performance

The system automatically:
- Trains multiple models and selects the best one
- Handles class imbalance with SMOTE (if available)
- Provides feature importance analysis
- Generates comprehensive evaluation metrics

## Requirements

- Python 3.7+
- See `requirements.txt` for package versions
- Optional: XGBoost and imbalanced-learn for enhanced performance
