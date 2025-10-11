"""
Example usage of the Customer Return Prediction System
"""

from customer_return_predictor import CustomerReturnPredictor
import pandas as pd

def example_with_sample_data():
    """
    Example using the built-in sample data generator
    """
    print("🎯 Example 1: Using Sample Data")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = CustomerReturnPredictor()
    
    # Run with sample data (no file needed)
    results = predictor.run_complete_pipeline()
    
    print(f"\nResults summary:")
    print(f"- Total customers: {len(results)}")
    print(f"- High risk customers: {(results['Return_Risk_Score'] == 'High').sum()}")
    print(f"- Model accuracy: {predictor.best_model['auc_score']:.3f}")
    
    return predictor, results

def example_with_custom_data():
    """
    Example showing how to use with your own CSV file
    """
    print("\n🎯 Example 2: Using Custom Data")
    print("=" * 50)
    
    # Create a small custom dataset for demonstration
    custom_data = pd.DataFrame({
        'Customer_ID': [1, 2, 3, 4, 5],
        'Age': [25, 35, 45, 28, 52],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Purchase_Amount': [150.0, 75.0, 300.0, 120.0, 200.0],
        'Product_Category': ['Electronics', 'Fashion', 'Home', 'Books', 'Sports'],
        'Payment_Method': ['Credit Card', 'Debit Card', 'Cash', 'UPI', 'Credit Card'],
        'Purchase_Date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-01-25', '2024-02-05'],
        'Previous_Returns': [0, 1, 2, 0, 1],
        'Customer_Satisfaction': [4, 2, 5, 3, 4],
        'Shipping_Time': [3, 7, 2, 5, 4],
        'Returns': [0, 1, 1, 0, 0]  # Target variable
    })
    
    # Save custom data
    custom_data.to_csv('custom_sample.csv', index=False)
    print("✅ Created custom_sample.csv")
    
    # Initialize predictor
    predictor = CustomerReturnPredictor()
    
    # Run with custom data
    try:
        results = predictor.run_complete_pipeline('custom_sample.csv')
        print(f"\nCustom data results:")
        print(f"- Total customers: {len(results)}")
        print(f"- Actual returns: {results['Returns'].sum()}")
        print(f"- Predicted returns: {results['Predicted_Return'].sum()}")
    except Exception as e:
        print(f"❌ Error with custom data: {e}")
        print("Note: Custom dataset might be too small for reliable predictions")

def analyze_results(results):
    """
    Analyze and display results in detail
    """
    print("\n📊 Detailed Analysis")
    print("=" * 50)
    
    # Risk distribution
    print("Risk Score Distribution:")
    risk_dist = results['Return_Risk_Score'].value_counts()
    for risk, count in risk_dist.items():
        percentage = (count / len(results)) * 100
        print(f"  {risk}: {count} customers ({percentage:.1f}%)")
    
    # High risk customers details
    high_risk = results[results['Return_Risk_Score'] == 'High']
    if len(high_risk) > 0:
        print(f"\n🔴 High Risk Customers ({len(high_risk)}):")
        print(high_risk[['Customer_ID', 'Age', 'Product_Category', 'Return_Probability']].to_string(index=False))
    
    # Model performance
    print(f"\n📈 Model Performance:")
    print(f"  Accuracy: {((results['Returns'] == results['Predicted_Return']).sum() / len(results)) * 100:.1f}%")
    
    # Feature insights
    print(f"\n🔍 Key Insights:")
    print(f"  - Average return probability: {results['Return_Probability'].mean():.3f}")
    print(f"  - Highest risk category: {results.groupby('Product_Category')['Return_Probability'].mean().idxmax()}")
    print(f"  - Age group with highest risk: {results.groupby('Age')['Return_Probability'].mean().idxmax()}")

if __name__ == "__main__":
    # Run examples
    predictor, results = example_with_sample_data()
    
    # Analyze results
    analyze_results(results)
    
    # Show custom data example
    example_with_custom_data()
    
    print("\n🎉 Examples completed!")
    print("Check the generated files:")
    print("- customer_return_predictions.csv")
    print("- customer_return_evaluation.png")
