"""
Customer Return Prediction System
A comprehensive ML pipeline for predicting customer return behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, use if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Try to import SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available. Install with: pip install imbalanced-learn")

class CustomerReturnPredictor:
    """
    A comprehensive customer return prediction system
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        
    def load_data(self, file_path):
        """
        Load dataset from CSV file
        """
        print("📊 Loading dataset...")
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def create_sample_data(self):
        """
        Create sample dataset for demonstration
        """
        print("🎯 Creating sample dataset...")
        np.random.seed(42)
        
        n_samples = 1000
        
        # Generate sample data
        data = {
            'Customer_ID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 80, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Purchase_Amount': np.random.exponential(100, n_samples).round(2),
            'Product_Category': np.random.choice(['Electronics', 'Fashion', 'Home', 'Books', 'Sports'], n_samples),
            'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'UPI'], n_samples),
            'Purchase_Date': pd.date_range('2023-01-01', '2024-01-01', periods=n_samples),
            'Previous_Returns': np.random.poisson(0.5, n_samples),
            'Customer_Satisfaction': np.random.randint(1, 6, n_samples),
            'Shipping_Time': np.random.randint(1, 10, n_samples)
        }
        
        # Create realistic return patterns
        returns = []
        for i in range(n_samples):
            # Higher return probability for certain conditions
            prob = 0.1  # Base probability
            
            # Age factor (younger customers return more)
            if data['Age'][i] < 30:
                prob += 0.15
            elif data['Age'][i] > 50:
                prob -= 0.05
                
            # Category factor
            if data['Product_Category'][i] in ['Fashion', 'Electronics']:
                prob += 0.1
                
            # Amount factor (expensive items returned more)
            if data['Purchase_Amount'][i] > 200:
                prob += 0.1
                
            # Previous returns factor
            if data['Previous_Returns'][i] > 0:
                prob += 0.2
                
            # Satisfaction factor
            if data['Customer_Satisfaction'][i] <= 2:
                prob += 0.2
                
            # Shipping time factor
            if data['Shipping_Time'][i] > 7:
                prob += 0.1
                
            returns.append(1 if np.random.random() < prob else 0)
        
        data['Returns'] = returns
        self.df = pd.DataFrame(data)
        
        print(f"✅ Sample dataset created: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("🔧 Handling missing values...")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            
            # Handle missing values
            # For numerical columns, use median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = self.df[numerical_cols].fillna(self.df[numerical_cols].median())
            
            # For categorical columns, use mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col] = self.df[col].fillna(mode_value)
        else:
            print("✅ No missing values found")
    
    def create_novel_features(self):
        """
        Create novel features for better prediction
        """
        print("🚀 Creating novel features...")
        
        # 1. Category Return Rate Feature
        print("  📊 Creating Category Return Rate feature...")
        category_return_rate = self.df.groupby('Product_Category')['Returns'].mean()
        self.df['Category_Return_Rate'] = self.df['Product_Category'].map(category_return_rate)
        
        # 2. Seasonal Effects from Purchase Date
        print("  📅 Creating Seasonal Effects feature...")
        self.df['Purchase_Date'] = pd.to_datetime(self.df['Purchase_Date'])
        self.df['Month'] = self.df['Purchase_Date'].dt.month
        self.df['Season'] = self.df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # 3. Additional engineered features
        print("  ⚡ Creating additional features...")
        
        # Age groups
        self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                    bins=[0, 25, 35, 50, 100], 
                                    labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        # Purchase amount categories
        self.df['Amount_Category'] = pd.cut(self.df['Purchase_Amount'],
                                          bins=[0, 50, 150, 300, float('inf')],
                                          labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Risk score based on multiple factors
        self.df['Risk_Score'] = (
            (self.df['Previous_Returns'] * 0.3) +
            ((5 - self.df['Customer_Satisfaction']) * 0.2) +
            (self.df['Shipping_Time'] * 0.1) +
            (self.df['Category_Return_Rate'] * 0.4)
        )
        
        print("✅ Novel features created successfully")
    
    def preprocess_data(self):
        """
        Preprocess data for machine learning
        """
        print("🔄 Preprocessing data...")
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col not in ['Returns', 'Customer_ID', 'Purchase_Date']]
        self.X = self.df[feature_cols]
        self.y = self.df['Returns']
        
        # Identify numerical and categorical columns
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"  📊 Numerical features: {len(numerical_cols)}")
        print(f"  📝 Categorical features: {len(categorical_cols)}")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        from sklearn.preprocessing import OneHotEncoder
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Fit and transform the data
        self.X_processed = self.column_transformer.fit_transform(self.X)
        
        print("✅ Data preprocessing completed")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        print("✂️ Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_processed, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"✅ Data split: Train {self.X_train.shape}, Test {self.X_test.shape}")
    
    def handle_class_imbalance(self):
        """
        Handle class imbalance using SMOTE if available
        """
        if SMOTE_AVAILABLE:
            print("⚖️ Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"✅ SMOTE applied. New training set shape: {self.X_train.shape}")
        else:
            print("⚠️ SMOTE not available. Proceeding without class balancing.")
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("🤖 Training models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in models.items():
            print(f"  🔄 Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            model_scores[name] = auc_score
            
            # Store model and predictions
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score
            }
            
            print(f"    ✅ {name} - AUC: {auc_score:.4f}")
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (AUC: {model_scores[best_model_name]:.4f})")
    
    def evaluate_models(self):
        """
        Evaluate all trained models
        """
        print("📊 Evaluating models...")
        
        for name, model_data in self.models.items():
            print(f"\n🔍 {name} Evaluation:")
            print("=" * 50)
            
            y_pred = model_data['predictions']
            y_pred_proba = model_data['probabilities']
            
            # Classification Report
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion Matrix:\n{cm}")
            
            # AUC Score
            print(f"AUC Score: {model_data['auc_score']:.4f}")
    
    def create_return_risk_score(self):
        """
        Create Return Risk Score classification
        """
        print("🎯 Creating Return Risk Score...")
        
        # Use best model probabilities
        probabilities = self.best_model['probabilities']
        
        # Create risk categories
        risk_scores = []
        for prob in probabilities:
            if prob <= 0.3:
                risk_scores.append('Low')
            elif prob <= 0.7:
                risk_scores.append('Moderate')
            else:
                risk_scores.append('High')
        
        # Add to test data
        test_indices = self.y_test.index
        self.df.loc[test_indices, 'Return_Risk_Score'] = risk_scores
        self.df.loc[test_indices, 'Return_Probability'] = probabilities
        
        # Print risk distribution
        risk_distribution = pd.Series(risk_scores).value_counts()
        print("Risk Score Distribution:")
        print(risk_distribution)
    
    def visualize_results(self):
        """
        Create visualizations for model evaluation
        """
        print("📈 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Return Prediction - Model Evaluation', fontsize=16)
        
        # 1. ROC Curves for all models
        ax1 = axes[0, 0]
        for name, model_data in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, model_data['probabilities'])
            auc_score = model_data['auc_score']
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Confusion Matrix for best model
        ax2 = axes[0, 1]
        cm = confusion_matrix(self.y_test, self.best_model['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix (Best Model)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. Return Probability by Age Group
        ax3 = axes[1, 0]
        test_indices = self.y_test.index
        age_prob_data = self.df.loc[test_indices, ['Age_Group', 'Return_Probability']]
        age_prob_data.boxplot(column='Return_Probability', by='Age_Group', ax=ax3)
        ax3.set_title('Return Probability by Age Group')
        ax3.set_xlabel('Age Group')
        ax3.set_ylabel('Return Probability')
        
        # 4. Risk Score Distribution
        ax4 = axes[1, 1]
        risk_dist = self.df.loc[test_indices, 'Return_Risk_Score'].value_counts()
        colors = ['green', 'orange', 'red']
        ax4.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Risk Score Distribution')
        
        plt.tight_layout()
        plt.savefig('customer_return_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'customer_return_evaluation.png'")
    
    def get_feature_importance(self):
        """
        Get feature importance for tree-based models
        """
        print("🔍 Analyzing feature importance...")
        
        # Get feature names from the column transformer
        try:
            # Get numerical feature names
            numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Get categorical feature names (after one-hot encoding)
            categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Get one-hot encoded feature names
            feature_names = numerical_cols.copy()
            for col in categorical_cols:
                unique_values = self.X[col].unique()
                for val in unique_values:
                    feature_names.append(f"{col}_{val}")
            
            # Get importance for tree-based models
            for name, model_data in self.models.items():
                if hasattr(model_data['model'], 'feature_importances_'):
                    importance = model_data['model'].feature_importances_
                    
                    # Ensure we have the right number of features
                    if len(importance) == len(feature_names):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        print(f"\n{name} - Top 10 Most Important Features:")
                        print(importance_df.head(10))
                        
                        # Store for best model
                        if model_data == self.best_model:
                            self.feature_importance = importance_df
                    else:
                        print(f"⚠️ Feature count mismatch for {name}: {len(importance)} vs {len(feature_names)}")
                        
        except Exception as e:
            print(f"⚠️ Could not extract feature importance: {e}")
    
    def save_predictions(self, filename='customer_return_predictions.csv'):
        """
        Save predictions to CSV file
        """
        print(f"💾 Saving predictions to {filename}...")
        
        # Get test data with predictions
        test_indices = self.y_test.index
        results_df = self.df.loc[test_indices, [
            'Customer_ID', 'Age', 'Gender', 'Product_Category', 
            'Purchase_Amount', 'Payment_Method', 'Returns'
        ]].copy()
        
        # Add predictions
        results_df['Predicted_Return'] = self.best_model['predictions']
        results_df['Return_Probability'] = self.best_model['probabilities']
        results_df['Return_Risk_Score'] = self.df.loc[test_indices, 'Return_Risk_Score']
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        print(f"✅ Predictions saved to {filename}")
        
        return results_df
    
    def run_complete_pipeline(self, data_file=None):
        """
        Run the complete prediction pipeline
        """
        print("🚀 Starting Customer Return Prediction Pipeline")
        print("=" * 60)
        
        # Load or create data
        if data_file:
            self.load_data(data_file)
        else:
            self.create_sample_data()
        
        # Data preprocessing
        self.handle_missing_values()
        self.create_novel_features()
        self.preprocess_data()
        self.split_data()
        self.handle_class_imbalance()
        
        # Model training and evaluation
        self.train_models()
        self.evaluate_models()
        self.create_return_risk_score()
        self.get_feature_importance()
        
        # Visualization and saving
        self.visualize_results()
        predictions_df = self.save_predictions()
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        
        return predictions_df

def main():
    """
    Main function to run the customer return prediction system
    """
    # Initialize the predictor
    predictor = CustomerReturnPredictor()
    
    # Run the complete pipeline
    # You can provide a CSV file path here, or it will create sample data
    results = predictor.run_complete_pipeline()
    
    # Display summary
    print("\n📋 Summary:")
    print(f"Total customers analyzed: {len(results)}")
    print(f"Actual returns: {results['Returns'].sum()}")
    print(f"Predicted returns: {results['Predicted_Return'].sum()}")
    print(f"High risk customers: {(results['Return_Risk_Score'] == 'High').sum()}")
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
