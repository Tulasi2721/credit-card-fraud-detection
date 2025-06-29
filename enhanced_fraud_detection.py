#!/usr/bin/env python3
"""
Enhanced Credit Card Fraud Detection
Robust implementation with multiple advanced techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, precision_score, 
                           recall_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class EnhancedFraudDetector:
    """
    Enhanced fraud detection with robust preprocessing
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.results = {}
        
    def load_and_analyze_data(self):
        """Load and perform comprehensive data analysis"""
        print("🔍 Loading and analyzing dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print(f"Missing values: {missing_values.sum()}")
        
        # Class distribution analysis
        class_dist = self.df['Class'].value_counts()
        fraud_rate = (class_dist[1] / len(self.df)) * 100
        
        print(f"\n📊 Class Distribution:")
        print(f"Normal: {class_dist[0]:,} ({100-fraud_rate:.3f}%)")
        print(f"Fraud: {class_dist[1]:,} ({fraud_rate:.3f}%)")
        
        # Amount analysis
        fraud_data = self.df[self.df['Class'] == 1]
        normal_data = self.df[self.df['Class'] == 0]
        
        print(f"\n💰 Amount Analysis:")
        print(f"Normal - Mean: ${normal_data['Amount'].mean():.2f}, Std: ${normal_data['Amount'].std():.2f}")
        print(f"Fraud - Mean: ${fraud_data['Amount'].mean():.2f}, Std: ${fraud_data['Amount'].std():.2f}")
        
        return self.df
    
    def robust_feature_engineering(self):
        """Create robust features without introducing NaN values"""
        print("\n🔧 Robust Feature Engineering...")
        
        # Create a copy to avoid modifying original data
        df_enhanced = self.df.copy()
        
        # Time-based features (safe operations)
        df_enhanced['Hour'] = (df_enhanced['Time'] % 86400) // 3600
        df_enhanced['Day'] = df_enhanced['Time'] // 86400
        df_enhanced['Is_Weekend'] = ((df_enhanced['Time'] // 86400) % 7 >= 5).astype(int)
        df_enhanced['Is_Night'] = ((df_enhanced['Hour'] >= 22) | (df_enhanced['Hour'] <= 6)).astype(int)
        
        # Amount-based features (handle edge cases)
        df_enhanced['Amount_Log'] = np.log1p(df_enhanced['Amount'])  # log1p handles 0 values
        df_enhanced['Amount_Sqrt'] = np.sqrt(np.maximum(df_enhanced['Amount'], 0))  # Ensure non-negative
        df_enhanced['Is_Round_Amount'] = (df_enhanced['Amount'] % 1 == 0).astype(int)
        
        # Amount categories (safe binning)
        amount_bins = [0, 10, 50, 100, 500, float('inf')]
        df_enhanced['Amount_Category'] = pd.cut(df_enhanced['Amount'], bins=amount_bins, labels=False)
        
        # Safe statistical features
        amount_mean = df_enhanced['Amount'].mean()
        amount_std = df_enhanced['Amount'].std()
        if amount_std > 0:  # Avoid division by zero
            df_enhanced['Amount_Zscore'] = (df_enhanced['Amount'] - amount_mean) / amount_std
        else:
            df_enhanced['Amount_Zscore'] = 0
        
        # V-feature interactions (most important ones)
        if 'V1' in df_enhanced.columns and 'V2' in df_enhanced.columns:
            df_enhanced['V1_V2_Interaction'] = df_enhanced['V1'] * df_enhanced['V2']
        
        if 'V3' in df_enhanced.columns and 'V4' in df_enhanced.columns:
            df_enhanced['V3_V4_Interaction'] = df_enhanced['V3'] * df_enhanced['V4']
        
        # High-importance features sum (safe operation)
        high_importance_features = ['V17', 'V14', 'V12', 'V10', 'V16']
        available_features = [f for f in high_importance_features if f in df_enhanced.columns]
        
        if available_features:
            df_enhanced['High_Importance_Sum'] = df_enhanced[available_features].sum(axis=1)
            df_enhanced['High_Importance_Mean'] = df_enhanced[available_features].mean(axis=1)
        
        # Check for any NaN values introduced
        nan_count = df_enhanced.isnull().sum().sum()
        if nan_count > 0:
            print(f"⚠️ Warning: {nan_count} NaN values detected. Filling with 0...")
            df_enhanced = df_enhanced.fillna(0)
        
        print(f"✅ Created {len(df_enhanced.columns) - len(self.df.columns)} new features")
        print(f"Final feature count: {len(df_enhanced.columns)}")
        
        self.df_enhanced = df_enhanced
        return df_enhanced
    
    def prepare_data(self):
        """Prepare data for machine learning"""
        print("\n🔄 Preparing data for ML...")
        
        # Feature engineering
        self.robust_feature_engineering()
        
        # Separate features and target
        X = self.df_enhanced.drop('Class', axis=1)
        y = self.df_enhanced['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    
    def apply_balancing_techniques(self, X_train, y_train):
        """Apply different balancing techniques"""
        print("\n⚖️ Applying Balancing Techniques...")
        
        balanced_datasets = {}
        
        # 1. Original (no balancing)
        balanced_datasets['Original'] = (X_train, y_train)
        
        # 2. SMOTE
        try:
            smote = SMOTE(random_state=42)
            X_smote, y_smote = smote.fit_resample(X_train, y_train)
            balanced_datasets['SMOTE'] = (X_smote, y_smote)
            print(f"✅ SMOTE: {X_smote.shape}")
        except Exception as e:
            print(f"❌ SMOTE failed: {e}")
        
        # 3. Random Undersampling
        try:
            undersampler = RandomUnderSampler(random_state=42)
            X_under, y_under = undersampler.fit_resample(X_train, y_train)
            balanced_datasets['Undersampling'] = (X_under, y_under)
            print(f"✅ Undersampling: {X_under.shape}")
        except Exception as e:
            print(f"❌ Undersampling failed: {e}")
        
        return balanced_datasets
    
    def initialize_models(self):
        """Initialize ML models"""
        print("\n🤖 Initializing Models...")
        
        models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                random_state=42, n_estimators=100, class_weight='balanced'
            ),
            'SVM': SVC(
                random_state=42, probability=True, class_weight='balanced'
            ),
            'Isolation_Forest': IsolationForest(
                random_state=42, contamination=0.002, n_estimators=100
            )
        }
        
        # Try to add XGBoost if available
        try:
            from xgboost import XGBClassifier
            models['XGBoost'] = XGBClassifier(
                random_state=42, eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
            )
            print("✅ XGBoost added")
        except ImportError:
            print("⚠️ XGBoost not available")
        
        print(f"Initialized {len(models)} models")
        return models
    
    def optimize_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a single model"""
        
        # Handle Isolation Forest differently
        if 'Isolation' in model_name:
            model.fit(X_train)
            y_pred_anomaly = model.predict(X_test)
            y_pred = (y_pred_anomaly == -1).astype(int)
            y_pred_proba = model.decision_function(X_test)
            # Normalize to [0,1]
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            optimal_threshold = 0.5
        else:
            # Regular supervised learning
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            optimal_threshold = self.optimize_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n🎯 {model_name} Results:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if 'Isolation' not in model_name:
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        return {
            'model': model,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'optimal_threshold': optimal_threshold
        }
    
    def train_and_evaluate_all(self, X_train_scaled, X_test_scaled, y_train, y_test, balanced_datasets):
        """Train and evaluate all models"""
        print("\n" + "="*80)
        print("🚀 TRAINING AND EVALUATING ALL MODELS")
        print("="*80)
        
        models = self.initialize_models()
        all_results = {}
        
        for balance_method, (X_bal, y_bal) in balanced_datasets.items():
            print(f"\n📊 Testing with {balance_method} data...")
            method_results = {}
            
            # Scale balanced data if needed
            if balance_method != 'Original':
                scaler = RobustScaler()
                X_bal_scaled = scaler.fit_transform(X_bal)
            else:
                X_bal_scaled = X_train_scaled
            
            for model_name, model in models.items():
                try:
                    if 'Isolation' in model_name:
                        # Use original scale for Isolation Forest
                        result = self.evaluate_model(
                            model, X_bal, X_test_scaled, y_bal, y_test,
                            f"{model_name}_{balance_method}"
                        )
                    else:
                        result = self.evaluate_model(
                            model, X_bal_scaled, X_test_scaled, y_bal, y_test,
                            f"{model_name}_{balance_method}"
                        )
                    
                    method_results[model_name] = result
                    
                except Exception as e:
                    print(f"❌ Error with {model_name}: {e}")
                    continue
            
            all_results[balance_method] = method_results
        
        self.all_results = all_results
        return all_results
    
    def find_best_model(self):
        """Find the best performing model"""
        print("\n🏆 Finding Best Model...")
        
        best_score = 0
        best_model_info = None
        
        for balance_method, method_results in self.all_results.items():
            for model_name, result in method_results.items():
                score = result['f1']  # Use F1-score as primary metric
                
                if score > best_score:
                    best_score = score
                    best_model_info = {
                        'balance_method': balance_method,
                        'model_name': model_name,
                        'score': score,
                        'result': result
                    }
        
        if best_model_info:
            print(f"🥇 Best Model: {best_model_info['model_name']} with {best_model_info['balance_method']}")
            print(f"🎯 F1-Score: {best_model_info['score']:.4f}")
            print(f"📊 ROC-AUC: {best_model_info['result']['roc_auc']:.4f}")
            
            self.best_model = best_model_info
        
        return best_model_info
    
    def create_visualizations(self, X_test_scaled, y_test):
        """Create comprehensive visualizations"""
        print("\n📈 Creating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for balance_method, method_results in self.all_results.items():
            for model_name, result in method_results.items():
                if 'Isolation' not in model_name:
                    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                    ax1.plot(fpr, tpr, 
                           label=f"{model_name}_{balance_method} ({result['roc_auc']:.3f})",
                           alpha=0.7)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Model Performance Comparison
        ax2 = axes[0, 1]
        comparison_data = []
        for balance_method, method_results in self.all_results.items():
            for model_name, result in method_results.items():
                comparison_data.append({
                    'Model': f"{model_name}_{balance_method}",
                    'F1_Score': result['f1']
                })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('F1_Score', ascending=False)
        top_models = comparison_df.head(8)
        
        bars = ax2.barh(range(len(top_models)), top_models['F1_Score'])
        ax2.set_yticks(range(len(top_models)))
        ax2.set_yticklabels(top_models['Model'], fontsize=8)
        ax2.set_xlabel('F1-Score')
        ax2.set_title('Top Models by F1-Score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for Best Model
        ax3 = axes[1, 0]
        if hasattr(self, 'best_model'):
            best_result = self.best_model['result']
            cm = confusion_matrix(y_test, best_result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title(f'Best Model Confusion Matrix\n{self.best_model["model_name"]}')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        
        # 4. Feature Importance (if available)
        ax4 = axes[1, 1]
        if (hasattr(self, 'best_model') and 
            hasattr(self.best_model['result']['model'], 'feature_importances_')):
            
            model = self.best_model['result']['model']
            feature_names = self.df_enhanced.drop('Class', axis=1).columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            ax4.barh(range(len(importance_df)), importance_df['importance'])
            ax4.set_yticks(range(len(importance_df)))
            ax4.set_yticklabels(importance_df['feature'], fontsize=8)
            ax4.set_xlabel('Importance')
            ax4.set_title('Top 10 Feature Importance')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('enhanced_fraud_detection_results.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'enhanced_fraud_detection_results.png'")
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete enhanced fraud detection pipeline"""
        print("🚀 Starting Enhanced Credit Card Fraud Detection Pipeline")
        print("="*80)
        
        # Step 1: Load and analyze data
        self.load_and_analyze_data()
        
        # Step 2: Prepare data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = self.prepare_data()
        
        # Step 3: Apply balancing techniques
        balanced_datasets = self.apply_balancing_techniques(X_train, y_train)
        
        # Step 4: Train and evaluate all models
        self.train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test, balanced_datasets)
        
        # Step 5: Find best model
        self.find_best_model()
        
        # Step 6: Create visualizations
        self.create_visualizations(X_test_scaled, y_test)
        
        print("\n" + "="*80)
        print("🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'best_model': getattr(self, 'best_model', None),
            'all_results': self.all_results
        }


def main():
    """Main function"""
    data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
    
    detector = EnhancedFraudDetector(data_path)
    results = detector.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()
