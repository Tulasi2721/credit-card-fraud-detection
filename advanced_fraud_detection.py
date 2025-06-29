#!/usr/bin/env python3
"""
Advanced Credit Card Fraud Detection
Implements multiple advanced techniques for fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, precision_score, 
                           recall_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class AdvancedFraudDetector:
    """
    Advanced fraud detection system with multiple techniques
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.optimal_threshold = 0.5
        
    def load_and_explore_data(self):
        """Enhanced data loading with detailed exploration"""
        print("🔍 Loading and exploring dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Class distribution
        class_dist = self.df['Class'].value_counts()
        fraud_rate = (class_dist[1] / len(self.df)) * 100
        
        print(f"\n📊 Class Distribution:")
        print(f"Normal: {class_dist[0]:,} ({100-fraud_rate:.3f}%)")
        print(f"Fraud: {class_dist[1]:,} ({fraud_rate:.3f}%)")
        
        # Statistical analysis
        print(f"\n📈 Statistical Analysis:")
        fraud_data = self.df[self.df['Class'] == 1]
        normal_data = self.df[self.df['Class'] == 0]
        
        print(f"Average fraud amount: ${fraud_data['Amount'].mean():.2f}")
        print(f"Average normal amount: ${normal_data['Amount'].mean():.2f}")
        print(f"Fraud amount std: ${fraud_data['Amount'].std():.2f}")
        print(f"Normal amount std: ${normal_data['Amount'].std():.2f}")
        
        return self.df
    
    def advanced_feature_engineering(self):
        """Create advanced features for better fraud detection"""
        print("\n🔧 Advanced Feature Engineering...")
        
        # Time-based features
        self.df['Hour'] = (self.df['Time'] % 86400) // 3600
        self.df['Day'] = self.df['Time'] // 86400
        self.df['Is_Weekend'] = ((self.df['Time'] // 86400) % 7 >= 5).astype(int)
        self.df['Is_Night'] = ((self.df['Hour'] >= 22) | (self.df['Hour'] <= 6)).astype(int)
        
        # Amount-based features
        self.df['Amount_Log'] = np.log1p(self.df['Amount'])
        self.df['Amount_Sqrt'] = np.sqrt(self.df['Amount'])
        self.df['Is_Round_Amount'] = (self.df['Amount'] % 1 == 0).astype(int)
        
        # Amount categories
        amount_bins = [0, 10, 50, 100, 500, float('inf')]
        self.df['Amount_Category'] = pd.cut(self.df['Amount'], bins=amount_bins, labels=False)
        
        # Statistical features
        self.df['Amount_Zscore'] = (self.df['Amount'] - self.df['Amount'].mean()) / self.df['Amount'].std()
        
        # V-feature combinations (most important ones)
        self.df['V1_V2_Interaction'] = self.df['V1'] * self.df['V2']
        self.df['V3_V4_Interaction'] = self.df['V3'] * self.df['V4']
        
        # High-importance V features sum
        high_importance_features = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 'V11']
        available_features = [f for f in high_importance_features if f in self.df.columns]
        if available_features:
            self.df['High_Importance_Sum'] = self.df[available_features].sum(axis=1)
            self.df['High_Importance_Mean'] = self.df[available_features].mean(axis=1)
        
        print(f"✅ Created {len(self.df.columns) - 31} new features")
        return self.df
    
    def preprocess_data(self):
        """Enhanced preprocessing with multiple scaling options"""
        print("\n🔄 Advanced Data Preprocessing...")
        
        # Feature engineering
        self.advanced_feature_engineering()
        
        # Prepare features
        feature_columns = [col for col in self.df.columns if col != 'Class']
        X = self.df[feature_columns]
        y = self.df['Class']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Multiple scaling approaches
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        self.scaled_data = {}
        for name, scaler in self.scalers.items():
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            
            self.scaled_data[name] = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'scaler': scaler
            }
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Features: {len(feature_columns)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_class_imbalance(self):
        """Multiple techniques for handling class imbalance"""
        print("\n⚖️ Handling Class Imbalance...")
        
        self.balanced_datasets = {}
        
        # 1. SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(self.X_train, self.y_train)
        self.balanced_datasets['SMOTE'] = (X_smote, y_smote)
        
        # 2. ADASYN
        try:
            adasyn = ADASYN(random_state=42)
            X_adasyn, y_adasyn = adasyn.fit_resample(self.X_train, self.y_train)
            self.balanced_datasets['ADASYN'] = (X_adasyn, y_adasyn)
        except:
            print("⚠️ ADASYN failed, skipping...")
        
        # 3. SMOTETomek
        smotetomek = SMOTETomek(random_state=42)
        X_smotetomek, y_smotetomek = smotetomek.fit_resample(self.X_train, self.y_train)
        self.balanced_datasets['SMOTETomek'] = (X_smotetomek, y_smotetomek)
        
        # 4. Random Undersampling
        undersampler = RandomUnderSampler(random_state=42)
        X_under, y_under = undersampler.fit_resample(self.X_train, self.y_train)
        self.balanced_datasets['Undersampling'] = (X_under, y_under)
        
        # 5. Class weights (for algorithms that support it)
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=self.y_train)
        self.class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        for method, (X, y) in self.balanced_datasets.items():
            print(f"{method}: {X.shape}, Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return self.balanced_datasets
    
    def initialize_advanced_models(self):
        """Initialize multiple advanced models"""
        print("\n🤖 Initializing Advanced Models...")
        
        self.models = {
            # Traditional ML
            'Logistic_Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                random_state=42, n_estimators=100, class_weight='balanced'
            ),
            'SVM': SVC(
                random_state=42, probability=True, class_weight='balanced'
            ),
            
            # Advanced ensemble
            'Voting_Classifier': VotingClassifier([
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
                ('svm', SVC(random_state=42, probability=True))
            ], voting='soft'),
            
            # Anomaly detection
            'Isolation_Forest': IsolationForest(
                random_state=42, contamination=0.002, n_estimators=100
            )
        }
        
        # Try to import and add XGBoost if available
        try:
            from xgboost import XGBClassifier
            self.models['XGBoost'] = XGBClassifier(
                random_state=42, eval_metric='logloss', 
                scale_pos_weight=len(self.y_train[self.y_train==0])/len(self.y_train[self.y_train==1])
            )
            print("✅ XGBoost added")
        except ImportError:
            print("⚠️ XGBoost not available")
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def optimize_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold using precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        return optimal_threshold, f1_scores[optimal_idx]
    
    def evaluate_model_advanced(self, model, X_train, X_test, y_train, y_test, model_name):
        """Advanced model evaluation with threshold optimization"""
        
        # Handle Isolation Forest differently (anomaly detection)
        if 'Isolation' in model_name:
            model.fit(X_train)
            y_pred_anomaly = model.predict(X_test)
            y_pred = (y_pred_anomaly == -1).astype(int)  # -1 means anomaly (fraud)
            y_pred_proba = model.decision_function(X_test)
            # Normalize decision function to [0,1]
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        else:
            # Regular supervised learning
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Optimize threshold
            optimal_threshold, best_f1 = self.optimize_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n🎯 {model_name} Results:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        if 'Isolation' not in model_name:
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        return {
            'model': model,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': avg_precision,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'optimal_threshold': optimal_threshold if 'Isolation' not in model_name else 0.5
        }
    
    def train_and_evaluate_all_advanced(self):
        """Train and evaluate all models with different balancing techniques"""
        print("\n" + "="*80)
        print("🚀 ADVANCED MODEL TRAINING AND EVALUATION")
        print("="*80)
        
        all_results = {}
        
        # Test different balancing techniques
        balancing_methods = ['Original'] + list(self.balanced_datasets.keys())
        
        for balance_method in balancing_methods:
            print(f"\n📊 Testing with {balance_method} data...")
            
            if balance_method == 'Original':
                X_train_bal, y_train_bal = self.X_train, self.y_train
            else:
                X_train_bal, y_train_bal = self.balanced_datasets[balance_method]
            
            method_results = {}
            
            for model_name, model in self.models.items():
                try:
                    # Use robust scaling for most models
                    if 'Isolation' in model_name:
                        # Isolation Forest works better with original scale
                        X_train_use = X_train_bal
                        X_test_use = self.X_test
                    else:
                        # Use scaled data for other models
                        scaler = RobustScaler()
                        X_train_use = scaler.fit_transform(X_train_bal)
                        X_test_use = scaler.transform(self.X_test)
                    
                    result = self.evaluate_model_advanced(
                        model, X_train_use, X_test_use, y_train_bal, self.y_test,
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
        """Find the best performing model across all techniques"""
        print("\n🏆 Finding Best Model...")

        best_score = 0
        best_model_info = None

        for balance_method, method_results in self.all_results.items():
            for model_name, result in method_results.items():
                # Use F1-score as primary metric (good for imbalanced data)
                score = result['f1']

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
            print(f"🎯 Precision: {best_model_info['result']['precision']:.4f}")
            print(f"🎯 Recall: {best_model_info['result']['recall']:.4f}")

            self.best_model = best_model_info

        return best_model_info

    def create_comprehensive_comparison(self):
        """Create comprehensive comparison of all models"""
        print("\n📊 Creating Comprehensive Model Comparison...")

        comparison_data = []

        for balance_method, method_results in self.all_results.items():
            for model_name, result in method_results.items():
                comparison_data.append({
                    'Balance_Method': balance_method,
                    'Model': model_name,
                    'ROC_AUC': result['roc_auc'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1'],
                    'Avg_Precision': result['avg_precision']
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1-score
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)

        print("\n🏆 Top 10 Model Combinations:")
        print(comparison_df.head(10).round(4))

        return comparison_df

    def plot_advanced_results(self):
        """Create advanced visualizations"""
        print("\n📈 Creating Advanced Visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. ROC Curves Comparison
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.all_results)))

        for i, (balance_method, method_results) in enumerate(self.all_results.items()):
            for model_name, result in method_results.items():
                if 'Isolation' not in model_name:  # Skip isolation forest for ROC
                    fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                    ax1.plot(fpr, tpr,
                           label=f"{model_name}_{balance_method} (AUC={result['roc_auc']:.3f})",
                           alpha=0.7)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for balance_method, method_results in self.all_results.items():
            for model_name, result in method_results.items():
                if 'Isolation' not in model_name:
                    precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
                    ax2.plot(recall, precision,
                           label=f"{model_name}_{balance_method} (AP={result['avg_precision']:.3f})",
                           alpha=0.7)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. F1-Score Comparison
        ax3 = axes[0, 2]
        comparison_df = self.create_comprehensive_comparison()
        top_models = comparison_df.head(10)

        bars = ax3.barh(range(len(top_models)), top_models['F1_Score'])
        ax3.set_yticks(range(len(top_models)))
        ax3.set_yticklabels([f"{row['Model']}_{row['Balance_Method']}"
                           for _, row in top_models.iterrows()], fontsize=8)
        ax3.set_xlabel('F1-Score')
        ax3.set_title('Top 10 Models by F1-Score')
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)

        # 4. Confusion Matrix for Best Model
        ax4 = axes[1, 0]
        if self.best_model:
            best_result = self.best_model['result']
            cm = confusion_matrix(self.y_test, best_result['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
            ax4.set_title(f'Confusion Matrix - Best Model\n{self.best_model["model_name"]}')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')

        # 5. Feature Importance (if available)
        ax5 = axes[1, 1]
        if self.best_model and hasattr(self.best_model['result']['model'], 'feature_importances_'):
            model = self.best_model['result']['model']
            feature_names = self.X_train.columns

            # Get top 15 features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            ax5.barh(range(len(importance_df)), importance_df['importance'])
            ax5.set_yticks(range(len(importance_df)))
            ax5.set_yticklabels(importance_df['feature'], fontsize=8)
            ax5.set_xlabel('Importance')
            ax5.set_title('Top 15 Feature Importance')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Feature Importance\nNot Available',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Feature Importance')

        # 6. Model Performance Metrics Radar Chart
        ax6 = axes[1, 2]
        if self.best_model:
            metrics = ['ROC_AUC', 'Precision', 'Recall', 'F1_Score', 'Avg_Precision']
            values = [self.best_model['result'][metric.lower()] for metric in metrics]

            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))

            ax6.plot(angles, values, 'o-', linewidth=2, label='Best Model')
            ax6.fill(angles, values, alpha=0.25)
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(metrics)
            ax6.set_ylim(0, 1)
            ax6.set_title(f'Performance Radar - {self.best_model["model_name"]}')
            ax6.grid(True)

        plt.tight_layout()
        plt.savefig('advanced_fraud_detection_results.png', dpi=150, bbox_inches='tight')
        print("✅ Advanced visualization saved as 'advanced_fraud_detection_results.png'")
        plt.show()

    def run_complete_advanced_pipeline(self):
        """Run the complete advanced fraud detection pipeline"""
        print("🚀 Starting Advanced Credit Card Fraud Detection Pipeline")
        print("="*80)

        # Step 1: Load and explore data
        self.load_and_explore_data()

        # Step 2: Preprocess data with feature engineering
        self.preprocess_data()

        # Step 3: Handle class imbalance
        self.handle_class_imbalance()

        # Step 4: Initialize advanced models
        self.initialize_advanced_models()

        # Step 5: Train and evaluate all models
        self.train_and_evaluate_all_advanced()

        # Step 6: Find best model
        self.find_best_model()

        # Step 7: Create comprehensive comparison
        comparison_df = self.create_comprehensive_comparison()

        # Step 8: Create advanced visualizations
        self.plot_advanced_results()

        print("\n" + "="*80)
        print("🎉 ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)

        if self.best_model:
            print(f"\n🏆 BEST MODEL SUMMARY:")
            print(f"Model: {self.best_model['model_name']}")
            print(f"Balance Method: {self.best_model['balance_method']}")
            print(f"F1-Score: {self.best_model['score']:.4f}")
            print(f"ROC-AUC: {self.best_model['result']['roc_auc']:.4f}")
            print(f"Precision: {self.best_model['result']['precision']:.4f}")
            print(f"Recall: {self.best_model['result']['recall']:.4f}")

        return {
            'best_model': self.best_model,
            'comparison': comparison_df,
            'all_results': self.all_results
        }


def main():
    """Main function to run advanced fraud detection"""
    # Update path to your dataset
    data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"

    # Initialize and run advanced pipeline
    detector = AdvancedFraudDetector(data_path)
    results = detector.run_complete_advanced_pipeline()

    return results


if __name__ == "__main__":
    results = main()
