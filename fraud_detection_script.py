#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Python Script Version
Author: Your Name
Date: 2024

This script implements machine learning algorithms to detect fraudulent credit card transactions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    A complete pipeline for credit card fraud detection
    """
    
    def __init__(self, data_path):
        """
        Initialize the fraud detection pipeline
        
        Args:
            data_path (str): Path to the credit card dataset
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Class distribution
        class_counts = self.df['Class'].value_counts()
        class_percentages = self.df['Class'].value_counts(normalize=True) * 100
        
        print("\nClass Distribution:")
        print(f"Normal transactions (0): {class_counts[0]:,} ({class_percentages[0]:.2f}%)")
        print(f"Fraudulent transactions (1): {class_counts[1]:,} ({class_percentages[1]:.2f}%)")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data: scaling and feature preparation"""
        print("\nPreprocessing data...")
        
        # Feature scaling
        scaler = RobustScaler()
        self.df['Amount_scaled'] = scaler.fit_transform(self.df[['Amount']])
        self.df['Time_scaled'] = scaler.fit_transform(self.df[['Time']])
        
        # Drop original columns
        df_processed = self.df.drop(['Time', 'Amount'], axis=1)
        
        # Prepare features and target
        X = df_processed.drop('Class', axis=1)
        y = df_processed['Class']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smote(self):
        """Apply SMOTE to handle class imbalance"""
        print("\nApplying SMOTE for class balancing...")
        
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"Original training set shape: {self.X_train.shape}")
        print(f"SMOTE training set shape: {X_train_smote.shape}")
        
        return X_train_smote, y_train_smote
    
    def initialize_models(self):
        """Initialize machine learning models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        print(f"\nInitialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"- {name}")
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate a single model and return metrics
        
        Args:
            model: Scikit-learn model
            X_train, X_test, y_train, y_test: Train/test data
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name} Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        
        return {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_and_evaluate_all(self, use_smote=True):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING MODELS")
        print("="*60)
        
        if use_smote:
            X_train_balanced, y_train_balanced = self.apply_smote()
        else:
            X_train_balanced, y_train_balanced = self.X_train, self.y_train
        
        self.results = {}
        for name, model in self.models.items():
            suffix = " (SMOTE)" if use_smote else ""
            self.results[name] = self.evaluate_model(
                model, X_train_balanced, self.X_test, 
                y_train_balanced, self.y_test, 
                f"{name}{suffix}"
            )
    
    def compare_models(self):
        """Compare model performance"""
        if not self.results:
            print("No results to compare. Run train_and_evaluate_all() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'ROC-AUC': result['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        return comparison_df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        if not self.results:
            print("No results to plot. Run train_and_evaluate_all() first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        if 'Random Forest' not in self.results:
            print("Random Forest model not found in results.")
            return None
        
        rf_model = self.results['Random Forest']['model']
        feature_names = self.X_train.columns
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def run_complete_pipeline(self):
        """Run the complete fraud detection pipeline"""
        print("Starting Credit Card Fraud Detection Pipeline...")
        print("="*60)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate models
        self.train_and_evaluate_all(use_smote=True)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Plot ROC curves
        self.plot_roc_curves()
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'comparison': comparison_df,
            'feature_importance': feature_importance,
            'results': self.results
        }


def main():
    """Main function to run the fraud detection pipeline"""
    # Update this path to your actual dataset location
    data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
    
    # Initialize and run pipeline
    pipeline = FraudDetectionPipeline(data_path)
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()
