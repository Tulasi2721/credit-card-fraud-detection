#!/usr/bin/env python3
"""
Basic Credit Card Fraud Detection Test
This script tests if we can load and analyze the credit card dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_data_loading():
    """Test if we can load the credit card dataset"""
    try:
        # Update this path to your actual dataset location
        data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
        
        print("Attempting to load dataset...")
        df = pd.read_csv(data_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print("❌ Dataset file not found!")
        print("Please update the data_path variable with the correct file location.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def basic_analysis(df):
    """Perform basic analysis of the dataset"""
    if df is None:
        return
    
    print("\n" + "="*50)
    print("BASIC DATASET ANALYSIS")
    print("="*50)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Class distribution
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts()
        class_percentages = df['Class'].value_counts(normalize=True) * 100
        
        print(f"\nClass Distribution:")
        print(f"Normal transactions (0): {class_counts[0]:,} ({class_percentages[0]:.2f}%)")
        print(f"Fraudulent transactions (1): {class_counts[1]:,} ({class_percentages[1]:.2f}%)")
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"Total transactions: {len(df):,}")
        print(f"Fraud rate: {class_percentages[1]:.3f}%")
        
        if 'Amount' in df.columns:
            print(f"\nAmount Statistics:")
            print(f"Average transaction amount: ${df['Amount'].mean():.2f}")
            print(f"Median transaction amount: ${df['Amount'].median():.2f}")
            print(f"Max transaction amount: ${df['Amount'].max():.2f}")
            
            print(f"\nFraud vs Normal Amount Comparison:")
            fraud_amounts = df[df['Class'] == 1]['Amount']
            normal_amounts = df[df['Class'] == 0]['Amount']
            
            print(f"Average fraud amount: ${fraud_amounts.mean():.2f}")
            print(f"Average normal amount: ${normal_amounts.mean():.2f}")
    
    # Display first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return True

def simple_visualization(df):
    """Create simple visualizations"""
    if df is None:
        return
    
    try:
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Class distribution plot
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Class distribution
        plt.subplot(1, 2, 1)
        class_counts = df['Class'].value_counts()
        plt.bar(['Normal', 'Fraud'], class_counts.values, color=['skyblue', 'salmon'])
        plt.title('Class Distribution')
        plt.ylabel('Count')
        
        # Subplot 2: Amount distribution
        plt.subplot(1, 2, 2)
        plt.hist(df['Amount'], bins=50, alpha=0.7, color='green')
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')  # Log scale due to skewed distribution
        
        plt.tight_layout()
        plt.savefig('credit_card_fraud_detection/basic_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'basic_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def simple_ml_test(df):
    """Test basic machine learning functionality"""
    if df is None:
        return
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        print("\n" + "="*50)
        print("BASIC MACHINE LEARNING TEST")
        print("="*50)
        
        # Prepare data
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train simple logistic regression
        print("Training Logistic Regression model...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Please install scikit-learn: pip install scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Error in ML test: {e}")
        return False

def main():
    """Main function to run all tests"""
    print("🚀 Starting Credit Card Fraud Detection Test")
    print("="*60)
    
    # Test 1: Data loading
    df = test_data_loading()
    
    if df is not None:
        # Test 2: Basic analysis
        basic_analysis(df)
        
        # Test 3: Simple visualization
        simple_visualization(df)
        
        # Test 4: Basic ML test
        simple_ml_test(df)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED!")
        print("="*60)
        print("\nNext steps:")
        print("1. If all tests passed, you can run the full Jupyter notebook")
        print("2. If any test failed, check the error messages above")
        print("3. Make sure all required libraries are installed")
        
    else:
        print("\n❌ Cannot proceed without dataset. Please check file path.")

if __name__ == "__main__":
    main()
