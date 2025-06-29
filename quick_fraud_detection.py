#!/usr/bin/env python3
"""
Quick Credit Card Fraud Detection
A simplified version that runs the essential fraud detection analysis
"""

def test_imports():
    """Test if all required libraries are available"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        from sklearn.preprocessing import StandardScaler
        print("✅ All libraries imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def quick_fraud_analysis():
    """Run a quick fraud detection analysis"""
    
    if not test_imports():
        print("Please install missing libraries first.")
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    
    print("🚀 Starting Quick Credit Card Fraud Detection Analysis")
    print("="*60)
    
    # Load data
    data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
    
    try:
        print("📊 Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape}")
        
        # Quick data overview
        print(f"\n📈 Dataset Overview:")
        print(f"Total transactions: {len(df):,}")
        print(f"Features: {df.shape[1]}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Class distribution
        class_dist = df['Class'].value_counts()
        fraud_rate = (class_dist[1] / len(df)) * 100
        print(f"\n🎯 Class Distribution:")
        print(f"Normal: {class_dist[0]:,} ({100-fraud_rate:.2f}%)")
        print(f"Fraud: {class_dist[1]:,} ({fraud_rate:.2f}%)")
        
        # Prepare data for ML
        print(f"\n🔧 Preparing data for machine learning...")
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🤖 Training {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ {name} trained successfully!")
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            
            # Classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # Compare models
        print(f"\n📊 Model Comparison:")
        print("-" * 40)
        for name, result in results.items():
            print(f"{name}: ROC-AUC = {result['roc_auc']:.4f}")
        
        # Feature importance (Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        # Simple visualization
        try:
            print(f"\n📈 Creating visualizations...")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Class distribution
            class_dist.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
            axes[0,0].set_title('Class Distribution')
            axes[0,0].set_xlabel('Class')
            axes[0,0].set_ylabel('Count')
            
            # Amount distribution
            df['Amount'].hist(bins=50, ax=axes[0,1], alpha=0.7, color='green')
            axes[0,1].set_title('Transaction Amount Distribution')
            axes[0,1].set_xlabel('Amount')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_yscale('log')
            
            # ROC comparison
            from sklearn.metrics import roc_curve
            for name, result in results.items():
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                axes[1,0].plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
            
            axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1,0].set_xlabel('False Positive Rate')
            axes[1,0].set_ylabel('True Positive Rate')
            axes[1,0].set_title('ROC Curves')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Feature importance
            if 'Random Forest' in results:
                top_features = feature_importance.head(10)
                axes[1,1].barh(range(len(top_features)), top_features['importance'])
                axes[1,1].set_yticks(range(len(top_features)))
                axes[1,1].set_yticklabels(top_features['feature'])
                axes[1,1].set_xlabel('Importance')
                axes[1,1].set_title('Top 10 Feature Importance')
            
            plt.tight_layout()
            plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
            print("✅ Visualization saved as 'fraud_detection_results.png'")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"="*60)
        print(f"📋 Summary:")
        print(f"- Dataset: {len(df):,} transactions")
        print(f"- Fraud rate: {fraud_rate:.3f}%")
        print(f"- Best model: {max(results.keys(), key=lambda x: results[x]['roc_auc'])}")
        print(f"- Best ROC-AUC: {max(result['roc_auc'] for result in results.values()):.4f}")
        
        return results
        
    except FileNotFoundError:
        print("❌ Dataset file not found!")
        print("Please check the file path in the script.")
        return None
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def main():
    """Main function"""
    results = quick_fraud_analysis()
    
    if results:
        print(f"\n🚀 Ready for advanced analysis!")
        print(f"You can now run:")
        print(f"- Full Jupyter notebook: credit_card_fraud_detection.ipynb")
        print(f"- Complete Python script: fraud_detection_script.py")
    else:
        print(f"\n🔧 Please fix the issues above and try again.")

if __name__ == "__main__":
    main()
