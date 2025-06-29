#!/usr/bin/env python3
"""
Real-Time Credit Card Fraud Detection API
Flask-based API for real-time fraud detection
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionAPI:
    """
    Real-time fraud detection API
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.threshold = 0.5
        self.is_trained = False
        
    def train_model(self, data_path):
        """Train the fraud detection model"""
        logger.info("Training fraud detection model...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Prepare features
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        
        # Optimize threshold (simplified)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        self.threshold = self._optimize_threshold(y, y_pred_proba)
        
        self.is_trained = True
        logger.info(f"Model trained successfully. Optimal threshold: {self.threshold:.4f}")
        
        return True
    
    def _engineer_features(self, df):
        """Engineer features for fraud detection"""
        # Time-based features
        df['Hour'] = (df['Time'] % 86400) // 3600
        df['Day'] = df['Time'] // 86400
        df['Is_Weekend'] = ((df['Time'] // 86400) % 7 >= 5).astype(int)
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        
        # Amount-based features
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Amount_Sqrt'] = np.sqrt(df['Amount'])
        df['Is_Round_Amount'] = (df['Amount'] % 1 == 0).astype(int)
        
        # Amount categories
        amount_bins = [0, 10, 50, 100, 500, float('inf')]
        df['Amount_Category'] = pd.cut(df['Amount'], bins=amount_bins, labels=False)
        
        # Statistical features
        df['Amount_Zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        
        return df
    
    def _optimize_threshold(self, y_true, y_pred_proba):
        """Simple threshold optimization"""
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_fraud(self, transaction_data):
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame(transaction_data)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select and order features
        X = df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        fraud_probability = self.model.predict_proba(X_scaled)[0, 1]
        is_fraud = fraud_probability >= self.threshold
        
        # Risk scoring
        if fraud_probability >= 0.8:
            risk_level = "HIGH"
        elif fraud_probability >= 0.5:
            risk_level = "MEDIUM"
        elif fraud_probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "VERY_LOW"
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'risk_level': risk_level,
            'threshold_used': float(self.threshold),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.threshold = model_data['threshold']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


# Initialize Flask app and fraud detector
app = Flask(__name__)
fraud_detector = FraudDetectionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': fraud_detector.is_trained,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train the fraud detection model"""
    try:
        data = request.get_json()
        data_path = data.get('data_path', r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv")
        
        success = fraud_detector.train_model(data_path)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'threshold': fraud_detector.threshold,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model training failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for a transaction"""
    try:
        if not fraud_detector.is_trained:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained. Call /train endpoint first.'
            }), 400
        
        transaction_data = request.get_json()
        
        # Validate required fields
        required_fields = ['Time', 'Amount']
        for field in required_fields:
            if field not in transaction_data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Make prediction
        result = fraud_detector.predict_fraud(transaction_data)
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict fraud for multiple transactions"""
    try:
        if not fraud_detector.is_trained:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained. Call /train endpoint first.'
            }), 400
        
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({
                'status': 'error',
                'message': 'No transactions provided'
            }), 400
        
        results = []
        for i, transaction in enumerate(transactions):
            try:
                result = fraud_detector.predict_fraud(transaction)
                result['transaction_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'transaction_id': i,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'total_transactions': len(transactions),
            'successful_predictions': len([r for r in results if 'error' not in r])
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if not fraud_detector.is_trained:
        return jsonify({
            'status': 'error',
            'message': 'Model not trained'
        }), 400
    
    return jsonify({
        'status': 'success',
        'model_info': {
            'model_type': 'RandomForestClassifier',
            'threshold': fraud_detector.threshold,
            'feature_count': len(fraud_detector.feature_columns),
            'features': fraud_detector.feature_columns[:10],  # First 10 features
            'is_trained': fraud_detector.is_trained
        }
    })

if __name__ == '__main__':
    # Auto-train model on startup
    try:
        data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
        fraud_detector.train_model(data_path)
        logger.info("Model auto-trained on startup")
    except Exception as e:
        logger.warning(f"Auto-training failed: {e}")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
