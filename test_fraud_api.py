#!/usr/bin/env python3
"""
Test Client for Real-Time Fraud Detection API
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

class FraudAPIClient:
    """
    Client for testing the fraud detection API
    """
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def train_model(self, data_path=None):
        """Train the model via API"""
        try:
            payload = {}
            if data_path:
                payload["data_path"] = data_path
            
            response = requests.post(f"{self.base_url}/train", json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, transaction):
        """Predict fraud for a single transaction"""
        try:
            response = requests.post(f"{self.base_url}/predict", json=transaction)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, transactions):
        """Predict fraud for multiple transactions"""
        try:
            payload = {"transactions": transactions}
            response = requests.post(f"{self.base_url}/batch_predict", json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model_info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def create_sample_transactions():
    """Create sample transactions for testing"""
    
    # Normal transaction examples
    normal_transactions = [
        {
            "Time": 3600,  # 1 hour
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 2.536347,
            "V4": 1.378155,
            "V5": -0.338321,
            "V6": 0.462388,
            "V7": 0.239599,
            "V8": 0.098698,
            "V9": 0.363787,
            "V10": 0.090794,
            "V11": -0.551600,
            "V12": -0.617801,
            "V13": -0.991390,
            "V14": -0.311169,
            "V15": 1.468177,
            "V16": -0.470401,
            "V17": 0.207971,
            "V18": 0.025791,
            "V19": 0.403993,
            "V20": 0.251412,
            "V21": -0.018307,
            "V22": 0.277838,
            "V23": -0.110474,
            "V24": 0.066928,
            "V25": 0.128539,
            "V26": -0.189115,
            "V27": 0.133558,
            "V28": -0.021053,
            "Amount": 149.62
        },
        {
            "Time": 7200,  # 2 hours
            "V1": 1.191857,
            "V2": 0.266151,
            "V3": 0.166480,
            "V4": 0.448154,
            "V5": 0.060018,
            "V6": -0.082361,
            "V7": -0.078803,
            "V8": 0.085102,
            "V9": -0.255425,
            "V10": -0.166974,
            "V11": 1.612727,
            "V12": 1.065235,
            "V13": 0.489095,
            "V14": -0.143772,
            "V15": 0.635558,
            "V16": 0.463917,
            "V17": -0.114805,
            "V18": -0.183361,
            "V19": -0.145783,
            "V20": -0.069083,
            "V21": -0.225775,
            "V22": -0.638672,
            "V23": 0.101288,
            "V24": -0.339846,
            "V25": 0.167170,
            "V26": 0.125895,
            "V27": -0.008983,
            "V28": 0.014724,
            "Amount": 2.69
        }
    ]
    
    # Suspicious transaction examples (high amounts, unusual patterns)
    suspicious_transactions = [
        {
            "Time": 86400,  # 1 day
            "V1": -3.043541,
            "V2": -3.157307,
            "V3": 1.088463,
            "V4": 2.288644,
            "V5": 1.359805,
            "V6": -1.064823,
            "V7": 0.325574,
            "V8": -0.067794,
            "V9": -0.270533,
            "V10": -0.838587,
            "V11": 1.202613,
            "V12": 0.618307,
            "V13": 0.610219,
            "V14": -0.686180,
            "V15": 0.679145,
            "V16": 0.392087,
            "V17": 0.675775,
            "V18": -0.094019,
            "V19": 0.313267,
            "V20": 0.014866,
            "V21": -0.342475,
            "V22": 0.721254,
            "V23": -0.110927,
            "V24": 0.335664,
            "V25": 0.160717,
            "V26": 0.123205,
            "V27": -0.569159,
            "V28": 0.133041,
            "Amount": 9999.99  # Very high amount
        },
        {
            "Time": 90000,  # Late night
            "V1": -2.312227,
            "V2": 1.951992,
            "V3": -1.609851,
            "V4": 3.997906,
            "V5": -0.522188,
            "V6": -1.426545,
            "V7": -2.537387,
            "V8": 1.391657,
            "V9": -2.770089,
            "V10": -2.772272,
            "V11": 3.202033,
            "V12": -2.899907,
            "V13": -0.595221,
            "V14": -4.289254,
            "V15": 0.389724,
            "V16": -1.140651,
            "V17": -2.830055,
            "V18": -0.168224,
            "V19": 0.177839,
            "V20": 0.507757,
            "V21": -0.287924,
            "V22": -0.631418,
            "V23": -0.053527,
            "V24": -0.026698,
            "V25": 0.275487,
            "V26": -0.181287,
            "V27": 0.197145,
            "V28": 0.042472,
            "Amount": 5000.00  # High amount at night
        }
    ]
    
    return normal_transactions, suspicious_transactions

def test_api_comprehensive():
    """Comprehensive API testing"""
    print("🧪 Starting Comprehensive API Testing")
    print("="*60)
    
    client = FraudAPIClient()
    
    # 1. Health Check
    print("\n1️⃣ Health Check...")
    health = client.health_check()
    print(f"Health Status: {health}")
    
    # 2. Model Info (before training)
    print("\n2️⃣ Model Info (before training)...")
    model_info = client.get_model_info()
    print(f"Model Info: {model_info}")
    
    # 3. Train Model
    print("\n3️⃣ Training Model...")
    train_result = client.train_model()
    print(f"Training Result: {train_result}")
    
    if train_result.get('status') == 'success':
        print("✅ Model trained successfully!")
        
        # 4. Model Info (after training)
        print("\n4️⃣ Model Info (after training)...")
        model_info = client.get_model_info()
        print(f"Model Info: {model_info}")
        
        # 5. Create test transactions
        normal_transactions, suspicious_transactions = create_sample_transactions()
        
        # 6. Test single predictions
        print("\n5️⃣ Testing Single Predictions...")
        
        print("\n🟢 Normal Transaction Test:")
        normal_result = client.predict_single(normal_transactions[0])
        print(f"Result: {normal_result}")
        
        print("\n🔴 Suspicious Transaction Test:")
        suspicious_result = client.predict_single(suspicious_transactions[0])
        print(f"Result: {suspicious_result}")
        
        # 7. Test batch predictions
        print("\n6️⃣ Testing Batch Predictions...")
        all_transactions = normal_transactions + suspicious_transactions
        batch_result = client.predict_batch(all_transactions)
        print(f"Batch Result Summary: {batch_result.get('status')}")
        print(f"Total Transactions: {batch_result.get('total_transactions')}")
        print(f"Successful Predictions: {batch_result.get('successful_predictions')}")
        
        if batch_result.get('predictions'):
            print("\n📊 Individual Predictions:")
            for pred in batch_result['predictions']:
                if 'error' not in pred:
                    print(f"Transaction {pred['transaction_id']}: "
                          f"Fraud={pred['is_fraud']}, "
                          f"Probability={pred['fraud_probability']:.4f}, "
                          f"Risk={pred['risk_level']}")
                else:
                    print(f"Transaction {pred['transaction_id']}: Error - {pred['error']}")
        
        # 8. Performance testing
        print("\n7️⃣ Performance Testing...")
        start_time = time.time()
        
        for i in range(10):
            client.predict_single(normal_transactions[0])
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        print(f"Average prediction time: {avg_time:.4f} seconds")
        print(f"Predictions per second: {1/avg_time:.2f}")
        
    else:
        print("❌ Model training failed!")
    
    print("\n" + "="*60)
    print("🎉 API Testing Completed!")

def demo_real_time_detection():
    """Demonstrate real-time fraud detection"""
    print("\n🚀 Real-Time Fraud Detection Demo")
    print("="*50)
    
    client = FraudAPIClient()
    
    # Check if model is trained
    health = client.health_check()
    if not health.get('model_trained', False):
        print("Training model first...")
        train_result = client.train_model()
        if train_result.get('status') != 'success':
            print("❌ Failed to train model")
            return
    
    # Simulate real-time transactions
    normal_transactions, suspicious_transactions = create_sample_transactions()
    
    print("\n📡 Simulating Real-Time Transaction Processing...")
    
    all_transactions = normal_transactions + suspicious_transactions
    transaction_types = ['Normal', 'Normal', 'Suspicious', 'Suspicious']
    
    for i, (transaction, tx_type) in enumerate(zip(all_transactions, transaction_types)):
        print(f"\n🔄 Processing Transaction {i+1} ({tx_type})...")
        
        start_time = time.time()
        result = client.predict_single(transaction)
        end_time = time.time()
        
        if result.get('status') == 'success':
            pred = result['prediction']
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"⚡ Processing Time: {processing_time:.2f}ms")
            print(f"💰 Amount: ${transaction['Amount']:.2f}")
            print(f"🎯 Fraud Probability: {pred['fraud_probability']:.4f}")
            print(f"🚨 Risk Level: {pred['risk_level']}")
            print(f"✅ Decision: {'BLOCK' if pred['is_fraud'] else 'APPROVE'}")
            
            # Simulate decision making
            if pred['is_fraud']:
                print("🛑 TRANSACTION BLOCKED - Manual review required")
            elif pred['risk_level'] == 'MEDIUM':
                print("⚠️ ADDITIONAL VERIFICATION - SMS/Email confirmation")
            else:
                print("✅ TRANSACTION APPROVED")
        else:
            print(f"❌ Error: {result}")
        
        # Simulate processing delay
        time.sleep(0.5)
    
    print("\n🎉 Real-Time Demo Completed!")

if __name__ == "__main__":
    print("🔍 Credit Card Fraud Detection API Testing")
    print("="*60)
    
    # Run comprehensive testing
    test_api_comprehensive()
    
    # Run real-time demo
    demo_real_time_detection()
    
    print("\n📋 API Endpoints Available:")
    print("- GET  /health          - Health check")
    print("- POST /train           - Train model")
    print("- POST /predict         - Single prediction")
    print("- POST /batch_predict   - Batch predictions")
    print("- GET  /model_info      - Model information")
    
    print("\n🌐 API running at: http://localhost:5000")
    print("📖 Use this client script to test the API functionality!")
