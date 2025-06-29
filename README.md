# Credit Card Fraud Detection - Machine Learning Project

## Project Overview
This project implements machine learning algorithms to detect fraudulent credit card transactions using various classification techniques.

## Dataset
- **Source**: Credit card transaction dataset
- **File**: `creditcard.csv`
- **Features**: Transaction details with anonymized features (V1-V28) + Time, Amount, Class
- **Target**: Binary classification (0 = Normal, 1 = Fraud)

## Project Structure
```
credit_card_fraud_detection/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── splits/                 # Train/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── models/                     # Saved trained models
├── results/                    # Model results and metrics
├── reports/                    # Analysis reports and visualizations
└── requirements.txt
```

## Key Challenges in Fraud Detection
1. **Imbalanced Dataset**: Fraud cases are typically <1% of all transactions
2. **Feature Engineering**: Working with anonymized features
3. **Model Selection**: Choosing appropriate algorithms for imbalanced data
4. **Evaluation Metrics**: Precision, Recall, F1-score more important than accuracy
5. **Real-time Performance**: Model must be fast for production use

## Machine Learning Approaches
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **Neural Networks**
- **Isolation Forest** (Anomaly Detection)

## Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score
- Precision-Recall Curve
- Cost-sensitive evaluation

## Getting Started
1. Place your dataset in `data/raw/creditcard.csv`
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks in order or execute Python scripts
4. Check results in `results/` folder

## Next Steps
- [ ] Data exploration and visualization
- [ ] Handle class imbalance (SMOTE, undersampling)
- [ ] Feature engineering and selection
- [ ] Model training and hyperparameter tuning
- [ ] Model evaluation and comparison
- [ ] Deploy best model
