# Credit Card Fraud Detection - Project Setup Guide

## 🚀 Quick Start

### 1. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 2. Update File Paths
Update the dataset path in both files:
- `credit_card_fraud_detection.ipynb` (cell 3)
- `fraud_detection_script.py` (line 285)

Change this line:
```python
data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
```

### 3. Run the Project

#### Option A: Jupyter Notebook (Recommended)
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

#### Option B: Python Script
```bash
python fraud_detection_script.py
```

## 📊 What the Project Does

### 1. **Data Exploration**
- Loads and analyzes the credit card dataset
- Visualizes class distribution (Normal vs Fraud)
- Explores feature distributions and relationships

### 2. **Data Preprocessing**
- Scales Amount and Time features using RobustScaler
- Handles missing values (if any)
- Prepares features for machine learning

### 3. **Class Imbalance Handling**
- Applies SMOTE (Synthetic Minority Oversampling Technique)
- Balances the dataset for better model training

### 4. **Model Training**
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble method with feature importance
- **Support Vector Machine**: Non-linear classification

### 5. **Model Evaluation**
- **Precision**: How many predicted frauds are actually frauds
- **Recall**: How many actual frauds were detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

### 6. **Results Visualization**
- ROC curves comparison
- Feature importance analysis
- Performance metrics comparison

## 🎯 Expected Results

### Key Insights You'll Discover:
1. **Highly Imbalanced Dataset**: ~99.83% normal, ~0.17% fraud
2. **SMOTE Impact**: Improves recall but may reduce precision
3. **Model Performance**: Different models excel at different metrics
4. **Important Features**: Certain V-features are most predictive

### Typical Performance Ranges:
- **Precision**: 0.85-0.95
- **Recall**: 0.75-0.90
- **F1-Score**: 0.80-0.92
- **ROC-AUC**: 0.95-0.99

## 🔧 Troubleshooting

### Common Issues:

#### 1. **File Not Found Error**
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: Update the `data_path` variable with correct file location

#### 2. **Memory Error**
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Use data sampling: `df = df.sample(n=50000, random_state=42)`
- Increase virtual memory
- Use a machine with more RAM

#### 3. **Import Errors**
```
ModuleNotFoundError: No module named 'imblearn'
```
**Solution**: Install missing packages
```bash
pip install imbalanced-learn
```

#### 4. **Slow Performance**
**Solutions**:
- Reduce dataset size for testing
- Use fewer estimators in Random Forest
- Skip SVM for large datasets

## 📈 Next Steps for Enhancement

### 1. **Advanced Models**
```python
# Add these to your models dictionary
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

models['XGBoost'] = XGBClassifier(random_state=42)
models['Neural Network'] = MLPClassifier(random_state=42, max_iter=500)
```

### 2. **Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
```

### 3. **Feature Engineering**
- Create time-based features (hour, day, month)
- Transaction frequency features
- Amount percentile features
- Rolling window statistics

### 4. **Model Deployment**
- Save best model using joblib
- Create Flask/FastAPI web service
- Implement real-time prediction API

### 5. **Advanced Evaluation**
- Cost-sensitive evaluation
- Precision-Recall curves
- Learning curves
- Cross-validation with time series split

## 📚 Learning Objectives

By completing this project, you will learn:

1. **Data Science Workflow**: Complete ML pipeline from data to deployment
2. **Imbalanced Classification**: Techniques for handling skewed datasets
3. **Model Evaluation**: Appropriate metrics for fraud detection
4. **Feature Analysis**: Understanding what drives predictions
5. **Business Impact**: Translating ML metrics to business value

## 🎓 Project Deliverables

### For Academic Submission:
1. **Jupyter Notebook**: With complete analysis and visualizations
2. **Python Script**: Clean, documented code
3. **Report**: Summary of findings and recommendations
4. **Presentation**: Key insights and model performance

### Recommended Report Structure:
1. **Introduction**: Problem statement and objectives
2. **Data Analysis**: Dataset characteristics and exploration
3. **Methodology**: Preprocessing and modeling approach
4. **Results**: Model performance and comparison
5. **Conclusions**: Key findings and business recommendations
6. **Future Work**: Potential improvements and extensions

## 💡 Tips for Success

1. **Start Simple**: Begin with basic models before advanced techniques
2. **Understand Metrics**: Focus on precision/recall trade-offs for fraud detection
3. **Visualize Everything**: Use plots to understand data and results
4. **Document Process**: Keep detailed notes of your approach
5. **Iterate**: Try different approaches and compare results

Good luck with your Credit Card Fraud Detection project! 🚀
