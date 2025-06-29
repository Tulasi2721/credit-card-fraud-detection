
# Credit Card Fraud Detection - Installation Guide

## Step 1: Install Required Libraries
Run these commands in your terminal/command prompt:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn imbalanced-learn
```

## Step 2: Verify Installation
Run this Python script to check if everything is installed:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
print("All libraries installed successfully!")
```

## Step 3: Update File Path
In your Python scripts, update the data_path variable:

```python
data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
```

## Step 4: Run the Project
- For Jupyter Notebook: `jupyter notebook credit_card_fraud_detection.ipynb`
- For Python Script: `python fraud_detection_script.py`

## Troubleshooting
- If pip install fails, try: `pip install --user [package_name]`
- For permission errors, run command prompt as administrator
- If download is slow, try: `pip install --timeout 1000 [package_name]`
