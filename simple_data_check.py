#!/usr/bin/env python3
"""
Simple Credit Card Dataset Check
This script checks if we can access and read the credit card dataset using basic Python
"""

import csv
import os

def check_file_exists():
    """Check if the credit card dataset file exists"""
    # Update this path to your actual dataset location
    data_path = r"C:\Users\dasar\OneDrive\Desktop\ds project\major\creditcard.csv"
    
    print("🔍 Checking if dataset file exists...")
    print(f"Looking for: {data_path}")
    
    if os.path.exists(data_path):
        print("✅ Dataset file found!")
        return data_path
    else:
        print("❌ Dataset file not found!")
        print("\nPlease check:")
        print("1. The file path is correct")
        print("2. The file exists at the specified location")
        print("3. You have permission to access the file")
        return None

def basic_csv_analysis(file_path):
    """Perform basic analysis using Python's built-in csv module"""
    if not file_path:
        return
    
    print("\n" + "="*50)
    print("BASIC CSV ANALYSIS")
    print("="*50)
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Read first few lines to understand structure
            reader = csv.reader(csvfile)
            
            # Get header
            header = next(reader)
            print(f"✅ Successfully opened CSV file!")
            print(f"Number of columns: {len(header)}")
            print(f"Column names: {header}")
            
            # Count rows and analyze data
            row_count = 0
            fraud_count = 0
            normal_count = 0
            amounts = []
            
            print("\n📊 Analyzing data...")
            
            for row in reader:
                row_count += 1
                
                # Check class (last column should be 'Class')
                if len(row) > 0:
                    try:
                        class_value = int(float(row[-1]))  # Last column is Class
                        if class_value == 1:
                            fraud_count += 1
                        else:
                            normal_count += 1
                    except (ValueError, IndexError):
                        pass
                
                # Collect amount data (second to last column should be 'Amount')
                if len(row) > 1:
                    try:
                        amount = float(row[-2])  # Second to last column is Amount
                        amounts.append(amount)
                    except (ValueError, IndexError):
                        pass
                
                # Progress indicator for large files
                if row_count % 50000 == 0:
                    print(f"   Processed {row_count:,} rows...")
            
            print(f"\n📈 ANALYSIS RESULTS:")
            print(f"Total transactions: {row_count:,}")
            print(f"Normal transactions: {normal_count:,}")
            print(f"Fraudulent transactions: {fraud_count:,}")
            
            if row_count > 0:
                fraud_rate = (fraud_count / row_count) * 100
                print(f"Fraud rate: {fraud_rate:.3f}%")
            
            if amounts:
                avg_amount = sum(amounts) / len(amounts)
                min_amount = min(amounts)
                max_amount = max(amounts)
                
                print(f"\n💰 AMOUNT STATISTICS:")
                print(f"Average transaction: ${avg_amount:.2f}")
                print(f"Minimum transaction: ${min_amount:.2f}")
                print(f"Maximum transaction: ${max_amount:.2f}")
            
            return True
            
    except FileNotFoundError:
        print("❌ File not found!")
        return False
    except PermissionError:
        print("❌ Permission denied! Check file permissions.")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def check_python_libraries():
    """Check which Python libraries are available"""
    print("\n" + "="*50)
    print("CHECKING AVAILABLE LIBRARIES")
    print("="*50)
    
    libraries = {
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computing',
        'matplotlib': 'Data visualization',
        'sklearn': 'Machine learning (scikit-learn)',
        'seaborn': 'Statistical data visualization',
        'imblearn': 'Imbalanced learning'
    }
    
    available = []
    missing = []
    
    for lib, description in libraries.items():
        try:
            if lib == 'sklearn':
                import sklearn
            else:
                __import__(lib)
            print(f"✅ {lib}: {description}")
            available.append(lib)
        except ImportError:
            print(f"❌ {lib}: {description} - NOT INSTALLED")
            missing.append(lib)
    
    print(f"\n📊 SUMMARY:")
    print(f"Available libraries: {len(available)}")
    print(f"Missing libraries: {len(missing)}")
    
    if missing:
        print(f"\n🔧 TO INSTALL MISSING LIBRARIES:")
        print(f"pip install {' '.join(missing)}")
        
        # Special case for sklearn
        if 'sklearn' in missing:
            missing.remove('sklearn')
            missing.append('scikit-learn')
            print(f"Note: Use 'scikit-learn' instead of 'sklearn'")
            print(f"pip install {' '.join(missing)}")
    
    return available, missing

def create_installation_guide():
    """Create a simple installation guide"""
    guide = """
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
data_path = r"C:\\Users\\dasar\\OneDrive\\Desktop\\ds project\\major\\creditcard.csv"
```

## Step 4: Run the Project
- For Jupyter Notebook: `jupyter notebook credit_card_fraud_detection.ipynb`
- For Python Script: `python fraud_detection_script.py`

## Troubleshooting
- If pip install fails, try: `pip install --user [package_name]`
- For permission errors, run command prompt as administrator
- If download is slow, try: `pip install --timeout 1000 [package_name]`
"""
    
    with open('INSTALLATION_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print("📝 Created INSTALLATION_GUIDE.md")

def main():
    """Main function"""
    print("🚀 CREDIT CARD FRAUD DETECTION - SETUP CHECK")
    print("="*60)
    
    # Step 1: Check if dataset file exists
    file_path = check_file_exists()
    
    # Step 2: Basic CSV analysis
    if file_path:
        success = basic_csv_analysis(file_path)
        if success:
            print("\n✅ Dataset is accessible and appears to be valid!")
    
    # Step 3: Check available libraries
    available, missing = check_python_libraries()
    
    # Step 4: Create installation guide
    create_installation_guide()
    
    # Final recommendations
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    
    if file_path and not missing:
        print("🎉 Everything looks good! You can run the full project now.")
        print("   Try: python fraud_detection_script.py")
    elif file_path and missing:
        print("📦 Dataset found, but some libraries are missing.")
        print("   Install missing libraries and try again.")
    elif not file_path and not missing:
        print("📁 Libraries are ready, but dataset file not found.")
        print("   Update the file path and try again.")
    else:
        print("🔧 Both dataset and libraries need attention.")
        print("   1. Fix the dataset file path")
        print("   2. Install missing libraries")
    
    print("\n📖 Check INSTALLATION_GUIDE.md for detailed instructions.")

if __name__ == "__main__":
    main()
