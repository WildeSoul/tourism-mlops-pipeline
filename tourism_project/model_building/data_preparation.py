"""
Data Preparation Script
=======================
Loads dataset from Hugging Face, performs data cleaning and preprocessing,
splits into train/test sets, saves locally and uploads back to Hugging Face.
"""

import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

def main():
    # Get HF token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")

    # Get username dynamically
    api = HfApi()
    user_info = api.whoami(token=token)
    hf_username = user_info["name"]
    repo_id = f"{hf_username}/tourism-dataset"

    # =========================================================================
    # Step 1: Load dataset from Hugging Face data space
    # =========================================================================
    print("📥 Loading dataset from Hugging Face Hub...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename="tourism.csv",
        repo_type="dataset",
        token=token
    )
    df = pd.read_csv(file_path)
    print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    # =========================================================================
    # Step 2: Data Cleaning
    # =========================================================================
    print("\n🧹 Performing data cleaning...")

    # 2a. Remove unnecessary columns (unnamed index column and CustomerID)
    cols_to_drop = ['CustomerID']
    # Check for unnamed index column
    for col in df.columns:
        if 'Unnamed' in str(col):
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"   Dropped columns: {cols_to_drop}")

    # 2b. Fix Gender column - "Fe Male" should be "Female"
    if 'Gender' in df.columns:
        fe_male_count = (df['Gender'] == 'Fe Male').sum()
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        print(f"   Fixed 'Fe Male' → 'Female' in Gender column ({fe_male_count} entries)")

    # 2c. Handle missing values
    print("\n   Missing values before cleaning:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"     - {col}: {missing[col]} missing values")

    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Filled {col} NaN with median: {median_val}")

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   Filled {col} NaN with mode: {mode_val}")

    print(f"\n   Missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"   Cleaned dataset shape: {df.shape}")

    # =========================================================================
    # Step 3: Split into training and testing sets
    # =========================================================================
    print("\n✂️ Splitting dataset into train and test sets...")
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"   Training set: {train_df.shape[0]} samples")
    print(f"   Testing set:  {test_df.shape[0]} samples")
    print(f"   Target distribution (train): {dict(y_train.value_counts())}")
    print(f"   Target distribution (test):  {dict(y_test.value_counts())}")

    # =========================================================================
    # Step 4: Save locally
    # =========================================================================
    os.makedirs("tourism_project/data", exist_ok=True)
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)
    print("\n💾 Saved train.csv and test.csv locally")

    # =========================================================================
    # Step 5: Upload train and test datasets to Hugging Face
    # =========================================================================
    print("\n📤 Uploading train and test datasets to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj="tourism_project/data/train.csv",
        path_in_repo="train.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    api.upload_file(
        path_or_fileobj="tourism_project/data/test.csv",
        path_in_repo="test.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print(f"✅ Train and test datasets uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
