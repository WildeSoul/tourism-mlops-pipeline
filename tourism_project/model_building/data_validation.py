"""
Data Validation Script
=======================
Validates the integrity of the prepared dataset before model training.
Performs:
  1. Schema validation (expected columns exist)
  2. Null value checks on critical columns
  3. Target class balance validation
  4. Row count sanity check
  5. Feature type verification

Exits with code 1 if any critical validation fails.
"""

import os
import sys
import json
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


# =========================================================================
# Validation Configuration
# =========================================================================
EXPECTED_COLUMNS = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
    'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
    'Designation', 'MonthlyIncome', 'ProdTaken',
    # Engineered features
    'IncomePerPerson', 'TotalVisitors', 'HighDesignation',
    'IsFrequentTraveler', 'HasChildren'
]

CRITICAL_NON_NULL_COLUMNS = [
    'Age', 'Gender', 'ProdTaken', 'MonthlyIncome', 'Occupation'
]

MIN_ROW_COUNT = 100
MAX_CLASS_IMBALANCE_RATIO = 10.0  # Max ratio of majority to minority class


def validate_schema(df, dataset_name):
    """Check that all expected columns exist."""
    print(f"\n   🔍 Schema Validation ({dataset_name}):")
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing_cols:
        print(f"      ❌ FAIL: Missing columns: {missing_cols}")
        return False
    else:
        print(f"      ✅ PASS: All {len(EXPECTED_COLUMNS)} expected columns present")

    if extra_cols:
        print(f"      ℹ️  INFO: Extra columns found (ok): {extra_cols}")

    return True


def validate_nulls(df, dataset_name):
    """Check that critical columns have no null values."""
    print(f"\n   🔍 Null Value Check ({dataset_name}):")
    has_critical_nulls = False

    for col in CRITICAL_NON_NULL_COLUMNS:
        if col not in df.columns:
            continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"      ❌ FAIL: {col} has {null_count} null values")
            has_critical_nulls = True

    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        print(f"      ✅ PASS: No null values in any column")
    elif not has_critical_nulls:
        print(f"      ⚠️  WARN: {total_nulls} total null values found (non-critical columns)")

    return not has_critical_nulls


def validate_class_balance(df, dataset_name):
    """Check target variable class balance."""
    print(f"\n   🔍 Class Balance Check ({dataset_name}):")
    value_counts = df['ProdTaken'].value_counts()
    majority = value_counts.max()
    minority = value_counts.min()
    ratio = majority / minority if minority > 0 else float('inf')

    print(f"      Class 0 (Not Purchased): {value_counts.get(0, 0)}")
    print(f"      Class 1 (Purchased):     {value_counts.get(1, 0)}")
    print(f"      Imbalance Ratio: {ratio:.2f}:1")

    if ratio > MAX_CLASS_IMBALANCE_RATIO:
        print(f"      ❌ FAIL: Imbalance ratio {ratio:.2f} exceeds threshold {MAX_CLASS_IMBALANCE_RATIO}")
        return False
    else:
        print(f"      ✅ PASS: Imbalance ratio within acceptable range")
        return True


def validate_row_count(df, dataset_name):
    """Check that dataset has sufficient rows."""
    print(f"\n   🔍 Row Count Check ({dataset_name}):")
    row_count = len(df)

    if row_count < MIN_ROW_COUNT:
        print(f"      ❌ FAIL: Only {row_count} rows (minimum: {MIN_ROW_COUNT})")
        return False
    else:
        print(f"      ✅ PASS: {row_count} rows (minimum: {MIN_ROW_COUNT})")
        return True


def validate_feature_types(df, dataset_name):
    """Verify feature data types are consistent."""
    print(f"\n   🔍 Feature Type Check ({dataset_name}):")
    expected_numeric = ['Age', 'CityTier', 'MonthlyIncome', 'ProdTaken',
                        'NumberOfPersonVisiting', 'Passport', 'OwnCar']
    expected_categorical = ['TypeofContact', 'Gender', 'Occupation',
                            'ProductPitched', 'MaritalStatus', 'Designation']
    issues = []

    for col in expected_numeric:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col} should be numeric but is {df[col].dtype}")

    for col in expected_categorical:
        if col in df.columns and not pd.api.types.is_object_dtype(df[col]):
            issues.append(f"{col} should be object/string but is {df[col].dtype}")

    if issues:
        for issue in issues:
            print(f"      ⚠️  WARN: {issue}")
    else:
        print(f"      ✅ PASS: All feature types are as expected")

    return len(issues) == 0


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

    print("=" * 60)
    print("🛡️  DATA VALIDATION PIPELINE")
    print("=" * 60)

    # Download train and test datasets
    print("\n📥 Downloading datasets from Hugging Face Hub...")
    train_path = hf_hub_download(
        repo_id=repo_id, filename="train.csv",
        repo_type="dataset", token=token
    )
    test_path = hf_hub_download(
        repo_id=repo_id, filename="test.csv",
        repo_type="dataset", token=token
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

    # Run all validations
    all_passed = True
    validation_results = {}

    for name, df in [("Train", train_df), ("Test", test_df)]:
        results = {
            'schema': validate_schema(df, name),
            'nulls': validate_nulls(df, name),
            'class_balance': validate_class_balance(df, name),
            'row_count': validate_row_count(df, name),
            'feature_types': validate_feature_types(df, name)
        }
        validation_results[name] = results
        if not all(results.values()):
            all_passed = False

    # Summary
    print(f"\n{'=' * 60}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    for name, results in validation_results.items():
        passed = sum(results.values())
        total = len(results)
        status = "✅ ALL PASSED" if passed == total else f"❌ {total - passed} FAILED"
        print(f"   {name}: {passed}/{total} checks passed — {status}")

    if all_passed:
        print(f"\n🎉 All data validation checks PASSED!")
        print(f"   Data is ready for model training.")
        sys.exit(0)
    else:
        print(f"\n💥 Some validation checks FAILED!")
        print(f"   Please review the data before proceeding to model training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
