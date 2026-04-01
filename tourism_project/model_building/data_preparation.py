"""
Data Preparation Script (Advanced)
====================================
Loads dataset from Hugging Face, performs:
  1. Exploratory Data Analysis (EDA) with publication-quality visualizations
  2. Data Cleaning (missing values, typos)
  3. Outlier Detection & IQR-based Capping
  4. Feature Engineering (5 new derived features)
  5. Train/Test Split with stratification
  6. Uploads processed data + EDA artifacts to Hugging Face
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# Set plot aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def generate_eda_visualizations(df, output_dir):
    """Generate and save publication-quality EDA visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []

    # ---- 1. Target Distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['ProdTaken'].value_counts()
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(counts.index.map({0: 'Not Purchased', 1: 'Purchased'}),
                  counts.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=12)
    ax.set_title('Target Variable Distribution (ProdTaken)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = os.path.join(output_dir, 'eda_target_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    chart_paths.append(path)
    print(f"   📊 Saved: Target Distribution")

    # ---- 2. Correlation Heatmap ----
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    path = os.path.join(output_dir, 'eda_correlation_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    chart_paths.append(path)
    print(f"   📊 Saved: Correlation Heatmap")

    # ---- 3. Numerical Feature Distributions ----
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('ProdTaken', errors='ignore')
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue='ProdTaken', kde=True, ax=axes[i],
                     palette=['#e74c3c', '#2ecc71'], alpha=0.6)
        axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[i].legend(title='ProdTaken', labels=['No', 'Yes'], fontsize=8)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Numerical Feature Distributions by Target', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, 'eda_numerical_distributions.png')
    fig.savefig(path)
    plt.close(fig)
    chart_paths.append(path)
    print(f"   📊 Saved: Numerical Distributions")

    # ---- 4. Boxplots for Outlier Detection ----
    outlier_cols = ['Age', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfTrips']
    outlier_cols = [c for c in outlier_cols if c in df.columns]
    fig, axes = plt.subplots(1, len(outlier_cols), figsize=(5*len(outlier_cols), 5))
    if len(outlier_cols) == 1:
        axes = [axes]
    for i, col in enumerate(outlier_cols):
        sns.boxplot(data=df, y=col, x='ProdTaken', hue='ProdTaken', ax=axes[i],
                    palette=['#e74c3c', '#2ecc71'], legend=False)
        axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[i].set_xticklabels(['Not Purchased', 'Purchased'])
    fig.suptitle('Outlier Analysis — Key Numerical Features', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, 'eda_outlier_boxplots.png')
    fig.savefig(path)
    plt.close(fig)
    chart_paths.append(path)
    print(f"   📊 Saved: Outlier Boxplots")

    # ---- 5. Categorical Feature Analysis ----
    cat_cols = df.select_dtypes(include=['object']).columns
    n_cols_cat = 3
    n_rows_cat = (len(cat_cols) + n_cols_cat - 1) // n_cols_cat
    fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(6*n_cols_cat, 5*n_rows_cat))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df['ProdTaken'], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=axes[i],
                color=['#e74c3c', '#2ecc71'], edgecolor='white')
        axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Percentage %')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].legend(title='ProdTaken', labels=['No', 'Yes'], fontsize=8)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Categorical Features vs Target (Purchase Rate)', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, 'eda_categorical_analysis.png')
    fig.savefig(path)
    plt.close(fig)
    chart_paths.append(path)
    print(f"   📊 Saved: Categorical Analysis")

    # ---- 6. Feature Importance (Mutual Information) ----
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    X_mi = df.drop('ProdTaken', axis=1).copy()
    for col in X_mi.select_dtypes(include=['object']).columns:
        X_mi[col] = LabelEncoder().fit_transform(X_mi[col].astype(str))
    X_mi = X_mi.fillna(0)

    mi_scores = mutual_info_classif(X_mi, df['ProdTaken'], random_state=42)
    mi_df = pd.DataFrame({'Feature': X_mi.columns, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(mi_df)))
    ax.barh(mi_df['Feature'], mi_df['MI_Score'], color=colors, edgecolor='white')
    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_title('Feature Importance (Mutual Information)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = os.path.join(output_dir, 'eda_feature_importance_mi.png')
    fig.savefig(path)
    plt.close(fig)
    chart_paths.append(path)
    print(f"   📊 Saved: Feature Importance (MI)")

    return chart_paths


def cap_outliers_iqr(df, columns, factor=1.5):
    """Cap outliers using IQR method."""
    capping_report = {}
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_lower = (df[col] < lower).sum()
        n_upper = (df[col] > upper).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        capping_report[col] = {
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2),
            'capped_below': int(n_lower),
            'capped_above': int(n_upper)
        }
        if n_lower + n_upper > 0:
            print(f"   Capped {col}: {n_lower} below ({lower:.1f}), {n_upper} above ({upper:.1f})")
    return df, capping_report


def engineer_features(df):
    """Create advanced derived features."""
    print("\n🔧 Engineering new features...")

    # 1. Income per person visiting
    df['IncomePerPerson'] = df['MonthlyIncome'] / df['NumberOfPersonVisiting'].clip(lower=1)
    print("   + IncomePerPerson = MonthlyIncome / NumberOfPersonVisiting")

    # 2. Total visitors (adults + children)
    df['TotalVisitors'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    print("   + TotalVisitors = NumberOfPersonVisiting + NumberOfChildrenVisiting")

    # 3. High designation flag (AVP or VP)
    df['HighDesignation'] = df['Designation'].isin(['AVP', 'VP']).astype(int)
    print("   + HighDesignation = 1 if Designation in (AVP, VP)")

    # 4. Frequent traveler flag
    df['IsFrequentTraveler'] = (df['NumberOfTrips'] > 3).astype(int)
    print("   + IsFrequentTraveler = 1 if NumberOfTrips > 3")

    # 5. Has children flag
    df['HasChildren'] = (df['NumberOfChildrenVisiting'] > 0).astype(int)
    print("   + HasChildren = 1 if NumberOfChildrenVisiting > 0")

    print(f"   Total features after engineering: {df.shape[1]}")
    return df


def generate_data_profile(df, output_path):
    """Generate a comprehensive data profile report as JSON."""
    profile = {
        'dataset_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
        'class_distribution': {
            'class_0_count': int((df['ProdTaken'] == 0).sum()),
            'class_1_count': int((df['ProdTaken'] == 1).sum()),
            'class_balance_ratio': round((df['ProdTaken'] == 1).sum() / (df['ProdTaken'] == 0).sum(), 4)
        },
        'feature_types': {
            'numerical': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': df.select_dtypes(include=['object']).columns.tolist(),
            'num_numerical': len(df.select_dtypes(include=[np.number]).columns),
            'num_categorical': len(df.select_dtypes(include=['object']).columns)
        },
        'descriptive_statistics': json.loads(df.describe().to_json()),
        'missing_values': json.loads(df.isnull().sum().to_json())
    }
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\n📄 Data profile saved to: {output_path}")
    return profile


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
    # Step 2: Exploratory Data Analysis (EDA)
    # =========================================================================
    print("\n📊 Generating EDA Visualizations...")
    eda_dir = "tourism_project/data/eda_charts"

    # Clean first for EDA (minimal cleaning for accurate visualizations)
    df_eda = df.copy()
    cols_to_drop = ['CustomerID']
    for col in df_eda.columns:
        if 'Unnamed' in str(col):
            cols_to_drop.append(col)
    df_eda = df_eda.drop(columns=cols_to_drop, errors='ignore')
    if 'Gender' in df_eda.columns:
        df_eda['Gender'] = df_eda['Gender'].replace('Fe Male', 'Female')
    # Fill NaNs for EDA plotting
    for col in df_eda.select_dtypes(include=['float64', 'int64']).columns:
        df_eda[col] = df_eda[col].fillna(df_eda[col].median())
    for col in df_eda.select_dtypes(include=['object']).columns:
        df_eda[col] = df_eda[col].fillna(df_eda[col].mode()[0])

    chart_paths = generate_eda_visualizations(df_eda, eda_dir)

    # =========================================================================
    # Step 3: Data Cleaning
    # =========================================================================
    print("\n🧹 Performing data cleaning...")

    # 3a. Remove unnecessary columns
    cols_to_drop = ['CustomerID']
    for col in df.columns:
        if 'Unnamed' in str(col):
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"   Dropped columns: {cols_to_drop}")

    # 3b. Fix Gender typo
    if 'Gender' in df.columns:
        fe_male_count = (df['Gender'] == 'Fe Male').sum()
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        print(f"   Fixed 'Fe Male' → 'Female' in Gender column ({fe_male_count} entries)")

    # 3c. Handle missing values
    print("\n   Missing values before cleaning:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"     - {col}: {missing[col]} missing values")

    # Fill numerical with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Filled {col} NaN with median: {median_val}")

    # Fill categorical with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   Filled {col} NaN with mode: {mode_val}")

    print(f"\n   Missing values after cleaning: {df.isnull().sum().sum()}")

    # =========================================================================
    # Step 4: Outlier Detection & Capping (IQR method)
    # =========================================================================
    print("\n📐 Capping outliers using IQR method...")
    outlier_columns = ['Age', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfTrips', 'NumberOfFollowups']
    outlier_columns = [c for c in outlier_columns if c in df.columns]
    df, capping_report = cap_outliers_iqr(df, outlier_columns)

    # =========================================================================
    # Step 5: Feature Engineering
    # =========================================================================
    df = engineer_features(df)
    print(f"\n   Final dataset shape: {df.shape}")

    # =========================================================================
    # Step 6: Generate Data Profile Report
    # =========================================================================
    os.makedirs("tourism_project/data", exist_ok=True)
    profile = generate_data_profile(df, "tourism_project/data/data_profile.json")

    # =========================================================================
    # Step 7: Split into training and testing sets
    # =========================================================================
    print("\n✂️ Splitting dataset into train and test sets...")
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"   Training set: {train_df.shape[0]} samples ({train_df.shape[1]} features)")
    print(f"   Testing set:  {test_df.shape[0]} samples")
    print(f"   Target distribution (train): {dict(y_train.value_counts())}")
    print(f"   Target distribution (test):  {dict(y_test.value_counts())}")

    # =========================================================================
    # Step 8: Save locally
    # =========================================================================
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)
    print("\n💾 Saved train.csv and test.csv locally")

    # =========================================================================
    # Step 9: Upload everything to Hugging Face
    # =========================================================================
    print("\n📤 Uploading datasets and artifacts to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj="tourism_project/data/train.csv",
        path_in_repo="train.csv",
        repo_id=repo_id, repo_type="dataset", token=token
    )
    api.upload_file(
        path_or_fileobj="tourism_project/data/test.csv",
        path_in_repo="test.csv",
        repo_id=repo_id, repo_type="dataset", token=token
    )
    api.upload_file(
        path_or_fileobj="tourism_project/data/data_profile.json",
        path_in_repo="data_profile.json",
        repo_id=repo_id, repo_type="dataset", token=token
    )

    # Upload EDA charts
    for chart_path in chart_paths:
        chart_name = os.path.basename(chart_path)
        api.upload_file(
            path_or_fileobj=chart_path,
            path_in_repo=f"eda/{chart_name}",
            repo_id=repo_id, repo_type="dataset", token=token
        )
    print(f"   📊 Uploaded {len(chart_paths)} EDA charts to HF dataset repo")

    print(f"\n✅ Data preparation complete!")
    print(f"   Dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"   Features: {list(X.columns)}")
    print(f"   Engineered features: IncomePerPerson, TotalVisitors, HighDesignation, IsFrequentTraveler, HasChildren")


if __name__ == "__main__":
    main()
