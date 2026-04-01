"""
Model Training Script with Advanced Experimentation Tracking
==============================================================
Loads train/test data from Hugging Face, builds 6 ML models with:
  1. SMOTE for class imbalance handling
  2. 6 models: DecisionTree, RandomForest, GradientBoosting, XGBoost, AdaBoost, LightGBM
  3. StratifiedKFold cross-validation
  4. Extended hyperparameter search
  5. SHAP explainability (summary plot + feature importance)
  6. Confusion matrix & ROC curve visualizations
  7. Model versioning with full comparison JSON
  8. MLflow enhanced logging with artifacts

Models used: Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost, LightGBM
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import HfApi, hf_hub_download

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

# Set plot aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150


def generate_confusion_matrix_plot(y_true, y_pred, model_name, output_dir):
    """Generate and save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Purchased', 'Purchased'],
                yticklabels=['Not Purchased', 'Purchased'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')

    # Add accuracy text
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(1.5, -0.15, f'Accuracy: {acc:.1%}', ha='center', fontsize=12,
            transform=ax.transAxes, fontweight='bold', color='#1a5276')

    path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_roc_curve_plot(y_true, y_proba, model_name, output_dir):
    """Generate and save a ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2980b9', linewidth=2.5,
            label=f'{model_name} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Baseline')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2980b9')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'roc_curve.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_multi_roc_plot(results_with_proba, y_test, output_dir):
    """Generate ROC curves for all models in one plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_with_proba)))

    for (name, y_proba), color in zip(results_with_proba.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — All Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'roc_curves_comparison.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_shap_plots(model, X_test_transformed, feature_names, output_dir):
    """Generate SHAP summary and importance plots for the best model."""
    import shap

    print("\n🔍 Generating SHAP Explainability Plots...")

    try:
        # Get the classifier from the pipeline
        classifier = model.named_steps['classifier']

        # Use appropriate SHAP explainer based on model type
        if hasattr(classifier, 'feature_importances_'):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_transformed)

            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1]  # Class 1 (positive)
            else:
                shap_values_plot = shap_values
        else:
            # Fallback for non-tree models
            explainer = shap.KernelExplainer(classifier.predict_proba, X_test_transformed[:100])
            shap_values = explainer.shap_values(X_test_transformed[:100])
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values

        # SHAP Summary Plot (beeswarm)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values_plot, X_test_transformed,
                          feature_names=feature_names, show=False, max_display=20)
        plt.title('SHAP Feature Impact Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        summary_path = os.path.join(output_dir, 'shap_summary_plot.png')
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
        print("   📊 Saved: SHAP Summary Plot")

        # SHAP Feature Importance Bar
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_plot, X_test_transformed,
                          feature_names=feature_names, plot_type='bar',
                          show=False, max_display=20)
        plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        importance_path = os.path.join(output_dir, 'shap_feature_importance.png')
        plt.savefig(importance_path, bbox_inches='tight')
        plt.close()
        print("   📊 Saved: SHAP Feature Importance Bar")

        return summary_path, importance_path, True

    except Exception as e:
        print(f"   ⚠️ SHAP generation failed: {e}")
        print("   Skipping SHAP plots (non-critical)")
        return None, None, False


def main():
    # Get HF token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")

    # Get username dynamically
    api = HfApi()
    user_info = api.whoami(token=token)
    hf_username = user_info["name"]
    dataset_repo = f"{hf_username}/tourism-dataset"
    model_repo = f"{hf_username}/tourism-model"

    # =========================================================================
    # Step 1: Load train and test data from Hugging Face
    # =========================================================================
    print("📥 Loading train and test data from Hugging Face Hub...")
    train_path = hf_hub_download(
        repo_id=dataset_repo, filename="train.csv",
        repo_type="dataset", token=token
    )
    test_path = hf_hub_download(
        repo_id=dataset_repo, filename="test.csv",
        repo_type="dataset", token=token
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"   Training data: {train_df.shape}")
    print(f"   Testing data:  {test_df.shape}")

    # Separate features and target
    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"\n   Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"   Numerical features ({len(numerical_features)}):   {numerical_features}")

    # =========================================================================
    # Step 2: Define preprocessing pipeline
    # =========================================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # =========================================================================
    # Step 3: Apply SMOTE for class imbalance
    # =========================================================================
    print("\n⚖️ Applying SMOTE for class imbalance handling...")
    print(f"   Before SMOTE: {dict(y_train.value_counts())}")

    # Preprocess first, then apply SMOTE
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    print(f"   After SMOTE:  {dict(pd.Series(y_train_resampled).value_counts())}")

    # Get feature names after preprocessing
    ohe_feature_names = []
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat']
        ohe_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numerical_features + ohe_feature_names

    # =========================================================================
    # Step 4: Define 6 models and hyperparameters
    # =========================================================================
    # StratifiedKFold for robust cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_config = {
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42, algorithm='SAMME'),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 1.0]
            }
        }
    }

    # =========================================================================
    # Step 5: Train and evaluate models with MLflow tracking
    # =========================================================================
    print("\n🧪 Starting model training with MLflow experimentation tracking...\n")

    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Tourism_Package_Prediction")

    output_dir = "tourism_project/model_building/plots"
    os.makedirs(output_dir, exist_ok=True)

    best_model_pipeline = None
    best_score = 0
    best_model_name = ""
    best_y_proba = None
    results = {}
    results_with_proba = {}
    all_models = {}

    for name, config in models_config.items():
        print(f"{'='*60}")
        print(f"🔄 Training: {name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=name):
            # Hyperparameter tuning directly on resampled data (no pipeline needed
            # since preprocessing already done)
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                cv=cv_strategy,
                scoring='f1',
                n_iter=min(20, max(10, len(config['params'])**2)),
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            search.fit(X_train_resampled, y_train_resampled)
            tuned_model = search.best_estimator_

            # Log all tuned parameters
            best_params = search.best_params_
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", name)
            mlflow.log_param("smote_applied", True)
            mlflow.log_param("cv_folds", 5)
            print(f"\n   Best Parameters: {best_params}")

            # Predictions on test set
            y_pred = tuned_model.predict(X_test_preprocessed)
            y_pred_proba = tuned_model.predict_proba(X_test_preprocessed)[:, 1]

            # Evaluate model performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)

            # Log metrics
            metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'cv_best_score': search.best_score_
            }
            mlflow.log_metrics(metrics_dict)

            # Log model
            mlflow.sklearn.log_model(tuned_model, name)

            # Store results
            results[name] = metrics_dict
            results_with_proba[name] = y_pred_proba
            all_models[name] = tuned_model

            print(f"\n   📊 Performance Metrics:")
            print(f"   Accuracy:       {accuracy:.4f}")
            print(f"   Precision:      {precision:.4f}")
            print(f"   Recall:         {recall:.4f}")
            print(f"   F1 Score:       {f1:.4f}")
            print(f"   AUC-ROC:        {auc_roc:.4f}")
            print(f"   CV Best Score:  {search.best_score_:.4f}")
            print(f"\n   Classification Report:")
            print(classification_report(y_test, y_pred))

            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                best_y_proba = y_pred_proba

    # =========================================================================
    # Step 6: Summary of all models
    # =========================================================================
    print(f"\n{'='*60}")
    print("📋 MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('f1_score', ascending=False)
    print(results_df.to_string())
    print(f"\n🏆 Best Model: {best_model_name} (F1 Score: {best_score:.4f})")

    # =========================================================================
    # Step 7: Generate Visualizations for Best Model
    # =========================================================================
    print(f"\n📊 Generating visualizations for best model ({best_model_name})...")
    best_y_pred = all_models[best_model_name].predict(X_test_preprocessed)

    # Confusion Matrix
    cm_path = generate_confusion_matrix_plot(y_test, best_y_pred, best_model_name, output_dir)
    print(f"   📊 Saved: Confusion Matrix")

    # ROC Curve (best model)
    roc_path = generate_roc_curve_plot(y_test, best_y_proba, best_model_name, output_dir)
    print(f"   📊 Saved: ROC Curve")

    # Multi-model ROC Comparison
    multi_roc_path = generate_multi_roc_plot(results_with_proba, y_test, output_dir)
    print(f"   📊 Saved: Multi-model ROC Comparison")

    # Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ['f1_score', 'auc_roc', 'accuracy']
    titles = ['F1 Score', 'AUC-ROC', 'Accuracy']
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        values = results_df[metric].values
        names = results_df.index
        bars = ax.barh(names, values, color=colors, edgecolor='white')
        ax.set_xlabel(title, fontsize=12)
        ax.set_title(f'Model Comparison — {title}', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                    va='center', fontsize=10)
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'model_comparison_chart.png')
    fig.savefig(comparison_path, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved: Model Comparison Chart")

    # =========================================================================
    # Step 8: SHAP Explainability
    # =========================================================================
    shap_summary_path, shap_importance_path, shap_success = generate_shap_plots(
        # We need to wrap the best model back in a pipeline-like structure
        # Since we applied SMOTE separately, the classifier IS the best model
        type('Pipeline', (), {
            'named_steps': {'classifier': all_models[best_model_name]}
        })(),
        X_test_preprocessed,
        all_feature_names,
        output_dir
    )

    # =========================================================================
    # Step 9: Create full pipeline for deployment (preprocessor + best model)
    # =========================================================================
    print(f"\n💾 Saving best model ({best_model_name}) as full pipeline...")
    os.makedirs("tourism_project/model_building", exist_ok=True)

    # Create a proper pipeline for deployment
    # Re-fit the best model's hyperparameters in a pipeline for clean deployment
    best_config = models_config[best_model_name]
    best_params_clean = {k.replace('classifier__', ''): v
                         for k, v in search.best_params_.items()} if 'classifier__' in str(search.best_params_) else search.best_params_

    # Clone the best model with its tuned params
    best_model_for_deploy = all_models[best_model_name]

    # Create deployment pipeline
    deploy_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model_for_deploy)
    ])

    # The preprocessor is already fit — we need to refit the full pipeline
    deploy_pipeline.fit(X_train, y_train)

    model_path = "tourism_project/model_building/best_model.joblib"
    joblib.dump(deploy_pipeline, model_path)

    # Save feature names for the Streamlit app
    feature_info = {
        'feature_names': list(X_train.columns),
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'best_model_name': best_model_name,
        'best_f1_score': best_score,
        'best_auc_roc': results[best_model_name]['auc_roc'],
        'smote_applied': True,
        'total_models_trained': len(results),
        'engineered_features': ['IncomePerPerson', 'TotalVisitors', 'HighDesignation',
                                 'IsFrequentTraveler', 'HasChildren']
    }
    features_path = "tourism_project/model_building/feature_info.json"
    with open(features_path, 'w') as f:
        json.dump(feature_info, f, indent=2)

    # Save full model comparison results
    comparison_data = {
        'best_model': best_model_name,
        'best_f1_score': best_score,
        'model_results': results,
        'training_config': {
            'smote': True,
            'cv_strategy': 'StratifiedKFold(5)',
            'scoring': 'f1',
            'n_models': len(results)
        }
    }
    comparison_path_json = "tourism_project/model_building/model_comparison.json"
    with open(comparison_path_json, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    # =========================================================================
    # Step 10: Register everything on Hugging Face
    # =========================================================================
    print("\n📤 Registering best model and artifacts on Hugging Face...")
    api.create_repo(repo_id=model_repo, exist_ok=True, token=token)

    # Upload model and metadata
    api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib",
                    repo_id=model_repo, token=token)
    api.upload_file(path_or_fileobj=features_path, path_in_repo="feature_info.json",
                    repo_id=model_repo, token=token)
    api.upload_file(path_or_fileobj=comparison_path_json, path_in_repo="model_comparison.json",
                    repo_id=model_repo, token=token)

    # Upload plots
    plot_files = [cm_path, roc_path, multi_roc_path, comparison_path]
    if shap_success:
        if shap_summary_path:
            plot_files.append(shap_summary_path)
        if shap_importance_path:
            plot_files.append(shap_importance_path)

    for plot_path in plot_files:
        if plot_path and os.path.exists(plot_path):
            plot_name = os.path.basename(plot_path)
            api.upload_file(path_or_fileobj=plot_path, path_in_repo=f"plots/{plot_name}",
                            repo_id=model_repo, token=token)
            print(f"   📊 Uploaded: {plot_name}")

    print(f"\n✅ All artifacts registered at: https://huggingface.co/{model_repo}")
    print(f"   - Model pipeline (preprocessor + {best_model_name})")
    print(f"   - Feature info JSON")
    print(f"   - Model comparison JSON")
    print(f"   - {len(plot_files)} visualization plots")


if __name__ == "__main__":
    main()
