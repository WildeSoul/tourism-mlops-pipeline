"""
Model Training Script with Experimentation Tracking
====================================================
Loads train/test data from Hugging Face, builds multiple ML models with
hyperparameter tuning, logs experiments with MLflow, evaluates performance,
and registers the best model on Hugging Face Model Hub.

Models used: Decision Tree, Random Forest, Gradient Boosting, XGBoost
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')


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
    print(f"\n   Categorical features: {categorical_features}")
    print(f"   Numerical features:   {numerical_features}")

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
    # Step 3: Define models and hyperparameters
    # =========================================================================
    models_config = {
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [3, 5, 10, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__subsample': [0.8, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
        }
    }

    # =========================================================================
    # Step 4: Train and evaluate models with MLflow tracking
    # =========================================================================
    print("\n🧪 Starting model training with MLflow experimentation tracking...\n")

    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Tourism_Package_Prediction")

    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}

    for name, config in models_config.items():
        print(f"{'='*60}")
        print(f"🔄 Training: {name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=name):
            # Create pipeline with preprocessor and classifier
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', config['model'])
            ])

            # Hyperparameter tuning with RandomizedSearchCV
            search = RandomizedSearchCV(
                pipeline,
                config['params'],
                cv=5,
                scoring='f1',
                n_iter=min(12, len(config['params'])**2),
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            search.fit(X_train, y_train)
            tuned_model = search.best_estimator_

            # Log all tuned parameters
            best_params = search.best_params_
            mlflow.log_params(best_params)
            print(f"\n   Best Parameters: {best_params}")

            # Predictions
            y_pred = tuned_model.predict(X_test)
            y_pred_proba = tuned_model.predict_proba(X_test)[:, 1]

            # Evaluate model performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc_roc)

            # Log model
            mlflow.sklearn.log_model(tuned_model, name)

            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc
            }

            print(f"\n   📊 Performance Metrics:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   AUC-ROC:   {auc_roc:.4f}")
            print(f"\n   Classification Report:")
            print(classification_report(y_test, y_pred))

            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = tuned_model
                best_model_name = name

    # =========================================================================
    # Step 5: Summary of all models
    # =========================================================================
    print(f"\n{'='*60}")
    print("📋 MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    print(f"\n🏆 Best Model: {best_model_name} (F1 Score: {best_score:.4f})")

    # =========================================================================
    # Step 6: Save and register best model on Hugging Face Model Hub
    # =========================================================================
    print(f"\n💾 Saving best model ({best_model_name})...")
    os.makedirs("tourism_project/model_building", exist_ok=True)

    # Save model pipeline (includes preprocessor)
    model_path = "tourism_project/model_building/best_model.joblib"
    joblib.dump(best_model, model_path)

    # Save feature names for the Streamlit app
    feature_info = {
        'feature_names': list(X_train.columns),
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'best_model_name': best_model_name,
        'best_f1_score': best_score
    }
    features_path = "tourism_project/model_building/feature_info.json"
    with open(features_path, 'w') as f:
        json.dump(feature_info, f, indent=2)

    print("📤 Registering best model on Hugging Face Model Hub...")
    api.create_repo(repo_id=model_repo, exist_ok=True, token=token)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.joblib",
        repo_id=model_repo,
        token=token
    )
    api.upload_file(
        path_or_fileobj=features_path,
        path_in_repo="feature_info.json",
        repo_id=model_repo,
        token=token
    )

    print(f"✅ Best model registered at: https://huggingface.co/{model_repo}")

if __name__ == "__main__":
    main()
