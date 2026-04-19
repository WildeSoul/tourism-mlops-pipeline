# Tourism MLOps Pipeline (Advanced)

End-to-end MLOps pipeline for predicting customer purchases of the **Wellness Tourism Package** — featuring EDA visualizations, feature engineering, SMOTE, 6 ML models, SHAP explainability, model quality gates, and CI/CD automation.

## 🔬 Advanced Features

| Feature | Description |
|---------|-------------|
| **EDA Visualizations** | 6 publication-quality charts (correlation heatmap, distributions, boxplots, MI importance) |
| **Feature Engineering** | 5 derived features — IncomePerPerson, TotalVisitors, HighDesignation, IsFrequentTraveler, HasChildren |
| **Outlier Detection** | IQR-based capping on Age, MonthlyIncome, DurationOfPitch, NumberOfTrips |
| **SMOTE** | Synthetic Minority Oversampling Technique for class imbalance handling |
| **6 ML Models** | DecisionTree, RandomForest, GradientBoosting, XGBoost, AdaBoost, LightGBM |
| **SHAP Explainability** | Global feature importance + beeswarm summary plots |
| **Model Quality Gate** | Blocks deployment if F1 < 0.70 or AUC < 0.75 |
| **Data Validation** | Schema, null, class balance, and row count checks before training |
| **Batch Prediction** | CSV upload for bulk customer scoring |
| **MLflow Tracking** | Full experiment tracking with visual artifacts |

## Project Structure

```
├── .github/workflows/pipeline.yml    # GitHub Actions CI/CD (6 jobs)
├── tourism_project/
│   ├── data/
│   │   ├── tourism.csv               # Raw dataset
│   │   ├── train.csv                 # Training split
│   │   ├── test.csv                  # Testing split
│   │   ├── data_profile.json         # Dataset statistics report
│   │   └── eda_charts/               # EDA visualizations (6 charts)
│   ├── model_building/
│   │   ├── data_registration.py      # Registers data on HF Hub
│   │   ├── data_preparation.py       # EDA + Cleaning + Feature Engineering
│   │   ├── data_validation.py        # Schema & quality validation
│   │   ├── model_training.py         # 6 models + SMOTE + SHAP + MLflow
│   │   ├── model_quality_gate.py     # F1/AUC threshold enforcement
│   │   └── plots/                    # Model performance visualizations
│   └── deployment/
│       ├── Dockerfile                # Docker config (slim base + healthcheck)
│       ├── app.py                    # Multi-page Streamlit dashboard
│       └── requirements.txt          # Deployment dependencies
├── hosting.py                        # Pushes to HF Space
└── requirements.txt                  # Root dependencies (13 packages)
```

## Pipeline Jobs (6 stages)

```
register-dataset → data-prep → data-validation → model-training → model-quality-gate → deploy-hosting
```

1. **register-dataset** — Uploads raw data to Hugging Face Hub
2. **data-prep** — EDA visualizations, data cleaning, outlier capping, feature engineering, train/test split
3. **data-validation** — Schema validation, null checks, class balance verification
4. **model-training** — Trains 6 models with SMOTE, SHAP explainability, generates plots, registers best model
5. **model-quality-gate** — Asserts F1 ≥ 0.70 and AUC ≥ 0.75 (blocks deployment on failure)
6. **deploy-hosting** — Deploys multi-page Streamlit dashboard to HF Spaces

## Streamlit Dashboard (4 Pages)

1. **🔮 Single Prediction** — Customer form with gauge chart, risk scoring, downloadable report
2. **📦 Batch Prediction** — CSV upload for bulk predictions with summary statistics
3. **📊 Model Analytics** — Model comparison table, SHAP plots, confusion matrix, ROC curves
4. **ℹ️ About** — Architecture diagram, feature list, model information

## Pipeline Triggers

- ✅ **Push to main** — Automatic trigger on code push
- ✅ **Manual trigger** — `workflow_dispatch` for on-demand runs
- ✅ **Weekly retraining** — Cron schedule (Mondays at 2 AM UTC)

## Links

- **Streamlit App**: https://huggingface.co/spaces/WILDESOUL/wellness-tourism-app
- **Dataset**: https://huggingface.co/datasets/WILDESOUL/tourism-dataset
- **Model**: https://huggingface.co/WILDESOUL/tourism-model
  
