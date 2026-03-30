# Tourism MLOps Pipeline

End-to-end MLOps pipeline for predicting customer purchases of the Wellness Tourism Package.

## Project Structure

```
├── .github/workflows/pipeline.yml    # GitHub Actions CI/CD pipeline
├── tourism_project/
│   ├── data/tourism.csv              # Raw dataset
│   ├── model_building/
│   │   ├── data_registration.py      # Registers data on HF Hub
│   │   ├── data_preparation.py       # Cleans, splits, uploads to HF
│   │   └── model_training.py         # Trains models with MLflow tracking
│   └── deployment/
│       ├── Dockerfile                # Docker config for HF Spaces
│       ├── app.py                    # Streamlit prediction app
│       └── requirements.txt          # Deployment dependencies
├── hosting.py                        # Pushes to HF Space
└── requirements.txt                  # Root dependencies
```

## Pipeline Jobs
1. **register-dataset** - Uploads raw data to Hugging Face Hub
2. **data-prep** - Cleans data, splits train/test, uploads to HF
3. **model-training** - Trains Decision Tree, Random Forest, Gradient Boosting, XGBoost with MLflow
4. **deploy-hosting** - Deploys Streamlit app to HF Spaces

## Links
- **Streamlit App**: https://huggingface.co/spaces/WILDESOUL/wellness-tourism-app
- **Dataset**: https://huggingface.co/datasets/WILDESOUL/tourism-dataset
- **Model**: https://huggingface.co/WILDESOUL/tourism-model
