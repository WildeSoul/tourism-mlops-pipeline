"""
Model Quality Gate Script
==========================
Downloads model comparison metrics from Hugging Face and validates
that the best model meets minimum quality thresholds before deployment.

Quality Thresholds:
  - F1 Score >= 0.70
  - AUC-ROC >= 0.75

Exits with code 1 if thresholds are not met (blocks deployment).
"""

import os
import sys
import json
from huggingface_hub import HfApi, hf_hub_download

# =========================================================================
# Quality Thresholds
# =========================================================================
MIN_F1_SCORE = 0.70
MIN_AUC_ROC = 0.75


def main():
    # Get HF token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")

    # Get username dynamically
    api = HfApi()
    user_info = api.whoami(token=token)
    hf_username = user_info["name"]
    model_repo = f"{hf_username}/tourism-model"

    print("=" * 60)
    print("🚦 MODEL QUALITY GATE")
    print("=" * 60)

    # Download model comparison JSON
    print("\n📥 Downloading model metrics from Hugging Face Hub...")
    try:
        metrics_path = hf_hub_download(
            repo_id=model_repo,
            filename="model_comparison.json",
            token=token
        )
    except Exception as e:
        print(f"   ❌ Failed to download model_comparison.json: {e}")
        print(f"   Attempting to download feature_info.json as fallback...")
        metrics_path = hf_hub_download(
            repo_id=model_repo,
            filename="feature_info.json",
            token=token
        )

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    # Handle both formats: model_comparison.json or feature_info.json
    if 'model_results' in data:
        model_results = data['model_results']
        best_model_name = data.get('best_model', 'Unknown')
    else:
        # Fallback: feature_info.json format
        best_model_name = data.get('best_model_name', 'Unknown')
        best_f1 = data.get('best_f1_score', 0)
        model_results = {
            best_model_name: {
                'f1_score': best_f1,
                'auc_roc': data.get('best_auc_roc', 0)
            }
        }

    # Display all model results
    print(f"\n📊 Model Comparison Results:")
    print(f"{'Model':<25} {'F1 Score':<12} {'AUC-ROC':<12} {'Status'}")
    print("-" * 65)

    for model_name, metrics in model_results.items():
        f1 = metrics.get('f1_score', 0)
        auc = metrics.get('auc_roc', 0)
        f1_status = "✅" if f1 >= MIN_F1_SCORE else "❌"
        auc_status = "✅" if auc >= MIN_AUC_ROC else "❌"
        is_best = " ⭐" if model_name == best_model_name else ""
        print(f"{model_name + is_best:<25} {f1:.4f} {f1_status}    {auc:.4f} {auc_status}")

    # Validate best model
    print(f"\n{'=' * 60}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"{'=' * 60}")

    best_metrics = model_results.get(best_model_name, {})
    best_f1 = best_metrics.get('f1_score', 0)
    best_auc = best_metrics.get('auc_roc', 0)

    gate_passed = True

    # F1 Score Check
    if best_f1 >= MIN_F1_SCORE:
        print(f"   ✅ F1 Score:  {best_f1:.4f} >= {MIN_F1_SCORE} (PASS)")
    else:
        print(f"   ❌ F1 Score:  {best_f1:.4f} < {MIN_F1_SCORE} (FAIL)")
        gate_passed = False

    # AUC-ROC Check
    if best_auc >= MIN_AUC_ROC:
        print(f"   ✅ AUC-ROC:   {best_auc:.4f} >= {MIN_AUC_ROC} (PASS)")
    else:
        print(f"   ❌ AUC-ROC:   {best_auc:.4f} < {MIN_AUC_ROC} (FAIL)")
        gate_passed = False

    # Final verdict
    if gate_passed:
        print(f"\n🎉 QUALITY GATE PASSED — Model is approved for deployment!")
        sys.exit(0)
    else:
        print(f"\n💥 QUALITY GATE FAILED — Model does NOT meet minimum thresholds!")
        print(f"   Deployment is BLOCKED. Please retrain with better hyperparameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
