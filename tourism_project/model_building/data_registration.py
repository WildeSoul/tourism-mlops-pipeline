"""
Data Registration Script
========================
Registers the tourism dataset on Hugging Face Dataset Hub.
This script uploads the raw tourism.csv file to a Hugging Face dataset repository.
"""

import os
from huggingface_hub import HfApi

def main():
    # Get HF token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")

    # Get username dynamically from token
    api = HfApi()
    user_info = api.whoami(token=token)
    hf_username = user_info["name"]

    # Define repository ID
    repo_id = f"{hf_username}/tourism-dataset"

    # Create dataset repository on Hugging Face Hub
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token
    )
    print(f"Dataset repository created/verified: {repo_id}")

    # Upload the raw tourism.csv file
    csv_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print(f"✅ Dataset registered successfully at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
