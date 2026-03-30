"""
Hosting Script
==============
Pushes all deployment files (Dockerfile, app.py, requirements.txt)
to a Hugging Face Space for hosting the Streamlit application.
"""

import os
from huggingface_hub import HfApi

def main():
    # Get HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")

    # Get username dynamically
    api = HfApi()
    user_info = api.whoami(token=token)
    hf_username = user_info["name"]
    space_repo = f"{hf_username}/wellness-tourism-app"

    # Create Hugging Face Space with Docker SDK
    print(f"🚀 Creating Hugging Face Space: {space_repo}")
    api.create_repo(
        repo_id=space_repo,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=token
    )

    # Upload deployment files to the Space
    deployment_dir = "tourism_project/deployment"
    files_to_upload = ["Dockerfile", "app.py", "requirements.txt"]

    for filename in files_to_upload:
        filepath = os.path.join(deployment_dir, filename)
        if os.path.exists(filepath):
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=space_repo,
                repo_type="space",
                token=token
            )
            print(f"   ✅ Uploaded: {filename}")
        else:
            print(f"   ⚠️ File not found: {filepath}")

    print(f"\n🎉 Streamlit app deployed at: https://huggingface.co/spaces/{space_repo}")
    print(f"   (It may take a few minutes for the Space to build and start)")

if __name__ == "__main__":
    main()
