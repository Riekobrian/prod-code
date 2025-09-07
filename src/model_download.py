import os
from pathlib import Path
import streamlit as st
import requests
from tqdm.auto import tqdm

# GitHub repository information
REPO_OWNER = "Riekobrian"
REPO_NAME = "prod-code"
RELEASE_TAG = "v1.0.0"  # You'll create this release tag on GitHub

# List of required model files
MODEL_FILES = [
    "feature_info.pkl",
    "gradientboosting_tuned.pkl",
    "priors.pkl",
    "target_transformer.pkl"
]

def get_download_url(filename: str) -> str:
    """Generate the GitHub release asset download URL"""
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}/{filename}"

def download_file(url: str, destination: Path, filename: str) -> bool:
    """
    Download a file from a URL with progress bar.
    Returns True if successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(destination.parent, exist_ok=True)
        
        with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress_bar.update(size)
        return True
    except requests.RequestException as e:
        st.error(f"Failed to download {filename}: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error saving {filename}: {str(e)}")
        return False

def check_model_files(model_dir: Path) -> bool:
    """
    Check if all required model files exist in the specified directory.
    Returns True if all files exist, False otherwise.
    """
    all_files_exist = True
    for filename in MODEL_FILES:
        file_path = model_dir / filename
        if not file_path.exists():
            st.warning(f"Missing model file: {filename}")
            all_files_exist = False
        elif file_path.stat().st_size == 0:
            st.warning(f"Empty model file: {filename}")
            all_files_exist = False
    return all_files_exist

def download_missing_models(model_dir: Path) -> bool:
    """
    Download missing model files from GitHub releases.
    Returns True if all files are present after download attempts, False otherwise.
    """
    success = True
    for filename in MODEL_FILES:
        file_path = model_dir / filename
        if not file_path.exists() or file_path.stat().st_size == 0:
            url = get_download_url(filename)
            if not download_file(url, file_path, filename):
                success = False
    return success

def ensure_model_files(base_path: str = "models") -> bool:
    """
    Ensure all required model files are present, downloading them if necessary.
    Returns True if all files are present after potential downloads, False otherwise.
    """
    model_dir = Path(base_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # First check if all files exist
    if check_model_files(model_dir):
        st.success("All model files are present.")
        return True
        
    # If any files are missing or empty, try to download them
    st.info("Downloading missing model files...")
    if download_missing_models(model_dir):
        st.success("All model files downloaded successfully.")
        return True
    else:
        st.error("Failed to download all model files.")
        return False