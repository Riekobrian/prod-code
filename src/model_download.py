import os
import requests
from tqdm import tqdm

import os
import gdown
from pathlib import Path
import streamlit as st

# Map of model files to their Google Drive file IDs
MODEL_FILES = {
    "feature_info.pkl": "1L-cKOasNPIsETavQPa6Fxxkc8qxa6Zgr",
    "gradientboosting_tuned.pkl": "1bPMIgFl0Dn0920oaZtdUg1W2gzYGxzxp",
    "priors.pkl": "1bUkgWivaOXlvRj4jMGVrTRwb7QdgK-n3",
    "target_transformer.pkl": "1pHd2xuYLOBlpp2AjY5Hnw4HULg14xJA5"
}

def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """
    Download a file from Google Drive using gdown.
    Returns True if successful, False otherwise.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construct the direct download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download the file
        st.info(f"Downloading {os.path.basename(output_path)}...")
        success = gdown.download(url, output_path, quiet=False)
        
        if success:
            st.success(f"✅ Downloaded {os.path.basename(output_path)}")
            return True
        else:
            st.error(f"Failed to download {os.path.basename(output_path)}")
            return False
            
    except Exception as e:
        st.error(f"Error downloading {os.path.basename(output_path)}: {str(e)}")
        return False

def ensure_model_files(model_dir: str) -> bool:
    """
    Check if all required model files exist, download them if they don't.
    Returns True if all files are available, False if any download failed.
    """
    model_dir = Path(model_dir)
    success = True
    
    for filename, file_id in MODEL_FILES.items():
        file_path = model_dir / filename
        
        # Skip if file exists and is not empty
        if file_path.exists() and file_path.stat().st_size > 0:
            continue
            
        # Download if missing or empty
        if not download_from_gdrive(file_id, str(file_path)):
            success = False
    
    return success

def download_file(url: str, destination: str) -> bool:
    """
    Download a file from a given URL to a destination path.
    Returns True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        file_size = int(response.headers.get('content-length', 0))

        # Open the local file to write the downloaded content
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=file_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def ensure_model_files(model_dir: str) -> bool:
    """
    Check if all required model files exist, download them if they don't.
    Returns True if all files are available, False if any download failed.
    """
    success = True
    for filename, url in MODEL_URLS.items():
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            if not download_file(url, file_path):
                success = False
                print(f"Failed to download {filename}")
                continue
            print(f"Successfully downloaded {filename}")
    return success