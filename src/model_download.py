import os
import requests
from tqdm import tqdm

MODEL_URLS = {
    "feature_info.pkl": "https://drive.google.com/uc?id=1L-cKOasNPIsETavQPa6Fxxkc8qxa6Zgr",
    "gradientboosting_tuned.pkl": "https://drive.google.com/uc?id=1bPMIgFl0Dn0920oaZtdUg1W2gzYGxzxp",
    "priors.pkl": "https://drive.google.com/uc?id=1bUkgWivaOXlvRj4jMGVrTRwb7QdgK-n3",
    "target_transformer.pkl": "https://drive.google.com/uc?id=1pHd2xuYLOBlpp2AjY5Hnw4HULg14xJA5"
}

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