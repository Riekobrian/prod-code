"""
Enhanced model loading functions for the Car Price Predictor app.
"""
import os
import streamlit as st
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from src.model_loader import safe_load_pickle, diagnose_pickle_file
from src.model_download import ensure_model_files, check_model_files, download_missing_models

def load_model_safely(models_dir: Path, artifact_name: str, filename: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load a single model file with enhanced error handling and recovery.
    """
    file_path = models_dir / filename
    
    # First attempt: Try to load existing file
    data, error = safe_load_pickle(file_path, artifact_name)
    if not error:
        return data, None
        
    # If loading failed or file doesn't exist, try to redownload
    st.warning(f"⚠️ Failed to load {artifact_name}, attempting to download...")
    
    # Remove potentially corrupted file
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass
            
    # Attempt to download and load again
    if download_missing_models(models_dir):
        data, error = safe_load_pickle(file_path, artifact_name)
        if not error:
            return data, None
    
    # If still failed after download, provide error details
    diagnosis = diagnose_pickle_file(file_path)
    return None, f"Failed to load {artifact_name}: {diagnosis}"

@st.cache_resource
def load_all_artifacts(project_root: str, model_mapping: Dict[str, str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Load all required model artifacts with comprehensive error handling.
    """
    try:
        # Ensure models directory exists
        models_dir = Path(project_root) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # First check if files exist and are valid
        missing_files = []
        for filename in model_mapping.values():
            file_path = models_dir / filename
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_files.append(filename)
        
        # Download missing files if needed
        if missing_files:
            with st.spinner("🔄 Downloading model files..."):
                if not download_missing_models(models_dir):
                    raise ValueError("Failed to download model files from GitHub release")
        
        # Load each artifact
        artifacts = {}
        errors = []
        
        with st.spinner("📦 Loading model files..."):
            for key, filename in model_mapping.items():
                data, error = safe_load_pickle(models_dir / filename, key)
                
                if error:
                    errors.append(f"❌ {key}: {error}")
                else:
                    artifacts[key] = data
        
        if not artifacts:
            raise ValueError("Failed to load model files. Please check that the GitHub release contains valid model files.")
            
        if errors:
            error_msg = "\n".join(errors)
            return None, error_msg
        
        return artifacts, None
        
    except Exception as e:
        return None, str(e)