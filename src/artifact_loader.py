"""
Enhanced model loading functions for the Car Price Predictor app.
"""
import os
import streamlit as st
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from src.model_loader import safe_load_pickle, diagnose_pickle_file
from src.model_download import ensure_model_files

def load_model_safely(models_dir: Path, artifact_name: str, filename: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load a single model file with enhanced error handling and recovery.
    """
    file_path = models_dir / filename
    
    # First attempt: Try to load existing file
    data, error = safe_load_pickle(file_path, artifact_name)
    if not error:
        return data, None
        
    # If loading failed or file doesn't exist, try to download
    st.warning(f"⚠️ Failed to load {artifact_name}, attempting to download...")
    if ensure_model_files(str(models_dir)):
        # Try loading again after download
        data, error = safe_load_pickle(file_path, artifact_name)
        if not error:
            return data, None
    
    # If still failed after download, run diagnostics
    diagnosis = diagnose_pickle_file(file_path)
    return None, f"{diagnosis}\n{error}"

@st.cache_resource
def load_all_artifacts(project_root: str, model_mapping: Dict[str, str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Load all required model artifacts with comprehensive error handling.
    """
    try:
        # Ensure models directory exists
        models_dir = Path(project_root) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # First, try to download/update all files
        st.info("🔄 Checking for model updates...")
        ensure_model_files(str(models_dir))
        
        # Load each artifact
        artifacts = {}
        errors = []
        
        for key, filename in model_mapping.items():
            st.info(f"📦 Loading {key}...")
            data, error = load_model_safely(models_dir, key, filename)
            
            if error:
                errors.append(f"❌ {key}: {error}")
            else:
                artifacts[key] = data
                st.success(f"✅ Loaded {key}")
        
        if not artifacts:
            raise ValueError("No artifacts could be loaded:\n" + "\n".join(errors))
        
        if errors:
            st.warning("⚠️ Some artifacts had loading issues:\n" + "\n".join(errors))
        
        return artifacts, None
        
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"