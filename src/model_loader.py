import pickle
import joblib
import os
from pathlib import Path
import tempfile
import streamlit as st

def validate_model_content(data):
    """Validate that loaded data appears to be a valid model"""
    try:
        # Check if it's a scikit-learn model (has predict method)
        if hasattr(data, 'predict'):
            return True
            
        # Check if it's a dictionary containing models/transformers
        if isinstance(data, dict):
            return any(hasattr(v, 'transform') or hasattr(v, 'predict') 
                      for v in data.values())
            
        # Check if it's a transformer (has transform method)
        if hasattr(data, 'transform'):
            return True
        
        return False
    except:
        return False

def safe_load_pickle(file_path: Path, file_key: str = None) -> tuple:
    """
    Safely load a pickle or joblib file.
    Returns (data, error_message)
    """
    try:
        if not file_path.exists():
            return None, "File not found"

        if file_path.stat().st_size == 0:
            return None, "File is empty"

        # Create a temporary copy for safe loading
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp.write(file_path.read_bytes())
            tmp_path = Path(tmp.name)

        try:
            errors = []
            
            # Try pickle first
            try:
                with open(tmp_path, 'rb') as f:
                    data = pickle.load(f)
                if validate_model_content(data):
                    return data, None
                errors.append("Invalid model format")
            except Exception as e:
                errors.append(f"Pickle error: {str(e)}")

            # Try joblib if pickle fails
            try:
                data = joblib.load(tmp_path)
                if validate_model_content(data):
                    return data, None
                errors.append("Invalid model format")
            except Exception as e:
                errors.append(f"Joblib error: {str(e)}")

            return None, f"Failed to load model: {' | '.join(errors)}"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return None, str(e)

def diagnose_pickle_file(file_path: Path) -> str:
    """
    Check if a file exists and is non-empty.
    """
    try:
        if not file_path.exists():
            return "File not found"
        if file_path.stat().st_size == 0:
            return "File is empty"
        return f"File exists ({file_path.stat().st_size} bytes)"
    except Exception as e:
        return str(e)