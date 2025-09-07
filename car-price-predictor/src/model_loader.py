import pickle
import joblib
import os
import sys
import warnings
from pathlib import Path
import tempfile
import streamlit as st

def handle_version_compatibility(file_path: Path, file_key: str = None):
    """Handle scikit-learn version compatibility issues"""
    try:
        # Mock the _RemainderColsList class if needed
        if 'sklearn.compose._column_transformer' in sys.modules:
            module = sys.modules['sklearn.compose._column_transformer']
            if not hasattr(module, '_RemainderColsList'):
                class _RemainderColsList:
                    def __init__(self, columns):
                        self.columns = columns
                    def __iter__(self):
                        return iter(self.columns)
                    def __len__(self):
                        return len(self.columns)
                setattr(module, '_RemainderColsList', _RemainderColsList)
        
        # Try loading with version compatibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except:
        return None

def validate_model_content(data, file_key=None):
    """Validate that loaded data appears to be a valid model or supported data structure"""
    try:
        # Special case for feature info which is just a dictionary
        if file_key == "feat_info" and isinstance(data, dict):
            return True
            
        # Special case for priors which is also a dictionary
        if file_key == "priors" and isinstance(data, dict):
            return True
            
        # Check if it's a scikit-learn model (has predict method)
        if hasattr(data, 'predict'):
            return True
            
        # Check if it's a transformer (has transform method)
        if hasattr(data, 'transform'):
            return True
            
        # Check if it's a dictionary containing models/transformers
        if isinstance(data, dict):
            return any(hasattr(v, 'transform') or hasattr(v, 'predict') 
                      for v in data.values())
        
        return False
    except:
        return False

def safe_load_pickle(file_path: Path, file_key: str = None) -> tuple:
    """
    Safely load a pickle or joblib file with version compatibility handling.
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
            
            # Try joblib first (better for sklearn objects)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = joblib.load(tmp_path)
                if validate_model_content(data, file_key):
                    return data, None
            except Exception as e:
                errors.append(f"Joblib error: {str(e)}")

            # Try pickle if joblib fails
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(tmp_path, 'rb') as f:
                        data = pickle.load(f)
                if validate_model_content(data, file_key):
                    return data, None
            except Exception as e:
                errors.append(f"Pickle error: {str(e)}")

            # If both methods fail, try with version compatibility handling
            try:
                data = handle_version_compatibility(tmp_path, file_key)
                if data is not None:
                    return data, None
            except Exception as e:
                errors.append(f"Version compatibility error: {str(e)}")

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