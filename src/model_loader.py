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
        
        return False
    except:
        return False

def safe_load_pickle(file_path: Path, file_key: str = None) -> tuple:
    """
    Safely load a pickle file with extensive error checking.
    Returns (data, error_message)
    """
    try:
        if not file_path.exists():
            return None, f"File not found: {file_path}"

        # Check file size
        if file_path.stat().st_size == 0:
            return None, f"Empty file: {file_path}"

        # Create a temporary copy to avoid file locking issues
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp.write(file_path.read_bytes())
            tmp_path = Path(tmp.name)

        try:
            # Try different loading methods
            errors = []
            
            # Method 1: Standard pickle
            try:
                with open(tmp_path, 'rb') as f:
                    data = pickle.load(f)
                if validate_model_content(data):
                    os.unlink(tmp_path)
                    return data, None
                errors.append("Pickle load succeeded but content validation failed")
            except Exception as e:
                errors.append(f"Pickle load failed: {str(e)}")

            # Method 2: Joblib
            try:
                data = joblib.load(tmp_path)
                if validate_model_content(data):
                    os.unlink(tmp_path)
                    return data, None
                errors.append("Joblib load succeeded but content validation failed")
            except Exception as e:
                errors.append(f"Joblib load failed: {str(e)}")

            # If we get here, all methods failed
            os.unlink(tmp_path)
            error_msg = f"All loading methods failed for {file_key or file_path}:\\n" + "\\n".join(errors)
            return None, error_msg

        finally:
            # Ensure temp file is cleaned up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return None, f"Unexpected error loading {file_key or file_path}: {str(e)}"

def diagnose_pickle_file(file_path: Path) -> str:
    """
    Diagnose issues with a pickle file.
    Returns a diagnostic message.
    """
    try:
        if not file_path.exists():
            return "❌ File not found"

        size = file_path.stat().st_size
        if size == 0:
            return "❌ File is empty"

        # Check first few bytes
        with open(file_path, 'rb') as f:
            header = f.read(50)
            
        # Try to decode as text to check for HTML content
        try:
            text = header.decode('utf-8', errors='ignore')
            if '<' in text or 'html' in text.lower():
                return "❌ File appears to contain HTML - likely corrupted by Google Drive"
        except:
            pass

        return f"ℹ️ File size: {size} bytes, appears to be binary data"

    except Exception as e:
        return f"❌ Error diagnosing file: {str(e)}"