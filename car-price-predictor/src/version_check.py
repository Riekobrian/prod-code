"""
Version compatibility checker for the Car Price Predictor app.
"""
import streamlit as st
import sys
import pkg_resources

REQUIRED_VERSIONS = {
    'scikit-learn': '1.6.1',
    'joblib': '1.4.2',
    'scipy': '1.13.1',
    'numpy': '1.26.4',
    'pandas': '2.1.4',
    'threadpoolctl': '3.5.0'
}

def get_installed_version(package_name):
    """Get the installed version of a package."""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_versions():
    """
    Check if installed package versions match required versions.
    Returns (is_compatible, error_message)
    """
    errors = []
    
    for package, required_version in REQUIRED_VERSIONS.items():
        installed_version = get_installed_version(package)
        if installed_version != required_version:
            errors.append(f"{package}: required={required_version}, installed={installed_version}")
    
    if errors:
        error_msg = "\n".join([
            "❌ Version mismatch detected!",
            "This app requires specific package versions to work correctly.",
            "Installed versions don't match requirements:",
            "",
            *[f"• {err}" for err in errors],
            "",
            "To fix this:",
            "1. Delete existing environment",
            "2. Create new environment",
            "3. Install requirements with: pip install -r requirements.txt --no-deps",
        ])
        return False, error_msg
    
    return True, None

def ensure_compatibility():
    """
    Check version compatibility and show error if incompatible.
    Stops the app if versions don't match.
    """
    is_compatible, error_msg = check_versions()
    
    if not is_compatible:
        st.error(error_msg)
        st.stop()