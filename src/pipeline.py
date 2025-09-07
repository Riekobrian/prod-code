from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

TARGET_COLUMN = "price"

class CarAgeAdder(BaseEstimator, TransformerMixin):
    """Add car_age = max(0, current_year - year_of_manufacture).
    Uses max valid year in the input as current_year_ if not provided."""
    def __init__(self, current_year: int | None = None):
        self.current_year = current_year

    def fit(self, X: pd.DataFrame, y=None):
        if self.current_year is None:
            years = pd.to_numeric(X["year_of_manufacture"], errors="coerce")
            finite = years[np.isfinite(years)]
            self.current_year_ = int(finite.max()) if finite.size else 2024
        else:
            self.current_year_ = int(self.current_year)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        years = pd.to_numeric(X["year_of_manufacture"], errors="coerce")
        X["car_age"] = np.maximum(0, self.current_year_ - years)
        return X

def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns from dataframe (excluding target if present)."""
    cols = [c for c in df.columns if c != TARGET_COLUMN]
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocess_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocess

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values
    return X, y

def make_target_transformer(log1p: bool = True) -> FunctionTransformer:
    if log1p:
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x, validate=True)

def prepare_dataset(
    df: pd.DataFrame,
    is_train: bool = False
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str], List[str]]:
    """
    Prepare dataset for training/inference:
      - Normalize 'usage type' → 'usage_type'
      - Add car_age
      - Return (X, y_or_None, numeric_cols, categorical_cols)
    """
    # Normalize 'usage type'
    if "usage type" in df.columns and "usage_type" not in df.columns:
        df = df.rename(columns={"usage type": "usage_type"})

    # Add car_age
    age_adder = CarAgeAdder()
    df = age_adder.fit(df).transform(df)

    if is_train:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Training mode expects '{TARGET_COLUMN}' column present.")
        numeric_cols, categorical_cols = get_feature_lists(df.drop(columns=[TARGET_COLUMN]))
        return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN].values, numeric_cols, categorical_cols
    else:
        # inference: no target required
        if TARGET_COLUMN in df.columns:
            X = df.drop(columns=[TARGET_COLUMN])
        else:
            X = df
        numeric_cols, categorical_cols = get_feature_lists(X)
        return X, None, numeric_cols, categorical_cols