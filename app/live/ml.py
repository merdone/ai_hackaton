from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from app.utils.action_normalization import normalize_action_value


@st.cache_resource
def load_rf_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_action(
    model: Any,
    features: dict[str, float],
    train_features: tuple[str, ...],
    fallback_confidence: float,
) -> tuple[str, float]:
    if model is None:
        return "Detected (No ML)", fallback_confidence

    feature_row = {name: float(features.get(name, 0.0)) for name in train_features}
    X = pd.DataFrame([feature_row], columns=list(train_features))

    predicted_label = normalize_action_value(model.predict(X)[0])
    confidence = fallback_confidence
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())
    return predicted_label, confidence

