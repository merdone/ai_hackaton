from __future__ import annotations

import pandas as pd


# Keep only true legacy aliases here.
ACTION_ALIASES: dict[str, str] = {}


def normalize_action_value(value: object) -> str:
    action = str(value).strip()
    return ACTION_ALIASES.get(action, action)


def normalize_action_column(df: pd.DataFrame, column_name: str = "action") -> pd.DataFrame:
    if column_name not in df.columns:
        return df

    normalized = df.copy()
    normalized[column_name] = normalized[column_name].map(normalize_action_value)
    return normalized

