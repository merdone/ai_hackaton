from __future__ import annotations

import pandas as pd


def safe_primary(values: pd.Series) -> str:
    if values.empty:
        return "-"
    mode = values.mode(dropna=True)
    if mode.empty:
        return "-"
    return str(mode.iloc[0])

