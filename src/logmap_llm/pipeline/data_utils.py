from __future__ import annotations

import pandas as pd


def normalise_prediction_column(series: pd.Series) -> pd.Series:
    """
    Normalise an Oracle_prediction column to boolean/string values
    NOTE: allows for round-tripping of boolean values:
        internal repr -> external repr -> internal repr
    """
    def _normalise(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            low = val.strip().lower()
            if low == "true":           # NOTE: should we include 'yes' and 'no' here too actually?
                return True
            if low == "false":
                return False
            return val  # e.g. "error", "skipped"
        return val  # otherwise some other datatype (NaN)

    return series.map(_normalise)
