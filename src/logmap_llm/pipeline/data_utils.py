from __future__ import annotations

import pandas as pd

# def normalise_prediction_column(series: pd.Series) -> pd.Series:
#     """
#     Normalise an Oracle_prediction column to boolean/string values
#     NOTE: allows for round-tripping of boolean values:
#         internal repr -> external repr -> internal repr
#     """
#     def _normalise(val):
#         if isinstance(val, bool):
#             return val
#         if isinstance(val, str):
#             low = val.strip().lower()
#             if low == "true":           # NOTE: should we include 'yes' and 'no' here too actually?
#                 return True
#             if low == "false":
#                 return False
#             return val  # e.g. "error", "skipped"
#         return val  # otherwise some other datatype (NaN)

#     return series.map(_normalise)


def _normalise_value(val):
    """Normalise a single prediction value to bool or passthrough string."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        return val  # e.g. "error", "skipped"
    return val  # NaN etc.


def normalise_prediction_column(df_or_series):
    """
    Normalise an Oracle_prediction column to boolean/string values.

    Handles the string-to-boolean round-trip that occurs when predictions
    are saved to CSV and reloaded.  Supports both true/false and yes/no
    answer formats.

    Parameters
    ----------
    df_or_series : pd.DataFrame or pd.Series
        If a DataFrame, modifies the 'Oracle_prediction' column in place.
        If a Series, returns a new normalised Series.

    Returns
    -------
    pd.Series or None
        Returns the normalised Series if given a Series;
        modifies DataFrame in place and returns None otherwise.
    """
    if isinstance(df_or_series, pd.DataFrame):
        df_or_series['Oracle_prediction'] = df_or_series['Oracle_prediction'].map(
            _normalise_value
        )
        return None
    else:
        return df_or_series.map(_normalise_value)
