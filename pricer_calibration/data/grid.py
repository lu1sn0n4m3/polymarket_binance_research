"""Build 100ms mid-price grid via previous-tick sampling.

Outputs DataFrame with columns: t, S, logS, r
where t is in milliseconds (UTC), S is the forward-filled mid,
logS = log(S), r = diff(logS).
"""

import numpy as np
import pandas as pd


def build_grid(
    bbo: pd.DataFrame,
    delta_ms: int = 100,
) -> pd.DataFrame:
    """Construct a fixed-interval grid from cleaned BBO data.

    Args:
        bbo: DataFrame with columns [ts_event, mid], sorted by ts_event.
        delta_ms: Grid spacing in milliseconds.

    Returns:
        DataFrame with columns [t, S, logS, r].
        First row has r = 0.
    """
    if bbo.empty:
        return pd.DataFrame(columns=["t", "S", "logS", "r"])

    ts = bbo["ts_event"].values
    t_start = int(np.ceil(ts[0] / delta_ms) * delta_ms)
    t_end = int(np.floor(ts[-1] / delta_ms) * delta_ms)

    grid_t = np.arange(t_start, t_end + 1, delta_ms, dtype=np.int64)
    grid_df = pd.DataFrame({"t": grid_t})

    # Previous-tick sample via merge_asof
    grid_df = pd.merge_asof(
        grid_df,
        bbo[["ts_event", "mid"]].rename(columns={"ts_event": "t"}),
        on="t",
        direction="backward",
    )

    grid_df = grid_df.rename(columns={"mid": "S"})

    # Drop rows before first observation
    grid_df = grid_df.dropna(subset=["S"]).reset_index(drop=True)

    # Log price and returns
    grid_df["logS"] = np.log(grid_df["S"].values)
    log_vals = grid_df["logS"].values
    r = np.empty(len(log_vals))
    r[0] = 0.0
    r[1:] = np.diff(log_vals)
    grid_df["r"] = r

    return grid_df
