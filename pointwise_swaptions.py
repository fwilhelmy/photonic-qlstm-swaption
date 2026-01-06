# swaptions_pointwise_neighbors.py
# ------------------------------------------------------------
# Pointwise swaption dataset + neighbor feature injection
# Column format expected (from your train.xlsx):
#   "Tenor : <number>; Maturity : <number>"
# plus a "Date" column.
#
# Pointwise sample schema:
#   {
#     "date": pd.Timestamp,
#     "tenor": float,
#     "maturity": float,
#     "vol": float,
#     # optional injected:
#     # "neighbors_8": [8 floats]
#     # "diffs_4": [4 floats]
#     # "stats_2": [mean, std]
#   }
# ------------------------------------------------------------

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


_TENOR_MAT_RE = re.compile(
    r"Tenor\s*:\s*(?P<tenor>[-+]?\d*\.?\d+)\s*;\s*Maturity\s*:\s*(?P<maturity>[-+]?\d*\.?\d+)",
    flags=re.IGNORECASE,
)


def load_swaptions_excel(path: str) -> pd.DataFrame:
    """
    Loads the Excel and returns a DataFrame indexed by Date (sorted).
    """
    df = pd.read_excel(path)
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the Excel file.")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def prepare_pointwise_dataset(df: pd.DataFrame) -> Tuple[List[dict], List[float], List[float]]:
    """
    Converts a swaption surface DataFrame into pointwise samples.

    Input:
      - df: index = Date, columns include tenor/maturity columns as strings.

    Output:
      - samples: list of dict samples (date, tenor, maturity, vol)
      - tenor_grid: sorted unique tenor values
      - maturity_grid: sorted unique maturity values
    """
    # Parse column metadata
    parsed_cols: List[Tuple[str, float, float]] = []
    tenors = set()
    maturities = set()

    for col in df.columns:
        m = _TENOR_MAT_RE.fullmatch(str(col).strip())
        if not m:
            # Skip non-surface columns if any exist besides Date (already index)
            continue
        tenor = float(m.group("tenor"))
        maturity = float(m.group("maturity"))
        parsed_cols.append((col, tenor, maturity))
        tenors.add(tenor)
        maturities.add(maturity)

    if not parsed_cols:
        raise ValueError(
            "No swaption columns matched the expected pattern "
            "'Tenor : <num>; Maturity : <num>'."
        )

    tenor_grid = sorted(tenors)
    maturity_grid = sorted(maturities)

    samples: List[dict] = []
    # Iterate by date and parsed columns
    for date, row in df.iterrows():
        for col, tenor, maturity in parsed_cols:
            vol = row[col]
            # If there are NaNs, you can skip or keep them; here we skip
            if pd.isna(vol):
                continue
            samples.append(
                {
                    "date": date,
                    "tenor": tenor,
                    "maturity": maturity,
                    "vol": float(vol),
                }
            )

    return samples, tenor_grid, maturity_grid


# -----------------------------
# Internal helpers
# -----------------------------

def _build_lookup(samples: List[dict]) -> Dict[Tuple[pd.Timestamp, float, float], float]:
    """
    surface[(date, tenor, maturity)] -> vol
    """
    surface: Dict[Tuple[pd.Timestamp, float, float], float] = {}
    for s in samples:
        surface[(s["date"], s["tenor"], s["maturity"])] = s["vol"]
    return surface


def _index_maps(tenor_grid: List[float], maturity_grid: List[float]) -> Tuple[Dict[float, int], Dict[float, int]]:
    t2i = {v: i for i, v in enumerate(tenor_grid)}
    m2i = {v: i for i, v in enumerate(maturity_grid)}
    return t2i, m2i


def _get_vol(
    surface: Dict[Tuple[pd.Timestamp, float, float], float],
    date: pd.Timestamp,
    tenor: float,
    maturity: float,
    default: float,
) -> float:
    return surface.get((date, tenor, maturity), default)


# ============================================================
# 1) Inject 8-neighborhood vols (adds 8 features)
# ============================================================

def inject_8_neighborhood(samples: List[dict], tenor_grid: List[float], maturity_grid: List[float]) -> None:
    """
    Adds:
      sample["neighbors_8"] = [8 neighbor vols]
    Neighbor order (Moore neighborhood, excluding center):
      (-1,-1), (-1,0), (-1,+1),
      ( 0,-1),         ( 0,+1),
      (+1,-1), (+1,0), (+1,+1)

    Boundary handling: clamp-to-center (uses center vol when neighbor missing).
    """
    surface = _build_lookup(samples)
    t2i, m2i = _index_maps(tenor_grid, maturity_grid)

    offsets = [(-1, -1), (-1, 0), (-1, +1),
               (0, -1),           (0, +1),
               (+1, -1), (+1, 0), (+1, +1)]

    for s in samples:
        date, tenor, maturity, center = s["date"], s["tenor"], s["maturity"], s["vol"]
        ti = t2i[tenor]
        mi = m2i[maturity]

        neigh = []
        for dti, dmi in offsets:
            nti = ti + dti
            nmi = mi + dmi
            if 0 <= nti < len(tenor_grid) and 0 <= nmi < len(maturity_grid):
                nt = tenor_grid[nti]
                nm = maturity_grid[nmi]
                neigh.append(_get_vol(surface, date, nt, nm, center))
            else:
                neigh.append(center)
        s["neighbors_8"] = neigh


# ============================================================
# 2) Inject directional differences (adds 4 features)
# ============================================================

def inject_directional_differences(samples: List[dict], tenor_grid: List[float], maturity_grid: List[float]) -> None:
    """
    Adds:
      sample["diffs_4"] = [dTenorPlus, dTenorMinus, dMatPlus, dMatMinus]

    Each difference is:
      neighbor_vol - center_vol

    Boundary handling: clamp-to-center (difference becomes 0 at boundary).
    """
    surface = _build_lookup(samples)
    t2i, m2i = _index_maps(tenor_grid, maturity_grid)

    # (+tenor, -tenor, +mat, -mat) as index steps
    dirs = [(+1, 0), (-1, 0), (0, +1), (0, -1)]

    for s in samples:
        date, tenor, maturity, center = s["date"], s["tenor"], s["maturity"], s["vol"]
        ti = t2i[tenor]
        mi = m2i[maturity]

        diffs = []
        for dti, dmi in dirs:
            nti = ti + dti
            nmi = mi + dmi
            if 0 <= nti < len(tenor_grid) and 0 <= nmi < len(maturity_grid):
                nt = tenor_grid[nti]
                nm = maturity_grid[nmi]
                v = _get_vol(surface, date, nt, nm, center)
                diffs.append(v - center)
            else:
                diffs.append(0.0)
        s["diffs_4"] = diffs


# ============================================================
# 3) Inject local summary stats (adds 2 features)
# ============================================================

def inject_local_stats(samples: List[dict], tenor_grid: List[float], maturity_grid: List[float]) -> None:
    """
    Adds:
      sample["stats_2"] = [mean_neighbors, std_neighbors]

    Uses the same 8 neighbors as inject_8_neighborhood (excluding center).
    Boundary handling: clamp-to-center (center substituted when missing).
    """
    surface = _build_lookup(samples)
    t2i, m2i = _index_maps(tenor_grid, maturity_grid)

    offsets = [(-1, -1), (-1, 0), (-1, +1),
               (0, -1),           (0, +1),
               (+1, -1), (+1, 0), (+1, +1)]

    for s in samples:
        date, tenor, maturity, center = s["date"], s["tenor"], s["maturity"], s["vol"]
        ti = t2i[tenor]
        mi = m2i[maturity]

        vals = []
        for dti, dmi in offsets:
            nti = ti + dti
            nmi = mi + dmi
            if 0 <= nti < len(tenor_grid) and 0 <= nmi < len(maturity_grid):
                nt = tenor_grid[nti]
                nm = maturity_grid[nmi]
                vals.append(_get_vol(surface, date, nt, nm, center))
            else:
                vals.append(center)

        vals_np = np.asarray(vals, dtype=float)
        s["stats_2"] = [float(vals_np.mean()), float(vals_np.std(ddof=0))]


# ------------------------------------------------------------
# Example usage (comment out in production):
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_swaptions_excel("train.xlsx")
    samples, tenor_grid, maturity_grid = prepare_pointwise_dataset(df)

    inject_8_neighborhood(samples, tenor_grid, maturity_grid)
    inject_directional_differences(samples, tenor_grid, maturity_grid)
    inject_local_stats(samples, tenor_grid, maturity_grid)

    print("n_samples:", len(samples))
    print("tenors:", tenor_grid[:10], "…", tenor_grid[-3:])
    print("maturities:", maturity_grid[:10], "…", maturity_grid[-3:])
    print("sample[0] keys:", samples[0].keys())
    print("neighbors_8 len:", len(samples[0]["neighbors_8"]))
    print("diffs_4 len:", len(samples[0]["diffs_4"]))
    print("stats_2:", samples[0]["stats_2"])
