#!/usr/bin/env python3
"""
Compute gaze-estimation metrics for two algorithms from a CSV like:

dot_x,dot_y,appearance_x,appearance_y,appearance_error,geometric_x,geometric_y,geometric_error
...

It will:
- parse numeric columns (handles blanks and "N/A")
- (re)compute pixel error from (dot_x,dot_y) vs (algo_x,algo_y) when possible
- use provided *_error if coordinates missing (optional)
- report per-algorithm:
  - N total rows
  - N valid predictions
  - success rate
  - mean / std
  - median
  - IQR (Q3-Q1)
  - p90 / p95
  - max
Optionally writes a per-row output CSV with computed errors.
"""

from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


NA_STRINGS = {"", "N/A", "NA", "NaN", "nan", "null", "None"}


def to_num(s) -> float:
    if pd.isna(s):
        return np.nan
    if isinstance(s, str):
        t = s.strip()
        if t in NA_STRINGS:
            return np.nan
        # allow commas as decimal separators just in case
        t = t.replace(",", ".")
        try:
            return float(t)
        except ValueError:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def px_error(dot_x, dot_y, pred_x, pred_y) -> float:
    if any(np.isnan(v) for v in [dot_x, dot_y, pred_x, pred_y]):
        return np.nan
    dx = pred_x - dot_x
    dy = pred_y - dot_y
    return float(math.hypot(dx, dy))


def robust_stats(errors: np.ndarray) -> Dict[str, Any]:
    """errors: 1D array, already filtered to finite values."""
    if errors.size == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    q1 = np.quantile(errors, 0.25)
    q3 = np.quantile(errors, 0.75)
    return {
        "n_valid": int(errors.size),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
        "median": float(np.median(errors)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p90": float(np.quantile(errors, 0.90)),
        "p95": float(np.quantile(errors, 0.95)),
        "max": float(np.max(errors)),
    }


def compute_algo_error(
    df: pd.DataFrame,
    algo: str,
    dot_x_col: str,
    dot_y_col: str,
) -> pd.Series:
    """
    Returns a Series of per-row errors for algorithm 'algo', using:
      1) computed error from coordinates if dot_x/dot_y and algo_x/algo_y exist
      2) else fall back to provided algo_error column if present
    """
    x_col = f"{algo}_x"
    y_col = f"{algo}_y"
    e_col = f"{algo}_error"

    # numeric versions (won't overwrite originals)
    dot_x = df[dot_x_col].map(to_num) if dot_x_col in df.columns else pd.Series(np.nan, index=df.index)
    dot_y = df[dot_y_col].map(to_num) if dot_y_col in df.columns else pd.Series(np.nan, index=df.index)

    pred_x = df[x_col].map(to_num) if x_col in df.columns else pd.Series(np.nan, index=df.index)
    pred_y = df[y_col].map(to_num) if y_col in df.columns else pd.Series(np.nan, index=df.index)

    computed = pd.Series(
        (px_error(dx, dy, px, py) for dx, dy, px, py in zip(dot_x, dot_y, pred_x, pred_y)),
        index=df.index,
        dtype="float64",
    )

    if e_col in df.columns:
        provided = df[e_col].map(to_num).astype("float64")
        # prefer computed when available, else use provided
        out = computed.where(~computed.isna(), provided)
        return out

    return computed


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    out = pd.DataFrame(rows)
    # nice ordering
    cols = [
        "algo",
        "n_total",
        "n_valid",
        "success_rate",
        "mean",
        "std",
        "median",
        "iqr",
        "p90",
        "p95",
        "max",
    ]
    out = out[cols]
    # formatting
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}" if np.isfinite(x) else "nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to input CSV")
    ap.add_argument("--dot-x-col", default="dot_x")
    ap.add_argument("--dot-y-col", default="dot_y")
    ap.add_argument(
        "--algos",
        default="appearance,geometric",
        help="Comma-separated algorithm prefixes (expects <algo>_x,<algo>_y and/or <algo>_error)",
    )
    ap.add_argument(
        "--out-per-row",
        default="",
        help="Optional path to save CSV with computed errors (<algo>_error_computed columns)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    print(df.head())

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    if not algos:
        print("No algorithms specified via --algos", file=sys.stderr)
        return 2

    summary_rows = []
    per_row = df.copy()

    n_total = len(df)
    for algo in algos:
        err = compute_algo_error(df, algo, args.dot_x_col, args.dot_y_col).astype("float64")
        per_row[f"{algo}_error_computed"] = err

        valid = err.to_numpy()
        valid = valid[np.isfinite(valid)]
        stats = robust_stats(valid)

        summary_rows.append(
            {
                "algo": algo,
                "n_total": n_total,
                "n_valid": stats["n_valid"],
                "success_rate": stats["n_valid"] / n_total if n_total else np.nan,
                "mean": stats["mean"],
                "std": stats["std"],
                "median": stats["median"],
                "iqr": stats["iqr"],
                "p90": stats["p90"],
                "p95": stats["p95"],
                "max": stats["max"],
            }
        )

    print_summary_table(summary_rows)

    if args.out_per_row:
        per_row.to_csv(args.out_per_row, index=False)
        print(f"\nSaved per-row output to: {args.out_per_row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

