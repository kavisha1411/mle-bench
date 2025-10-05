#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal, robust sklearn baseline for MLE-bench tabular competitions.

Usage (one line):
  python -u agents/mysklearnagent/tabular_lr.py \
    --competition new-york-city-taxi-fare-prediction \
    --output submissions/nyc_sklearn_baseline.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Helpers to load competition
# -----------------------------


def _load_with_mlebench(competition_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Try to load data using mlebench helper if the editable install is available.
    Returns (train_df, test_df, sample_sub_df).
    """
    from mlebench.mlebench import load_competition  # available after `pip install -e .`

    bundle = load_competition(competition_id)
    # bundle keys may include train_path, test_path, sample_submission_path, etc.
    # Normalize to DataFrames:
    def _maybe_df(x):
        if isinstance(x, (str, Path)) and str(x).endswith(".json"):
            return pd.read_json(x, lines=True)
        if isinstance(x, (str, Path)) and str(x).endswith(".csv"):
            return pd.read_csv(x)
        return x

    train = _maybe_df(bundle["train"])
    test = _maybe_df(bundle["test"])
    sample_sub = _maybe_df(bundle["sample_submission"])
    return train, test, sample_sub


def _load_from_cache(competition_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fallback loader for prepared datasets in the default cache:
      ~/.cache/mlebench/<id>/<id>/{train.json,test.json,sample_submission.csv}
    """
    base = Path.home() / "Library" / "Caches" / "mle-bench" / competition_id / competition_id
    train_p = base / "train.json"
    test_p = base / "test.json"
    sample_p = base / "sample_submission.csv"

    if not train_p.exists() or not test_p.exists() or not sample_p.exists():
        raise FileNotFoundError(
            f"Could not find prepared data in {base}. " f"Run: mlebench prepare -c {competition_id}"
        )

    train = pd.read_json(train_p, lines=True)
    test = pd.read_json(test_p, lines=True)
    sample_sub = pd.read_csv(sample_p)
    return train, test, sample_sub


def load_data(competition_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (train, test, sample_submission) with a robust fallback."""
    try:
        return _load_with_mlebench(competition_id)
    except Exception:
        # Fall back to cache path if mlebench import not available
        return _load_from_cache(competition_id)


# -----------------------------
# Feature engineering utilities
# -----------------------------


def add_time_features(df: pd.DataFrame, cand_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add hour/day-of-week/month from obvious datetime columns,
    then drop the original timestamp column.
    """
    df = df.copy()
    if cand_cols is None:
        cand_cols = [c for c in df.columns if "time" in c or "date" in c or c.endswith("_dt")]
    for col in cand_cols:
        if col in df.columns:
            t = pd.to_datetime(df[col], errors="coerce")
            if t.notna().sum() > 0:
                df[f"{col}_hour"] = t.dt.hour
                df[f"{col}_dow"] = t.dt.dayofweek
                df[f"{col}_month"] = t.dt.month
                # Many MLE-bench tasks don't need the raw timestamp
                df.drop(columns=[col], inplace=True)
    return df


def guess_id_and_target(train: pd.DataFrame, sample_sub: pd.DataFrame) -> Tuple[str, str]:
    """
    Heuristically determine (id_col, target_col).
    - id_col: first column in sample_submission.
    - target_col: the other column in sample_submission that also appears in train.
      If multiple candidates exist, pick the first.
    """
    ss_cols = list(sample_sub.columns)
    if len(ss_cols) < 2:
        raise ValueError("sample_submission.csv must have at least 2 columns (id + prediction).")

    id_col = ss_cols[0]
    # Prefer the second column in sample submission if present in train
    for c in ss_cols[1:]:
        if c in train.columns:
            return id_col, c

    # Fallback: common names
    candidates = ["target", "label", "fare_amount"]
    for c in candidates:
        if c in train.columns:
            return id_col, c

    # Last resort: any sample-sub column that is present in train
    for c in ss_cols:
        if c != id_col and c in train.columns:
            return id_col, c

    raise ValueError("Could not infer target column from sample_submission and train.")


# -----------------------------
# Model pipeline
# -----------------------------


def make_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build a simple but solid tabular pipeline:
    - Numeric: median impute
    - Categorical: most_frequent + one-hot (bounded), dense output
    - Ridge regression for stability
    """
    # Determine numeric vs categorical after basic sanitization
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore", max_categories=64, sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Ridge(alpha=1.0, random_state=0)

    pipe = Pipeline(
        [
            ("prep", preprocess),
            ("reg", model),
        ]
    )
    return pipe


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Sklearn tabular baseline for MLE-bench.")
    parser.add_argument(
        "--competition",
        required=True,
        help="Competition ID (e.g., new-york-city-taxi-fare-prediction)",
    )
    parser.add_argument("--output", required=True, help="Path to save the submission CSV")
    args = parser.parse_args()

    comp_id = args.competition
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading data for '{comp_id}' ...", flush=True)
    train, test, sample_sub = load_data(comp_id)

    id_col, target_col = guess_id_and_target(train, sample_sub)
    print(f"[info] inferred id_col='{id_col}', target_col='{target_col}'", flush=True)

    # Split target & features BEFORE column detection to avoid leaking the target
    if target_col not in train.columns:
        raise ValueError(f"Target column '{target_col}' not found in train columns.")

    y_train = pd.to_numeric(train[target_col], errors="coerce").values
    X_train = train.drop(columns=[target_col]).copy()
    X_test = test.copy()

    # Light, safe time features (e.g., 'pickup_datetime' in NYC)
    # Try obvious timestamp column names on both frames
    time_like = [c for c in X_train.columns if "time" in c or "date" in c or c.endswith("_dt")]
    X_train = add_time_features(X_train, time_like)
    X_test = add_time_features(X_test, time_like)

    # Build and fit pipeline
    pipe = make_pipeline(X_train)
    print(f"[fit] X_train={X_train.shape}, y_train={y_train.shape}", flush=True)
    pipe.fit(X_train, y_train)

    # Predict in the exact sample-submission order if the ID column exists in test
    if id_col in test.columns:
        # Ensure we align predictions to sample_sub ordering if needed
        # Most Kaggle formats expect the same order as sample_sub.
        # We'll predict on test in its current order, then reorder to sample_sub.
        preds = pipe.predict(X_test)
        pred_df = pd.DataFrame({id_col: test[id_col].values, target_col: preds})
        # If sample_sub IDs match a subset ordering, left-join to enforce order
        sub = sample_sub[[id_col]].merge(pred_df, on=id_col, how="left")
        # Fill any missing (shouldn't happen) with model predictions by order
        if sub[target_col].isna().any():
            sub[target_col] = sub[target_col].fillna(preds[: len(sub)])
    else:
        # Fall back: just write predictions in sample size order
        preds = pipe.predict(X_test)
        sub = sample_sub.copy()
        if target_col not in sub.columns:
            # If sample_sub had a different name, replace the second column
            second_col = sub.columns[1]
            sub = sub.rename(columns={second_col: target_col})
        sub[target_col] = preds

    # Basic safety: ensure correct columns and length
    assert list(sub.columns[:2]) == [
        id_col,
        target_col,
    ], f"Submission first two columns should be [{id_col}, {target_col}], got {list(sub.columns[:2])}"
    assert len(sub) == len(
        sample_sub
    ), "Submission must have the same number of rows as sample_submission."

    sub.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} with {len(sub)} rows.", flush=True)


if __name__ == "__main__":
    main()
