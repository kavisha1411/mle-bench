import argparse
import json
import os
from pathlib import Path

import pandas as pd

CACHE = Path(os.path.expanduser("~/Library/Caches/mle-bench"))


def data_paths(competition_id: str):
    d = CACHE / "data" / competition_id / competition_id
    # many comps use train.csv / test.csv and sampleSubmission.csv OR sample_submission.csv
    # we try both sample names
    samp = d / "sample_submission.csv"
    if not samp.exists():
        samp = d / "sampleSubmission.csv"
    return d / "train.csv", d / "test.csv", samp


def make_predictions(competition_id: str, out_csv: Path):
    train_p, test_p, sample_p = data_paths(competition_id)
    sample = pd.read_csv(sample_p)
    # Heuristic: if a “label” column exists use majority class; otherwise constant/mean baseline
    y_col_candidates = ["label", "target", "Survived", "isFraud", "fare_amount"]  # add as needed
    y_col = None
    if train_p.exists():
        train = pd.read_csv(train_p, nrows=50_000)  # keep small; adjust per dataset
        for c in y_col_candidates:
            if c in train.columns:
                y_col = c
                break

    preds = sample.copy()
    target_cols = [c for c in sample.columns if c not in ("id", "ID", "Id", "request_id")]
    if y_col and y_col in train.columns:
        # categorical or numeric?
        if str(train[y_col].dtype).startswith(("int", "int64", "object", "category")):
            mode = train[y_col].mode().iloc[0]
            for c in target_cols:
                preds[c] = mode
        else:
            mean_val = float(train[y_col].mean())
            for c in target_cols:
                preds[c] = mean_val
    else:
        # No clear label; fill with zeros (or 0.5 for probs)
        fill_val = 0.5 if "prob" in target_cols[0].lower() else 0
        for c in target_cols:
            preds[c] = fill_val

    preds.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_predictions(args.competition, out)
    print(json.dumps({"wrote": str(out)}))


if __name__ == "__main__":
    main()
