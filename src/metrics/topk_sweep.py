#!/usr/bin/env python3
"""Sweep top-k output truncation and score each k with the official CDA scorer.

The per-image CDA term divides by the number of predictions, so truncating
output to the k most confident detections changes the score substantially.
This script evaluates several k values and reports the curve.

Requires the official challenge scorer (distributed with the NASA dataset):

    python src/metrics/topk_sweep.py --debug preds/seg-debug.csv \
        --truth data/nasa-craters/val-gt.csv \
        --scorer data/nasa-craters/scorer.py --tag seg
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def score(pred_csv, truth, scorer, out_dir):
    subprocess.run([sys.executable, str(scorer), "--pred", str(pred_csv),
                    "--truth", str(truth), "--out_dir", str(out_dir)],
                   capture_output=True, text=True)
    return float((Path(out_dir) / "result.txt").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", required=True, help="CSV with per-detection confidence")
    ap.add_argument("--truth", required=True, help="validation truth CSV")
    ap.add_argument("--scorer", default="data/nasa-craters/scorer.py")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ks", default="3,5,8,10,11,15,25,60")
    ap.add_argument("--out-dir", default="preds/sweep")
    args = ap.parse_args()

    dbg = pd.read_csv(args.debug)
    truth = Path(args.truth)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer_out = out_dir / "scorer-out"

    print(f"[{args.tag}] total detections: {len(dbg)}")
    print(f"{'k':>4} | {'CDA':>6} | {'dets':>5}")
    for k in [int(x) for x in args.ks.split(",")] + [0]:
        if k == 0:
            sub = dbg.drop(columns=["conf"])
            key = "all"
        else:
            sub = dbg.sort_values("conf", ascending=False).groupby("inputImage").head(k).drop(columns=["conf"])
            key = k
        pred_path = out_dir / f"pred-{args.tag}-top{key}.csv"
        sub.to_csv(pred_path, index=False)
        s = score(pred_path, truth, args.scorer, scorer_out)
        print(f"{key:>4} | {s:>6.2f} | {len(sub):>5}")


if __name__ == "__main__":
    main()
