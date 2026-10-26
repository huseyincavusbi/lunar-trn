#!/usr/bin/env python3
"""Quantify spatial leakage: score predictions on adjacent vs isolated validation tiles.

Validation patches are split by distance to the nearest TRAINING tile (degrees):
  near : <= --near-deg  (edge-adjacent; border craters may appear in both splits)
  iso  : >  --iso-deg   (spatially separated; closest to clean generalization)

For each subset and each prediction CSV the script filters the CSV and runs
score_real.py (with --no-robbins). Example:

    python src/metrics/subset_eval.py \
        --base data/lu3m6tgt/extracted/LU3M6TGT_yolo_format \
        --pred zero-shot=preds/box_1024.csv \
        --pred finetuned=preds/ft_1024.csv \
        --out /tmp/lu_sub
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def bounds(path):
    return [float(x) for x in Path(path).stem.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dataset root with train/ and valid/")
    ap.add_argument("--pred", action="append", required=True,
                    help="label=path/of/predictions.csv (repeatable)")
    ap.add_argument("--out", default="/tmp/lu_sub")
    ap.add_argument("--near-deg", type=float, default=1.5)
    ap.add_argument("--iso-deg", type=float, default=2.5)
    ap.add_argument("--score-script", default=str(Path(__file__).parent / "score_real.py"))
    args = ap.parse_args()

    base = Path(args.base)
    train_imgs = sorted((base / "train" / "images").glob("*.png"))
    valid_imgs = sorted((base / "valid" / "images").glob("*.png"))
    tb = [bounds(p) for p in train_imgs]
    vb = [bounds(p) for p in valid_imgs]

    tc = np.array([[(b[0] + b[1]) / 2 % 360, (b[2] + b[3]) / 2] for b in tb])
    vc = np.array([[(b[0] + b[1]) / 2 % 360, (b[2] + b[3]) / 2] for b in vb])
    dlon = np.abs(vc[:, 0][:, None] - tc[:, 0][None, :])
    dlon = np.minimum(dlon, 360 - dlon)
    dlat = np.abs(vc[:, 1][:, None] - tc[:, 1][None, :])
    nearest = np.sqrt(dlon ** 2 + dlat ** 2).min(axis=1)
    stems = [p.stem for p in valid_imgs]

    subsets = {
        "near": {s for s, n in zip(stems, nearest) if n <= args.near_deg},
        "iso": {s for s, n in zip(stems, nearest) if n > args.iso_deg},
    }

    out_root = Path(args.out)
    results = {}
    for name, subset in subsets.items():
        ddir = out_root / name
        (ddir / "gt").mkdir(parents=True, exist_ok=True)
        (ddir / "images").mkdir(parents=True, exist_ok=True)
        for stem in subset:
            for src, dst in ((base / "valid" / "labels" / f"{stem}.txt", ddir / "gt"),
                             (base / "valid" / "images" / f"{stem}.png", ddir / "images")):
                link = dst / src.name
                if not link.exists():
                    os.symlink(src.resolve(), link)
        print(f"[{name}] {len(subset)} patches", flush=True)

        for spec in args.pred:
            label, pred_path = spec.split("=", 1)
            pred_path = Path(pred_path)
            if not pred_path.exists():
                print(f"  missing {pred_path}")
                continue
            filtered = ddir / f"pred_{label}.csv"
            with open(pred_path, newline="") as f, open(filtered, "w", newline="") as g:
                reader = csv.reader(f)
                writer = csv.writer(g)
                writer.writerow(next(reader))
                for row in reader:
                    if row[0].rsplit(".", 1)[0] in subset:
                        writer.writerow(row)
            tag = f"{label}_{name}"
            cmd = [sys.executable, args.score_script, "--pred", str(filtered),
                   "--gt-dir", str(ddir / "gt"), "--images-dir", str(ddir / "images"),
                   "--tag", tag, "--out-dir", str(out_root / "out"), "--no-robbins"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = proc.stdout.strip().splitlines()
            print("  " + (lines[-1] if lines else f"FAILED rc={proc.returncode} {proc.stderr[-200:]}"))
            if lines:
                import json
                summary = json.loads(lines[0])
                results[tag] = summary

    print("\n=== SUBSET SUMMARY (IoU-based AP) ===")
    for tag, s in results.items():
        print(f"{tag:16s} mAP50={s['mAP50']:.4f}  mAP50-95={s['mAP50_95']:.4f}  n_gt={s['gt']}")


if __name__ == "__main__":
    main()
