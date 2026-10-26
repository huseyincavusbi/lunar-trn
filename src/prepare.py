#!/usr/bin/env python3
"""Build a YOLO dataset from the NASA Crater Detection Challenge ellipse labels.

The challenge ground truth describes each crater as an ellipse
(centre, semi-major, semi-minor, rotation). This script converts it to

  --task box : axis-aligned bounding boxes (width/height from the rotated ellipse)
  --task seg : 48-point polygons following the same ellipse convention as the scorer

Example
-------
python src/prepare.py --task box \
    --gt data/nasa-craters/train-gt.csv \
    --images-root data/nasa-craters/train-extracted/train \
    --out data/c1-full-yolo --region all --val-count 200

python src/prepare.py --task seg \
    --gt data/nasa-craters/train-gt.csv \
    --images-root data/nasa-craters/train-extracted/train \
    --out data/c1-full-seg --region all --val-count 200
"""
import argparse
import math
import random
import shutil
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd

from common.geometry import ellipse_polygon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["box", "seg"], required=True)
    ap.add_argument("--gt", required=True, help="train-gt.csv with ellipse columns")
    ap.add_argument("--images-root", required=True,
                    help="root containing altitudeXX/longitudeXX/... image folders")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--region", default="all",
                    help="inputImage prefix filter, e.g. 'altitude01/longitude02' (default: all)")
    ap.add_argument("--val-count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-gt-out", default=None,
                    help="where to write the validation truth CSV (default: next to --gt)")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    if args.region != "all":
        gt = gt[gt["inputImage"].str.startswith(args.region)]
    ids = sorted(gt["inputImage"].unique())
    random.seed(args.seed)
    random.shuffle(ids)
    val_ids = set(ids[:args.val_count])
    print(f"images: {len(ids)} | train {len(ids) - len(val_ids)} | val {len(val_ids)}")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_labels = 0
    for img_id in ids:
        split = "val" if img_id in val_ids else "train"
        src = Path(args.images_root) / f"{img_id}.png"
        if not src.exists():
            print("MISSING:", src)
            continue
        stem = img_id.replace("/", "___")
        shutil.copy2(src, out / "images" / split / f"{stem}.png")
        W, H = Image.open(src).size
        lines = []
        for _, r in gt[gt["inputImage"] == img_id].iterrows():
            cx, cy = r["ellipseCenterX(px)"], r["ellipseCenterY(px)"]
            a, b = r["ellipseSemimajor(px)"], r["ellipseSemiminor(px)"]
            phi_deg = r["ellipseRotation(deg)"]
            if args.task == "box":
                phi = math.radians(phi_deg)
                dx = math.sqrt((a * math.cos(phi)) ** 2 + (b * math.sin(phi)) ** 2)
                dy = math.sqrt((a * math.sin(phi)) ** 2 + (b * math.cos(phi)) ** 2)
                lines.append(f"0 {cx / W:.6f} {cy / H:.6f} {2 * dx / W:.6f} {2 * dy / H:.6f}")
            else:
                pts = ellipse_polygon(cx, cy, a, b, phi_deg)
                norm = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in pts)
                lines.append(f"0 {norm}")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_labels += len(lines)

    val_gt = Path(args.val_gt_out) if args.val_gt_out else Path(args.gt).parent / "val-gt.csv"
    gt[gt["inputImage"].isin(val_ids)].to_csv(val_gt, index=False)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: crater\n")

    kind = "boxes" if args.task == "box" else "polygons"
    print(f"{kind}: {n_labels} | dataset: {out} | val truth: {val_gt}")


if __name__ == "__main__":
    main()
