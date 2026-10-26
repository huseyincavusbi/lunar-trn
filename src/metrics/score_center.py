#!/usr/bin/env python3
"""Diagnostic recall (per-GT 'any det hits') at multiple alignment criteria, per size band.
center: det center inside GT box | circle: within min(w,h)/2 | iou10/30/50: any det IoU >= thr.
Diagnostic only (no det reuse management) - complements official AP.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

KM_PER_PX = 0.1
BANDS = [(0.0, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]


def load_gt(gt_dir, img_size=416):
    gt = {}
    for f in sorted(Path(gt_dir).glob("*.txt")):
        boxes = []
        for line in open(f):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = [float(v) * img_size for v in p[1:5]]
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        gt[f.stem] = boxes
    return gt


def load_pred(pred_csv):
    pred = defaultdict(list)
    with open(pred_csv) as f:
        for r in csv.DictReader(f):
            stem = r["image"].rsplit(".", 1)[0]
            pred[stem].append((float(r["conf"]), float(r["x1"]), float(r["y1"]),
                               float(r["x2"]), float(r["y2"])))
    return dict(pred)


def iou_mat(d, g):
    d = np.asarray(d, dtype=float).reshape(-1, 4)
    g = np.asarray(g, dtype=float).reshape(-1, 4)
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    x1 = np.maximum(d[:, None, 0], g[None, :, 0]); y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 2], g[None, :, 2]); y2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    da = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]); ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (da[:, None] + ga[None, :] - inter + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conf-op", type=float, default=0.25)
    args = ap.parse_args()

    gt = load_gt(args.gt_dir)
    pred = load_pred(args.pred)

    stats = {f"{lo}-{hi}": defaultdict(int) for lo, hi in BANDS}
    totals = {f"{lo}-{hi}": 0 for lo, hi in BANDS}
    for stem, gts in gt.items():
        dets = [d for d in pred.get(stem, []) if d[0] >= args.conf_op]
        boxes = [d[1:] for d in dets]
        M = iou_mat(boxes, gts)
        for j, g in enumerate(gts):
            lo, hi = None, None
            size_km = max(g[2] - g[0], g[3] - g[1]) * KM_PER_PX
            for a, b in BANDS:
                if a <= size_km < b:
                    key = f"{a}-{b}"
                    break
            totals[key] += 1
            if not boxes:
                continue
            gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
            r = min(g[2] - g[0], g[3] - g[1]) / 2
            hit_center = hit_circle = hit10 = hit30 = hit50 = False
            for i, b in enumerate(boxes):
                bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if g[0] <= bc[0] <= g[2] and g[1] <= bc[1] <= g[3]:
                    hit_center = True
                if (bc[0] - gc[0]) ** 2 + (bc[1] - gc[1]) ** 2 <= r * r:
                    hit_circle = True
                v = M[i, j]
                if v >= 0.5:
                    hit50 = True
                elif v >= 0.3:
                    hit30 = True
                elif v >= 0.1:
                    hit10 = True
            s = stats[key]
            s["n"] += 1
            s["center"] += hit_center
            s["circle"] += hit_circle
            s["iou10"] += hit10 or hit30 or hit50
            s["iou30"] += hit30 or hit50
            s["iou50"] += hit50
            s["anyrec"] += (hit_center or hit50 or hit30 or hit10)

    print(f"== {args.tag} ==")
    hdr = "band        n      center  circle  iou.1   iou.3   iou.5"
    print(hdr)
    for key in [f"{a}-{b}" for a, b in BANDS]:
        s = stats[key]; n = max(s["n"], 1)
        print(f"{key:<11} {s['n']:<6} {s['center']/n:.3f}   {s['circle']/n:.3f}   {s['iou10']/n:.3f}   {s['iou30']/n:.3f}   {s['iou50']/n:.3f}")


if __name__ == "__main__":
    main()
