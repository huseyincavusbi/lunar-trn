#!/usr/bin/env python3
"""Calibrate the detector's ellipse-ratio bias (parallax/foreshortening response).

Uses oracle correspondences (GT pose) across regions/angles: predicted incidence
ratio (cos) vs detected fitted ratio (b/a).  Fits an affine map and writes JSON.

Note: applying this calibration to blind matching was tested and did NOT improve
success (compressing the predicted ratios reduces discrimination between candidate
poses).  It is kept as a diagnostic/negative-result artifact.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--robbins", required=True)
    ap.add_argument("--regions", default="")
    ap.add_argument("--angles", default="0,20,40,60")
    ap.add_argument("--max-images", type=int, default=15)
    ap.add_argument("--min-det-px", type=float, default=13.0)
    ap.add_argument("--min-cat-km", type=float, default=4.0)
    ap.add_argument("--out", default="preds/ratio_calibration.json")
    args = ap.parse_args()

    regions = args.regions.split(",") if args.regions else common.REGIONS
    angles = [int(a) for a in args.angles.split(",")]
    r_preds, r_dets = [], []

    for region in regions:
        P = common.load_catalog(args.robbins, args.base, region, args.min_cat_km)
        for angle in angles:
            dets = common.dets_for(args.dets, region, angle)
            for idx, (centers, sizes, ratios) in sorted(dets.items())[:args.max_images]:
                keep = sizes >= args.min_det_px
                if keep.sum() < 6:
                    continue
                uv = centers[keep]
                A_gt, C_gt = common.abs_pose(args.base, region, angle, idx)
                uv_pred, ok = common.project(P, A_gt, C_gt)
                inf = ok & (uv_pred[:, 0] > 0) & (uv_pred[:, 0] < 1024) & \
                    (uv_pred[:, 1] > 0) & (uv_pred[:, 1] < 1024)
                pairs = common.mutual_match(uv_pred[inf], uv, 10.0)
                if not pairs:
                    continue
                pr = common.predicted_ratios(P[inf], C_gt)
                for pi, di in pairs:
                    r_preds.append(float(pr[pi]))
                    r_dets.append(float(ratios[keep][di]))

    rp = np.array(r_preds)
    rd = np.array(r_dets)
    print(f"pairs: {len(rp)}")
    bins = np.linspace(0.45, 1.0, 12)
    bx, by = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rp >= lo) & (rp < hi)
        if m.sum() >= 10:
            bx.append(float(np.median(rp[m])))
            by.append(float(np.median(rd[m])))
    bx, by = np.array(bx), np.array(by)
    coef = np.polyfit(bx, by, 1) if len(bx) >= 2 else np.array([0.0, 1.0])
    for x, y in zip(bx, by):
        print(f"  r_pred {x:.3f} -> r_det {y:.3f}")
    print(f"affine: r_det_exp = {coef[0]:.4f} * r_pred + {coef[1]:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(slope=float(coef[0]), intercept=float(coef[1]), n=len(rp)), f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
