#!/usr/bin/env python3
"""Bootstrap confidence interval for the CDA score under top-k selection.

Replicates the official scorer's per-image computation, then resamples
images with replacement to quantify how precisely the 200-image validation
estimate is known.

    python src/metrics/bootstrap_cda.py --debug preds/seg-debug.csv \
        --truth data/nasa-craters/val-gt.csv --k 11 --n-boot 10000
"""
import argparse
import math

import numpy as np
import pandas as pd

XI_2_THRESH = 13.277
NN_PIX_ERR_RATIO = 0.07


def calcYmat(a, b, phi):
    unit_1 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    unit_2 = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
    unit_3 = np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])
    return unit_1 @ unit_2 @ unit_3


def calc_dGA(Yi, Yj, yi, yj):
    multiplicand = 4 * np.sqrt(np.linalg.det(Yi) * np.linalg.det(Yj)) / np.linalg.det(Yi + Yj)
    exponent = (-0.5 * (yi - yj).T @ Yi @ np.linalg.inv(Yi + Yj) @ Yj @ (yi - yj))
    cos = min(1, multiplicand * np.exp(exponent[0, 0]))
    return np.arccos(cos)


def dGA(A, B):
    A_phi = A['ellipseRotation(deg)'] / 180 * math.pi
    B_phi = B['ellipseRotation(deg)'] / 180 * math.pi
    A_Y = calcYmat(A['ellipseSemimajor(px)'], A['ellipseSemiminor(px)'], A_phi)
    B_Y = calcYmat(B['ellipseSemimajor(px)'], B['ellipseSemiminor(px)'], B_phi)
    A_y = np.array([[A['ellipseCenterX(px)']], [A['ellipseCenterY(px)']]])
    B_y = np.array([[B['ellipseCenterX(px)']], [B['ellipseCenterY(px)']]])
    d = calc_dGA(A_Y, B_Y, A_y, B_y)
    ab_min = np.min([A['ellipseSemimajor(px)'], A['ellipseSemiminor(px)']])
    comparison_sig = NN_PIX_ERR_RATIO * ab_min
    ref_sig = 0.85 / np.sqrt(A['ellipseSemimajor(px)'] * A['ellipseSemiminor(px)']) * comparison_sig
    return d, d * d / (ref_sig * ref_sig)


def score1(ts, ps):
    if len(ps) == 0:
        return 0.0
    t_empty = len(ts) == 1 and ts[0].get('ellipseSemimajor(px)') == -1
    p_empty = len(ps) == 1 and ps[0].get('ellipseSemimajor(px)') == -1
    if t_empty and p_empty:
        return 1.0
    if t_empty != p_empty:
        return 0.0
    dgas = []
    for t in ts:
        best_p, best_dGA, best_xi_2 = None, math.pi / 2, float('inf')
        for p in ps:
            if p['matched']:
                continue
            rA = min(t['ellipseSemimajor(px)'], t['ellipseSemiminor(px)'])
            rB = min(p['ellipseSemimajor(px)'], p['ellipseSemiminor(px)'])
            if rA > 1.5 * rB or rB > 1.5 * rA:
                continue
            r = min(rA, rB)
            if abs(t['ellipseCenterX(px)'] - p['ellipseCenterX(px)']) > r:
                continue
            if abs(t['ellipseCenterY(px)'] - p['ellipseCenterY(px)']) > r:
                continue
            d, xi_2 = dGA(t, p)
            if d < best_dGA:
                best_dGA, best_p, best_xi_2 = d, p, xi_2
        if best_xi_2 < XI_2_THRESH:
            t['matched'] = True
            best_p['matched'] = True
            dgas.append(1 - best_dGA / math.pi)
    if len(dgas) == 0:
        return 0.0
    return (sum(dgas) / len(ps)) * min(1.0, len(dgas) / min(10, len(ts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", required=True, help="CSV with per-detection confidence")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    truth_df = pd.read_csv(args.truth)
    truth_df['matched'] = False
    truth = truth_df.set_index('inputImage').groupby(level='inputImage').apply(
        lambda g: g.to_dict(orient='records')).to_dict()

    det_df = pd.read_csv(args.debug)
    det = {}
    for img, g in det_df.groupby('inputImage'):
        rows = g.sort_values('conf', ascending=False).head(args.k).to_dict(orient='records')
        for r in rows:
            r['matched'] = False
        det[img] = rows

    imgs = sorted(truth.keys())
    scores = np.zeros(len(imgs))
    for i, img in enumerate(imgs):
        scores[i] = score1([dict(t) for t in truth[img]], det.get(img, []))

    print(f"k={args.k}: replicated CDA = {scores.mean() * 100:.4f}  (n={len(scores)} images)")

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(scores), size=(args.n_boot, len(scores)))
    means = scores[idx].mean(axis=1) * 100
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]  (std {means.std():.2f}, {args.n_boot} resamples)")


if __name__ == "__main__":
    main()
