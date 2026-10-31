#!/usr/bin/env python3
"""CRESENT pose, Tier B: blind position matching (no position prior).

Attitude and altitude are assumed known (star-tracker / orbit determination).
With attitude known, each detection defines a world ray.  For a catalogue crater
P and ray d, the camera must satisfy C = P - s*d with |C| = R + altitude
(line-sphere intersection).  Every (detection, crater) pair votes for a camera
position; peaks are scored by reprojecting the catalogue and, optionally, by
checking the apparent foreshortening (ellipse axis ratio vs cos incidence).

Example
-------
python src/pose/pose_blind.py \
    --base data/cresent/extracted/CRESENT \
    --dets preds/crescent.csv --robbins data/robbins/lunar_crater_database_robbins_2018.csv \
    --out preds/pose_blind.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose import common

H_DEFAULT = 100000.0


def vote_peaks(uv, P_vote, A, r_cam, bin_deg, n_peaks=20):
    """Line-sphere voting; returns (peaks, lat0, lon0) in histogram index space."""
    dirs = common.ground_dirs(uv, A)
    pd_ = dirs @ P_vote.T
    p2 = (P_vote ** 2).sum(axis=1)
    s = pd_ + np.sqrt(np.maximum(pd_ ** 2 + (r_cam ** 2 - p2)[None, :], 0))
    C = P_vote[None, :, :] - s[:, :, None] * dirs[:, None, :]
    cn = np.linalg.norm(C, axis=2)
    clat = np.degrees(np.arcsin(np.clip(C[:, :, 2] / np.maximum(cn, 1e-9), -1, 1)))
    clon = np.degrees(np.arctan2(C[:, :, 1], C[:, :, 0])) % 360.0
    ilat = np.round(clat / bin_deg).astype(np.int64)
    ilon = np.round(clon / bin_deg).astype(np.int64)
    la0, lo0 = int(ilat.min()), int(ilon.min())
    nla = int(ilat.max() - la0) + 1
    nlo = int(ilon.max() - lo0) + 1
    flat = (ilat - la0) * nlo + (ilon - lo0)
    hist = np.bincount(flat.ravel(), minlength=nla * nlo).reshape(nla, nlo).astype(float)
    acc = np.zeros_like(hist)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            acc[max(0, di):nla + min(0, di), max(0, dj):nlo + min(0, dj)] += \
                hist[max(0, -di):nla - max(0, di), max(0, -dj):nlo - max(0, dj)]
    top = np.argsort(acc.ravel())[::-1][:max(120, n_peaks * 6)]
    peaks = []
    for f in top:
        i, j = divmod(int(f), nlo)
        if all(abs(i - p[0]) > 3 or abs(j - p[1]) > 3 for p in peaks):
            peaks.append((i, j))
        if len(peaks) >= n_peaks:
            break
    return peaks, la0, lo0


def score_position(uv, ratios, P_score, A, lat0, lon0, r_cam, img_tol, use_ratio, ratio_tol):
    Cc = r_cam * np.array([np.cos(np.radians(lat0)) * np.cos(np.radians(lon0)),
                           np.cos(np.radians(lat0)) * np.sin(np.radians(lon0)),
                           np.sin(np.radians(lat0))])
    puv, pok = common.project(P_score, A, Cc)
    inf = pok & (puv[:, 0] > 0) & (puv[:, 0] < 1024) & (puv[:, 1] > 0) & (puv[:, 1] < 1024)
    if inf.sum() == 0:
        return 0
    preds = puv[inf]
    d2 = ((uv[:, None, :] - preds[None, :, :]) ** 2).sum(axis=2)
    j = d2.argmin(axis=1)
    nn = np.sqrt(d2[np.arange(len(uv)), j])
    hit = nn <= img_tol
    if use_ratio:
        pr = common.predicted_ratios(P_score[inf], Cc)
        hit = hit & (np.abs(ratios - pr[j]) <= ratio_tol)
    return int(hit.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--robbins", required=True)
    ap.add_argument("--regions", default="")
    ap.add_argument("--angles", default="0,20,40,60")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--altitude-km", type=float, default=100.0)
    ap.add_argument("--vote-min-km", type=float, default=5.0)
    ap.add_argument("--score-min-km", type=float, default=4.0)
    ap.add_argument("--min-det-px", type=float, default=13.0)
    ap.add_argument("--bin-deg", type=float, default=0.05)
    ap.add_argument("--img-tol", type=float, default=10.0)
    ap.add_argument("--ratio-tol", type=float, default=0.25)
    ap.add_argument("--score-mode", choices=["ellipse", "position"], default="ellipse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r_cam = common.R_MOON + args.altitude_km * 1000.0
    regions = args.regions.split(",") if args.regions else common.REGIONS
    angles = [int(a) for a in args.angles.split(",")]
    use_ratio = args.score_mode == "ellipse"
    rows = []

    for region in regions:
        P_vote = common.load_catalog(args.robbins, args.base, region, args.vote_min_km)
        P_score = common.load_catalog(args.robbins, args.base, region, args.score_min_km)
        for angle in angles:
            dets = common.dets_for(args.dets, region, angle)
            errs = []
            for idx, (centers, sizes, ratios) in sorted(dets.items())[:args.max_images]:
                keep = sizes >= args.min_det_px
                if keep.sum() < 8:
                    continue
                uv = centers[keep]
                rr = ratios[keep]
                A_gt, C_gt = common.abs_pose(args.base, region, angle, idx)
                peaks, la0, lo0 = vote_peaks(uv, P_vote, A_gt, r_cam, args.bin_deg)
                best = None
                for (i, j) in peaks:
                    lat_c = (la0 + i) * args.bin_deg
                    lon_c = (lo0 + j) * args.bin_deg
                    for dlat in (-0.04, -0.02, 0.0, 0.02, 0.04):
                        for dlon in (-0.04, -0.02, 0.0, 0.02, 0.04):
                            sc = score_position(uv, rr, P_score, A_gt, lat_c + dlat, lon_c + dlon,
                                                r_cam, args.img_tol, use_ratio, args.ratio_tol)
                            if best is None or sc > best[0]:
                                best = (sc, lat_c + dlat, lon_c + dlon)
                if best is None:
                    continue
                _, lat_est, lon_est = best
                tl = np.degrees(np.arcsin(C_gt[2] / np.linalg.norm(C_gt)))
                tn = np.degrees(np.arctan2(C_gt[1], C_gt[0])) % 360.0
                err = float(np.hypot((lat_est - tl) * 30.34,
                                     (lon_est - tn) * 30.34 * np.cos(np.radians(tl))))
                errs.append(err)
                rows.append(dict(region=region, angle=angle, idx=idx, err_km=err))
            if errs:
                a = np.array(errs)
                print(f"{region} {angle:2d}deg: n={len(a):2d} | success(<5km) "
                      f"{100 * (a < 5).mean():3.0f}% | median {np.median(a):7.1f} km")

    a = np.array([r["err_km"] for r in rows])
    if len(a):
        print(f"\npooled: n={len(a)} | success(<5km) {100 * (a < 5).mean():.0f}% | "
              f"median {np.median(a):.1f} km")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
