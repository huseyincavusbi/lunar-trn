#!/usr/bin/env python3
"""CRESENT pose, Tier A: oracle-correspondence evaluation.

Catalogue craters are projected with the reconstructed ground-truth pose, matched
to detections, and the pose is re-estimated with solvePnP.  This measures the
accuracy the detector supports *given* correspondences (an upper bound; not blind).

Example
-------
python src/pose/pose_tier_a.py \
    --base data/cresent/extracted/CRESENT \
    --dets preds/crescent.csv --robbins data/robbins/lunar_crater_database_robbins_2018.csv \
    --dem data/lola/ldem_64.img --out preds/pose_tier_a.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose import common

try:
    from pose import dem
except Exception:  # DEM is optional
    dem = None


def agg_table(rows):
    if not rows:
        return []
    out = []
    for angle in sorted({r["angle"] for r in rows}):
        rs = [r for r in rows if r["angle"] == angle]
        t = np.array([r["trans_m"] for r in rs])
        rot = np.array([r["rot_deg"] for r in rs])
        out.append(dict(angle=angle, n=len(rs),
                        matches=float(np.median([r["n_match"] for r in rs])),
                        trans_med=float(np.median(t)),
                        trans_p25=float(np.percentile(t, 25)),
                        trans_p75=float(np.percentile(t, 75)),
                        rot_med=float(np.median(rot)),
                        rot_p75=float(np.percentile(rot, 75))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="CRESENT dataset root")
    ap.add_argument("--dets", required=True, help="detections CSV (region, angle, image_index, conf, ecx..)")
    ap.add_argument("--robbins", required=True, help="Robbins 2018 catalog CSV")
    ap.add_argument("--regions", default="", help="comma list; default all")
    ap.add_argument("--angles", default="0,10,20,30,40,50,60")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--min-det-px", type=float, default=13.0)
    ap.add_argument("--min-cat-km", type=float, default=1.5)
    ap.add_argument("--dem", default=None, help="LDEM_64 path (omit to use a sphere)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dem:
        os.environ["LDEM_PATH"] = args.dem
        use_dem = dem
    else:
        use_dem = None

    regions = args.regions.split(",") if args.regions else common.REGIONS
    angles = [int(a) for a in args.angles.split(",")]
    rows, sanity = [], []

    for region in regions:
        P = common.load_catalog(args.robbins, args.base, region, args.min_cat_km, dem=use_dem)
        for angle in angles:
            dets = common.dets_for(args.dets, region, angle)
            n_img = 0
            for idx, (centers, sizes, _) in sorted(dets.items()):
                if n_img >= args.max_images:
                    break
                keep = sizes >= args.min_det_px
                if keep.sum() < 6:
                    continue
                uv = centers[keep]
                A_gt, C_gt = common.abs_pose(args.base, region, angle, idx)
                uv_pred, ok = common.project(P, A_gt, C_gt)
                inf = ok & (uv_pred[:, 0] > 0) & (uv_pred[:, 0] < 1024) & \
                    (uv_pred[:, 1] > 0) & (uv_pred[:, 1] < 1024)
                pairs = common.mutual_match(uv_pred[inf], uv, 10.0)
                if len(pairs) < 6:
                    continue
                obj = P[inf][[p for p, _ in pairs]].astype(np.float64)
                img = uv[[d for _, d in pairs]].astype(np.float64)

                if n_img < 3:
                    pred_pts = uv_pred[inf][[p for p, _ in pairs]].astype(np.float64)
                    ok2, rvec2, tvec2 = cv2.solvePnP(obj, pred_pts, common.K, None,
                                                     flags=cv2.SOLVEPNP_SQPNP)
                    R2 = cv2.Rodrigues(rvec2)[0]
                    C2 = -R2.T @ tvec2.ravel()
                    cosang = np.clip((np.trace(R2 @ A_gt.T) - 1) / 2, -1, 1)
                    sanity.append(dict(trans_m=float(np.linalg.norm(C2 - C_gt)),
                                       rot_deg=float(np.degrees(np.arccos(cosang)))))

                okr, rvec, tvec, inl = cv2.solvePnPRansac(
                    obj, img, common.K, None, iterationsCount=2000,
                    reprojectionError=6.0, flags=cv2.SOLVEPNP_EPNP)
                if not okr or inl is None or len(inl) < 6:
                    continue
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnP(obj[inl], img[inl], common.K, None, rvec, tvec,
                                          useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)[1:]
                R_est = cv2.Rodrigues(rvec)[0]
                C_est = -R_est.T @ tvec.ravel()
                trans = float(np.linalg.norm(C_est - C_gt))
                cosang = np.clip((np.trace(R_est @ A_gt.T) - 1) / 2, -1, 1)
                rot = float(np.degrees(np.arccos(cosang)))
                rows.append(dict(region=region, angle=angle, idx=idx,
                                 n_match=len(pairs), n_inlier=int(len(inl)),
                                 trans_m=trans, rot_deg=rot))
                n_img += 1

    table = agg_table(rows)
    print(f"{'angle':>5} | {'n':>3} | {'matches':>7} | {'trans med (m)':>13} | {'rot med (deg)':>13}")
    for r in table:
        print(f"{r['angle']:5d} | {r['n']:3d} | {r['matches']:7.0f} | {r['trans_med']:13.1f} | {r['rot_med']:13.3f}")
    if sanity:
        print(f"sanity (exact projections): n={len(sanity)} "
              f"trans {np.median([s['trans_m'] for s in sanity]):.3f} m "
              f"rot {np.median([s['rot_deg'] for s in sanity]):.5f} deg")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(rows=rows, table=table, sanity=sanity), f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
