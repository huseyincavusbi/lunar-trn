#!/usr/bin/env python3
"""Phase 2 scorer: detections on LU3M6TGT valid -> AP (IoU .5:.95), size-band recall/AP,
Robbins-2018 anchor audit (recall >=2 km, FP candidates, GT label coverage), overlay images."""
import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

S_PER_BUCKET = 1.0  # degrees, spatial hash bucket for patch lookup
BANDS = [(0.0, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
IOU_THRS = [0.5 + 0.05 * i for i in range(10)]
KM_PER_PX = 0.1  # lat direction: 1.3718 deg = 41.62 km over 416 px


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
    for v in pred.values():
        v.sort(key=lambda x: -x[0])
    return dict(pred)


def iou_matrix(dets, gts):
    d = np.asarray(dets, dtype=float).reshape(-1, 4)
    g = np.asarray(gts, dtype=float).reshape(-1, 4)
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    x1 = np.maximum(d[:, None, 0], g[None, :, 0])
    y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 2], g[None, :, 2])
    y2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    da = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (da[:, None] + ga[None, :] - inter + 1e-12)


def match_at_iou(gt_boxes, dets):
    """Greedy matching (dets sorted by conf desc). Returns matched_gt (bool list per gt)."""
    matched = [False] * len(gt_boxes)
    m = iou_matrix([d[1:] for d in dets], gt_boxes)
    for i in range(len(dets)):
        order = np.argsort(-m[i])
        for j in order:
            if m[i, j] >= 0.5 and not matched[j]:
                matched[j] = True
                break
    return matched


def voc_ap(gt, pred, iou_thr, gt_filter=None, pred_filter=None):
    """VOC all-point AP. Filters (COCO-style area range) restrict BOTH GT positives and detections."""
    npos = 0
    gt_used = {}
    for img, boxes in gt.items():
        keep = [b for b in boxes if (gt_filter is None or gt_filter(b))]
        gt_used[img] = keep
        npos += len(keep)
    if npos == 0:
        return float("nan")
    all_dets = []
    for img, dets in pred.items():
        for d in dets:
            if pred_filter is not None and not pred_filter(d[1:]):
                continue
            all_dets.append((d[0], img, d[1:]))
    all_dets.sort(key=lambda x: -x[0])
    matched = {img: [False] * len(boxes) for img, boxes in gt_used.items()}
    tp, fp = [], []
    for conf, img, box in all_dets:
        boxes = gt_used.get(img, [])
        hit = False
        if boxes:
            m = iou_matrix([box], boxes)[0]
            order = np.argsort(-m)
            for j in order:
                if m[j] >= iou_thr and not matched[img][j]:
                    matched[img][j] = True
                    hit = True
                    break
        tp.append(1 if hit else 0)
        fp.append(0 if hit else 1)
    if not tp:
        return 0.0
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-12)
    # precision envelope
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    prev_r = 0.0
    for i in range(len(rec)):
        if rec[i] > prev_r:
            ap += (rec[i] - prev_r) * prec[i]
            prev_r = rec[i]
    return float(ap)


def build_robbins(robbins_csv, patch_names, cache_path):
    """Project Robbins craters into each patch. Returns {patch_stem: [(x, y, D_km)]}."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    patches = {}
    for n in patch_names:
        p = n.split(",")
        patches[n] = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    warehouse = defaultdict(list)
    for n, (l0, l1, b0, b1) in patches.items():
        l0n, l1n = l0 % 360.0, l1 % 360.0
        lon_ranges = [(l0n, l1n)] if l0n < l1n else [(l0n, 360.0), (0.0, l1n)]
        for a, b in lon_ranges:
            for i in range(int(a // S_PER_BUCKET), int(b // S_PER_BUCKET) + 1):
                for j in range(int(b0 // S_PER_BUCKET), int(b1 // S_PER_BUCKET) + 1):
                    warehouse[(j, i)].append(n)
    res = defaultdict(list)
    with open(robbins_csv) as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["LAT_CIRC_IMG"]); lon = float(row["LON_CIRC_IMG"]) % 360.0
                d = float(row["DIAM_CIRC_IMG"])
            except (KeyError, ValueError):
                continue
            key = (int(lat // S_PER_BUCKET), int(lon // S_PER_BUCKET))
            for n in warehouse.get(key, []):
                l0, l1, b0, b1 = patches[n]
                if not (b0 <= lat <= b1):
                    continue
                l0n, l1n = l0 % 360.0, l1 % 360.0
                if l0n < l1n:
                    if not (l0n <= lon <= l1n):
                        continue
                    x = (lon - l0n) / (l1n - l0n) * 416.0
                else:
                    b2 = l1n + 360.0
                    ll = lon if lon >= l0n else lon + 360.0
                    if not (l0n <= ll <= b2):
                        continue
                    x = (ll - l0n) / (b2 - l0n) * 416.0
                y = (b1 - lat) / (b1 - b0) * 416.0
                if 0.0 <= x <= 416.0 and 0.0 <= y <= 416.0:
                    res[n].append((x, y, d))
    with open(cache_path, "wb") as f:
        pickle.dump(dict(res), f)
    return dict(res)


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def rob_metrics(gt, pred, rob, conf_op):
    """Robbins >=2 km recall, FP candidates, GT label coverage."""
    rec_hit = 0; rec_n = 0
    cov_hit = 0
    fp_cand = 0; label_only = 0; rob_matched = 0
    fp_examples = []
    for stem, craters in rob.items():
        dets = [d for d in pred.get(stem, []) if d[0] >= conf_op]
        gts = gt.get(stem, [])
        for (rx, ry, rd) in craters:
            if rd < 2.0:
                continue
            rec_n += 1
            r = rd * 5.0  # px radius: D_km*10 px diameter / 2
            hit = any((box_center(d[1:])[0] - rx) ** 2 + (box_center(d[1:])[1] - ry) ** 2 <= r * r for d in dets)
            if hit:
                rec_hit += 1
            if any((box_center(g)[0] - rx) ** 2 + (box_center(g)[1] - ry) ** 2 <= r * r for g in gts):
                cov_hit += 1
        for d in dets:
            b = d[1:]
            est_d = max(b[2] - b[0], b[3] - b[1]) * KM_PER_PX
            if est_d < 2.0:
                continue
            cx, cy = box_center(b)
            m_rob = any((cx - rx) ** 2 + (cy - ry) ** 2 <= max(rd * 5.0, est_d * 5.0) ** 2 for (rx, ry, rd) in craters)
            if m_rob:
                rob_matched += 1
                continue
            m_gt = any(iou_matrix([b], [g])[0][0] >= 0.5 for g in gts) if gts else False
            if m_gt:
                label_only += 1
            else:
                fp_cand += 1
                if len(fp_examples) < 20:
                    fp_examples.append(dict(image=stem, x=round(cx, 1), y=round(cy, 1),
                                            conf=round(d[0], 3), est_km=round(est_d, 2)))
    return dict(robbins_ge2km=rec_n, recall_ge2km=round(rec_hit / max(rec_n, 1), 4),
                gt_label_coverage_ge2km=round(cov_hit / max(rec_n, 1), 4),
                big_dets_matched_robbins=rob_matched, big_dets_label_only=label_only,
                big_dets_fp_candidates=fp_cand, fp_examples=fp_examples)


def make_overlays(pred, gt, rob, images_dir, out_dir, tag, conf_op, n=4):
    imgs = sorted(pred.keys())
    if not imgs:
        imgs = sorted(gt.keys())[:n]
    by_det = sorted(imgs, key=lambda s: -len(pred.get(s, [])))
    picks = []
    for s in by_det[:n]:
        if s not in picks:
            picks.append(s)
    for i, stem in enumerate(picks):
        ipath = os.path.join(images_dir, stem + ".png")
        if not os.path.exists(ipath):
            continue
        im = cv2.imread(ipath)
        up = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        for g in gt.get(stem, []):
            cv2.rectangle(up, (int(g[0] * 2), int(g[1] * 2)), (int(g[2] * 2), int(g[3] * 2)), (0, 255, 0), 1)
        for d in pred.get(stem, []):
            if d[0] < conf_op:
                continue
            b = d[1:]
            cv2.rectangle(up, (int(b[0] * 2), int(b[1] * 2)), (int(b[2] * 2), int(b[3] * 2)), (0, 0, 255), 1)
        for (rx, ry, rd) in rob.get(stem, []):
            if rd < 2.0:
                continue
            r = int(rd * 5.0 * 2)
            cv2.circle(up, (int(rx * 2), int(ry * 2)), r, (255, 128, 0), 1)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{stem[:60]}_{i}.png"), up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="preds/out")
    ap.add_argument("--robbins", default=None)
    ap.add_argument("--cache", default="preds/robbins_valid.pkl")
    ap.add_argument("--conf-op", type=float, default=0.25)
    ap.add_argument("--no-robbins", action="store_true")
    ap.add_argument("--overlay-n", type=int, default=4)
    args = ap.parse_args()

    gt = load_gt(args.gt_dir)
    pred = load_pred(args.pred)
    os.makedirs(args.out_dir, exist_ok=True)

    ap50 = voc_ap(gt, pred, 0.5)
    ap75 = voc_ap(gt, pred, 0.75)
    aps = [voc_ap(gt, pred, t) for t in IOU_THRS]
    aps = [a for a in aps if not np.isnan(a)]
    ap5095 = float(np.mean(aps)) if aps else float("nan")

    n_gt = sum(len(v) for v in gt.values())
    n_det = sum(len(v) for v in pred.values())
    n_det_op = sum(1 for v in pred.values() for d in v if d[0] >= args.conf_op)

    # size bands (GT size = max(w,h) in km, lat-true)
    bands = {}
    for lo, hi in BANDS:
        f = lambda b: lo <= max(b[2] - b[0], b[3] - b[1]) * KM_PER_PX < hi
        a50 = voc_ap(gt, pred, 0.5, gt_filter=f, pred_filter=f)
        npos = sum(1 for boxes in gt.values() for b in boxes if f(b))
        bands[f"{lo}-{hi}km"] = dict(n=npos, ap50=None if np.isnan(a50) else round(a50, 4))

    # band recall at operating conf (IoU 0.5, greedy)
    band_rec = {}
    for lo, hi in BANDS:
        m = t = 0
        for stem, boxes in gt.items():
            dets = [d for d in pred.get(stem, []) if d[0] >= args.conf_op]
            matched = match_at_iou(boxes, dets)
            for b, hit in zip(boxes, matched):
                if lo <= max(b[2] - b[0], b[3] - b[1]) * KM_PER_PX < hi:
                    t += 1
                    m += 1 if hit else 0
        band_rec[f"{lo}-{hi}km"] = dict(n=t, recall=round(m / max(t, 1), 4))

    summary = dict(tag=args.tag, imgs=len(gt), gt=sum(len(v) for v in gt.values()),
                   dets=n_det, dets_conf_op=n_det_op, conf_op=args.conf_op,
                   mAP50=round(ap50, 4), mAP75=round(ap75, 4), mAP50_95=round(ap5095, 4),
                   size_bands_ap50={k: v["ap50"] for k, v in bands.items()},
                   size_bands={k: v for k, v in bands.items()},
                   band_recall_at_op=band_rec)

    if not args.no_robbins:
        rob = build_robbins(args.robbins, list(gt.keys()), args.cache)
        summary["robbins"] = rob_metrics(gt, pred, rob, args.conf_op)
        make_overlays(pred, gt, rob, args.images_dir, os.path.join(args.out_dir, "overlays"),
                      args.tag, args.conf_op, args.overlay_n)
    else:
        rob = {}

    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "robbins"}))
    if "robbins" in summary:
        r = summary["robbins"]
        print(f"RESULT {args.tag}: mAP50={summary['mAP50']} mAP50-95={summary['mAP50_95']} "
              f"dets={n_det} | rob>=2km recall={r['recall_ge2km']} (n={r['robbins_ge2km']}) "
              f"gt_cov={r['gt_label_coverage_ge2km']} fp_cand={r['big_dets_fp_candidates']} "
              f"label_only={r['big_dets_label_only']}")


if __name__ == "__main__":
    main()
