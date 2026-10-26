#!/usr/bin/env python3
"""Run a YOLO model on C1 images and write challenge-format ellipse CSV.

The task is detected from the checkpoint:
  * box models  -> axis-aligned ellipse proxy (rotation 0, axes from the box)
  * seg models  -> cv2.fitEllipse on instance masks (rotation-aware)

Example:
    python src/predict.py --model runs/detect/runs/full-v1s-seg/weights/best.pt \
        --images data/c1-full-seg/images/val \
        --out preds/seg.csv --conf 0.05 --imgsz 1024 --conf-out preds/seg-debug.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from common.geometry import fit_ellipse_from_mask

COLS = ["ellipseCenterX(px)", "ellipseCenterY(px)", "ellipseSemimajor(px)",
        "ellipseSemiminor(px)", "ellipseRotation(deg)", "inputImage", "crater_classification"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf-out", default=None,
                    help="optional sidecar CSV with per-detection confidence")
    args = ap.parse_args()

    model = YOLO(args.model)
    is_seg = model.task == "segment"

    rows, debug = [], []
    imgs = sorted(Path(args.images).glob("*.png"))
    for img in imgs:
        res = model.predict(str(img), conf=args.conf, imgsz=args.imgsz,
                            retina_masks=is_seg, verbose=False)[0]
        img_id = img.stem.replace("___", "/")
        n = 0 if res.boxes is None else len(res.boxes)
        if n == 0:
            rows.append(dict(zip(COLS, [-1, -1, -1, -1, -1, img_id, -1])))
            continue

        confs = res.boxes.conf.cpu().numpy()
        if is_seg:
            masks = res.masks.data.cpu().numpy() if res.masks is not None else []
            for mask, conf in zip(masks, confs):
                ellipse = fit_ellipse_from_mask(mask)
                if ellipse is None:
                    continue
                cx, cy, a, b, phi = ellipse
                row = dict(zip(COLS, [cx, cy, a, b, phi, img_id, -1]))
                rows.append(row)
                debug.append({**row, "conf": float(conf)})
        else:
            for (cx, cy, w, h), conf in zip(res.boxes.xywh.cpu().numpy(), confs):
                row = dict(zip(COLS, [float(cx), float(cy), float(max(w, h) / 2),
                                      float(min(w, h) / 2), 0.0, img_id, -1]))
                rows.append(row)
                debug.append({**row, "conf": float(conf)})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.conf_out:
        pd.DataFrame(debug).to_csv(args.conf_out, index=False)
    print(f"wrote {args.out}: {len(rows)} rows from {len(imgs)} images")


if __name__ == "__main__":
    main()
