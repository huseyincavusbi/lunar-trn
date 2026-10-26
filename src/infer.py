#!/usr/bin/env python3
"""Dump YOLO detections for evaluation imagery.

Datasets
--------
lu3m6tgt : flat directory of georeferenced patches (--images)
           output columns: image, conf, x1..y2 [, ecx, ecy, a, b, phi]
crescent : CRESENT tree (--base) with region/angle folders
           output columns: region, angle, image, image_index, conf, x1..y2 [, ellipse...]

Examples
--------
python src/infer.py --dataset lu3m6tgt --model runs/detect/runs/v1s-ft-real/weights/best.pt \
    --images data/lu3m6tgt/extracted/LU3M6TGT_yolo_format/valid/images \
    --out preds/ft.csv --imgsz 1024 --conf 0.05

python src/infer.py --dataset crescent --model runs/segment/runs/full-v1s-seg/weights/best.pt \
    --base data/cresent/extracted/CRESENT --out preds/crescent.csv --per-angle 80
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from common.geometry import fit_ellipse_from_mask

REGIONS = ["LDEM_-90_-45E_0_45N", "LDEM_0_45E_135_180N",
           "LDEM_0_45E_270_315N", "LDEM_45_90E_135_180N"]
BATCH_BY_IMGSZ = {416: 12, 1024: 8, 1536: 6, 2048: 4, 2560: 1}


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def predict_rows(model, paths, imgsz, conf):
    """Return a list of (path, row_dict_base) detections for one batch group."""
    is_seg = model.task == "segment"
    out = []
    batch_size = BATCH_BY_IMGSZ.get(imgsz, 8)
    for batch in chunks(paths, batch_size):
        results = model.predict([str(p) for p in batch], conf=conf, imgsz=imgsz,
                                retina_masks=is_seg, verbose=False)
        for path, res in zip(batch, results):
            n = 0 if res.boxes is None else len(res.boxes)
            if n == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ellipses = [None] * n
            if is_seg and res.masks is not None and len(res.masks) > 0:
                for i, mask in enumerate(res.masks.data.cpu().numpy()):
                    ellipses[i] = fit_ellipse_from_mask(mask)
            for i in range(n):
                row = dict(conf=float(confs[i]),
                           x1=float(xyxy[i][0]), y1=float(xyxy[i][1]),
                           x2=float(xyxy[i][2]), y2=float(xyxy[i][3]))
                e = ellipses[i]
                if e is not None:
                    row.update(ecx=e[0], ecy=e[1], a=e[2], b=e[3], phi=e[4])
                out.append((path, row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["lu3m6tgt", "crescent"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.05)

    ap.add_argument("--images", default=None, help="lu3m6tgt: patch directory")
    ap.add_argument("--base", default="data/cresent/extracted/CRESENT",
                    help="crescent: dataset root")
    ap.add_argument("--regions", default="", help="crescent: comma-separated region folders")
    ap.add_argument("--angles", default="0,10,20,30,40,50,60")
    ap.add_argument("--per-angle", type=int, default=80,
                    help="crescent: max images per region/angle")
    args = ap.parse_args()

    model = YOLO(args.model)
    rows = []

    if args.dataset == "lu3m6tgt":
        if not args.images:
            raise SystemExit("error: --images is required for lu3m6tgt")
        paths = sorted(Path(args.images).glob("*.png"))
        for path, row in predict_rows(model, paths, args.imgsz, args.conf):
            rows.append(dict(image=path.name, **row))
    else:
        regions = args.regions.split(",") if args.regions else REGIONS
        angles = [int(a) for a in args.angles.split(",")]
        for region in regions:
            for angle in angles:
                folder = Path(args.base) / region / f"{region}_float_60fov_1024_1024_{angle}deg_off_nadir" / "images"
                if not folder.exists():
                    print("missing", folder)
                    continue
                paths = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))[:args.per_angle]
                batch_rows = predict_rows(model, paths, args.imgsz, args.conf)
                for path, row in batch_rows:
                    rows.append(dict(region=region, angle=angle, image=path.name,
                                     image_index=int(path.stem), **row))
                print(f"{region} {angle}deg done ({len(paths)} imgs, dets so far {len(rows)})")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} dets")


if __name__ == "__main__":
    main()
