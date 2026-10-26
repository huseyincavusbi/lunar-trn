#!/usr/bin/env python3
"""Config-driven YOLO trainer (detect or segment).

CLI arguments override values from --config. Example:

    python src/train.py --config configs/c1-seg.yaml
    python src/train.py --config configs/lu3m6tgt-real.yaml --epochs 5 --name ft-short

Config keys map directly to Ultralytics train() arguments, e.g.:
    data, model, epochs, imgsz, batch, device, project, name, seed,
    patience, workers, flipud, exist_ok
"""
import argparse
import multiprocessing
import sys

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--flipud", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=None,
                    help="fraction of the training set to use (debug/smoke runs)")
    ap.add_argument("--exist-ok", action="store_true", default=None)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value

    cfg.setdefault("model", "yolo26s.pt")
    cfg.setdefault("epochs", 15)
    cfg.setdefault("imgsz", 1024)
    cfg.setdefault("batch", 16)
    cfg.setdefault("device", 0)
    cfg.setdefault("project", "runs")
    cfg.setdefault("name", "train")
    cfg.setdefault("seed", 42)
    cfg.setdefault("workers", 4)
    cfg.setdefault("exist_ok", True)

    try:
        multiprocessing.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    if "data" not in cfg:
        sys.exit("error: --data is required (or set 'data:' in the config)")

    from ultralytics import YOLO

    weights = cfg.pop("model")
    model = YOLO(weights)
    model.train(**cfg)


if __name__ == "__main__":
    main()
