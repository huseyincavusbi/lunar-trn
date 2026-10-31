"""LOLA LDEM_64 elevation sampler.

Download the global DEM (507 MB) from the PDS Geosciences node:

    https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/
        data/lola_gdr/cylindrical/img/ldem_64.img

Format: 11,520 lines x 23,040 samples, int16 little-endian, simple cylindrical,
gridline registered; height = DN * 0.5 m above a reference sphere of 1737.4 km.

Point ``LDEM_PATH`` at the file, pass ``path=`` explicitly, or place it at
``data/lola/ldem_64.img`` relative to the working directory.
"""
import os
from pathlib import Path

import numpy as np

LINES, SAMPLES = 11520, 23040
PPD = 64.0
SCALE = 0.5
BASE_RADIUS = 1737400.0

_cache = {}


def _resolve(path=None):
    return Path(path or os.environ.get("LDEM_PATH", "data/lola/ldem_64.img"))


def load(path=None):
    p = _resolve(path)
    if p not in _cache:
        _cache[p] = np.memmap(p, dtype="<i2", mode="r", shape=(LINES, SAMPLES))
    return _cache[p]


def elevation(lat, lon, path=None):
    """Bilinear elevation (m, relative to the 1737.4 km sphere)."""
    row = (90.0 - np.asarray(lat, dtype=float)) * PPD - 0.5
    col = np.mod(np.asarray(lon, dtype=float), 360.0) * PPD - 0.5
    row = np.clip(row, 0, LINES - 1.001)
    col = np.clip(col, 0, SAMPLES - 1.001)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    fr = row - r0
    fc = col - c0
    d = load(path)
    v00 = d[r0, c0].astype(np.float64)
    v01 = d[r0, np.minimum(c0 + 1, SAMPLES - 1)].astype(np.float64)
    v10 = d[np.minimum(r0 + 1, LINES - 1), c0].astype(np.float64)
    v11 = d[np.minimum(r0 + 1, LINES - 1), np.minimum(c0 + 1, SAMPLES - 1)].astype(np.float64)
    val = (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc +
           v10 * fr * (1 - fc) + v11 * fr * fc)
    return val * SCALE


def radius(lat, lon, path=None):
    return BASE_RADIUS + elevation(lat, lon, path)


def rim_elevation(lat, lon, diam_km, path=None, n=8):
    """Median elevation on a ring at radius D/2 (the crater rim the detector sees)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    d = np.asarray(diam_km, dtype=float)
    vals = []
    for ang in np.linspace(0, 2 * np.pi, n, endpoint=False):
        dlat = (d / 2.0) / 30.34 * np.cos(ang)
        dlon = (d / 2.0) / (30.34 * np.maximum(np.cos(np.radians(lat)), 0.1)) * np.sin(ang)
        vals.append(elevation(lat + dlat, lon + dlon, path))
    return np.median(np.vstack(vals), axis=0)
