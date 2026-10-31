"""Shared utilities for CRESENT pose experiments.

Conventions
-----------
* ``abs_pose`` returns ``A`` = world->camera rotation (v = A @ (P - C)) and the
  camera position ``C``.  It reconstructs the pose from the images' own geometry:
  the 0 deg view looks down the local vertical with image x = lunar east and
  y = south (calibrated against catalogue projections); relative rotations
  between angles come from the pose Euler angles (ZXY convention, validated by
  sibling-image warping at correlation 0.993).
* ``project`` uses the same world->camera convention.
* ``ground_dirs`` returns world unit rays for image points (camera ray -> world).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["LDEM_-90_-45E_0_45N", "LDEM_0_45E_135_180N",
           "LDEM_0_45E_270_315N", "LDEM_45_90E_135_180N"]
ANGLES = [0, 10, 20, 30, 40, 50, 60]
R_MOON = 1737400.0
F = 512.0 / math.tan(math.radians(30.0))
K = np.array([[F, 0, 512.0], [0, F, 512.0], [0, 0, 1.0]])
K_INV = np.linalg.inv(K)


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    return Rz(y) @ Rx(p) @ Ry(r)


def load_poses(base, region, angle):
    path = Path(base) / region / f"{region}_float_60fov_1024_1024_{angle}deg_off_nadir" / "poses.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def region_bounds(base, region):
    path = Path(base) / region / f"{region}_float_60fov_1024_1024_0deg_off_nadir" / "LDEM_data.txt"
    vals = [float(x) for x in path.read_text().strip().splitlines()[-1].split(",")]
    return dict(max_lat=vals[0], min_lat=vals[1], west=vals[2], east=vals[3])


def load_catalog(robbins_csv, base, region, min_km=2.0, dem=None):
    """Catalogue craters in a CRESENT region as 3D positions (N,3)."""
    b = region_bounds(base, region)
    df = pd.read_csv(robbins_csv)
    lon = df["LON_CIRC_IMG"] % 360.0
    m = ((df["LAT_CIRC_IMG"] >= b["min_lat"]) & (df["LAT_CIRC_IMG"] <= b["max_lat"]) &
         (lon >= b["west"]) & (lon <= b["east"]) & (df["DIAM_CIRC_IMG"] >= min_km))
    sub = df[m]
    lat = np.radians(sub["LAT_CIRC_IMG"].values)
    lonr = np.radians((sub["LON_CIRC_IMG"].values % 360.0))
    r = np.full(len(sub), R_MOON)
    if dem is not None:
        try:
            r = R_MOON + dem.rim_elevation(sub["LAT_CIRC_IMG"].values,
                                           sub["LON_CIRC_IMG"].values % 360.0,
                                           sub["DIAM_CIRC_IMG"].values)
        except Exception as exc:  # DEM missing -> sphere fallback
            print("WARNING: DEM unavailable:", exc)
    P = np.stack([r * np.cos(lat) * np.cos(lonr),
                  r * np.cos(lat) * np.sin(lonr),
                  r * np.sin(lat)], axis=1)
    return P


def abs_pose(base, region, angle, idx):
    """Reconstructed ground-truth pose: (A world->camera, C camera position)."""
    p0 = load_poses(base, region, 0).iloc[idx]
    pt = load_poses(base, region, angle).iloc[idx]
    C = np.array([p0["X"], p0["Y"], p0["Z"]], dtype=float)
    u = C / np.linalg.norm(C)
    east = np.cross([0.0, 0.0, 1.0], u)
    east /= np.linalg.norm(east)
    north = np.cross(u, east)
    A0 = np.column_stack([east, -north, -u])          # camera -> world at 0 deg
    M0 = euler(p0["yaw"], p0["pitch"], p0["roll"])
    Mt = euler(pt["yaw"], pt["pitch"], pt["roll"])
    Rh = Mt.T @ M0
    At = A0 @ Rh.T                                    # camera -> world at angle
    return At.T, C                                    # world -> camera


def project(P, A, C):
    """Project 3D points with world->camera A and camera position C."""
    v = (A @ (P - C).T).T
    ok = v[:, 2] > 1e-6
    uv = np.full((len(P), 2), -1.0)
    uv[ok, 0] = 512.0 + F * v[ok, 0] / v[ok, 2]
    uv[ok, 1] = 512.0 + F * v[ok, 1] / v[ok, 2]
    return uv, ok


def ground_dirs(uv, A):
    """World unit rays for image points (camera ray -> world)."""
    d = A.T @ K_INV @ np.hstack([uv, np.ones((len(uv), 1))]).T
    return (d / np.linalg.norm(d, axis=0, keepdims=True)).T


def predicted_ratios(P, C):
    """Apparent axis ratio of a circular crater on the local surface."""
    v = P - C
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    n = P / np.linalg.norm(P, axis=1, keepdims=True)
    return np.clip(-(v * n).sum(axis=1), 0.0, 1.0)


def dets_for(dets_csv, region, angle, conf=0.25):
    """Detections per image index: (centers (N,2), sizes px (N,), ratios (N,))."""
    df = pd.read_csv(dets_csv)
    df = df[(df.region == region) & (df.angle == angle) & (df.conf >= conf)]
    out = {}
    for idx, g in df.groupby("image_index"):
        if "ecx" in g.columns and g.ecx.notna().all():
            c = np.stack([g.ecx.values, g.ecy.values], axis=1)
            s = 2 * np.maximum(g.a.values, g.b.values)
            r = (g.b / g.a.clip(lower=1e-6)).values
        else:
            c = np.stack([(g.x1 + g.x2).values / 2, (g.y1 + g.y2).values / 2], axis=1)
            s = np.maximum((g.x2 - g.x1).values, (g.y2 - g.y1).values)
            r = np.ones(len(g))
        out[int(idx)] = (c, s, r)
    return out


def mutual_match(pred, dets, tol):
    """Mutual nearest-neighbour matches between predicted and detected points."""
    if len(pred) == 0 or len(dets) == 0:
        return []
    d = np.sqrt(((pred[:, None] - dets[None]) ** 2).sum(axis=2))
    nn_p = d.argmin(axis=1)
    nn_d = d.argmin(axis=0)
    return [(int(pi), int(nn_p[pi])) for pi in range(len(pred))
            if nn_d[nn_p[pi]] == pi and d[pi, nn_p[pi]] <= tol]
