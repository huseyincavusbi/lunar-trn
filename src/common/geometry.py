"""Shared geometry utilities."""
import cv2
import numpy as np


def fit_ellipse_from_mask(mask, threshold=0.5):
    """Fit an ellipse to the largest contour of an instance mask.

    Args:
        mask: (H, W) array at original image resolution (float or bool).
        threshold: binarisation threshold.

    Returns:
        (cx, cy, a, b, phi_deg) with a >= b semi-axes and phi in [0, 180),
        or None if no valid contour exists. The convention matches the
        NASA Crater Detection Challenge scorer: axes are semi-axes and the
        rotation is the major-axis angle in [0, 180) degrees.
    """
    mm = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(contour)
    a, b = max(w, h) / 2.0, min(w, h) / 2.0
    phi = ang + 90.0 if h > w else ang
    return float(cx), float(cy), float(a), float(b), float(phi % 180.0)


def ellipse_polygon(cx, cy, a, b, phi_deg, n=48):
    """Sample a polygon around an ellipse (scorer convention, y-down pixels)."""
    phi = np.radians(phi_deg)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dx = a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    dy = a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    return np.stack([cx + dx, cy + dy], axis=1)
