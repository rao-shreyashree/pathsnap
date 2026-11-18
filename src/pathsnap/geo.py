"""Geographical and geometric utilities for PathSnap.

Implements:
- Local equirectangular metric projection centered at trip centroid.
- Vectorized point-to-segment projection and perpendicular distance.
- Haversine geodesic distance for ground-truth validation.
- Heading / bearing calculation.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6371000.0


def get_centroid(lons: np.ndarray, lats: np.ndarray) -> tuple[float, float]:
    """Compute centroid (lon0, lat0) in degrees."""
    return float(np.mean(lons)), float(np.mean(lats))


def project_equirectangular(
    lons: np.ndarray,
    lats: np.ndarray,
    lon0: float,
    lat0: float,
    radius: float = EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Project spherical coordinates (lon, lat) to local metric Cartesian coordinates (x, y) in metres.
    
    Formula:
      x = R * (lon - lon0) * cos(lat0) * (pi / 180)
      y = R * (lat - lat0) * (pi / 180)
    """
    lat0_rad = np.radians(lat0)
    dlon_rad = np.radians(lons - lon0)
    dlat_rad = np.radians(lats - lat0)
    x = radius * dlon_rad * np.cos(lat0_rad)
    y = radius * dlat_rad
    return x, y


def unproject_equirectangular(
    x: np.ndarray,
    y: np.ndarray,
    lon0: float,
    lat0: float,
    radius: float = EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local metric coordinates (x, y) back to spherical (lon, lat) in degrees."""
    lat0_rad = np.radians(lat0)
    dlat_rad = y / radius
    dlon_rad = x / (radius * np.cos(lat0_rad))
    lats = lat0 + np.degrees(dlat_rad)
    lons = lon0 + np.degrees(dlon_rad)
    return lons, lats


def haversine_distance(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Compute great-circle distance between two points in metres using Haversine formula."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def point_to_segment_distance(
    px: float | np.ndarray,
    py: float | np.ndarray,
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate perpendicular distance from point(s) P(px, py) to line segment(s) S((x1,y1) -> (x2, y2)).
    
    Returns:
      dist: perpendicular distance in metres
      t: fraction along segment in [0, 1]
      qx: projected point x on segment
      qy: projected point y on segment
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    # Avoid division by zero for zero-length segments / self-loops
    zero_len = seg_len_sq < 1e-12
    seg_len_sq_safe = np.where(zero_len, 1.0, seg_len_sq)

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq_safe
    t = np.clip(t, 0.0, 1.0)
    t = np.where(zero_len, 0.0, t)

    qx = x1 + t * dx
    qy = y1 + t * dy

    dist = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
    return dist, t, qx, qy


def compute_bearing(
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
) -> np.ndarray:
    """Compute direction angle in degrees [0, 360) from (x1, y1) to (x2, y2) in local metric projection.
    
    0 deg = North (+y direction), 90 deg = East (+x direction),
    180 deg = South (-y direction), 270 deg = West (-x direction).
    """
    dx = x2 - x1
    dy = y2 - y1
    # angle from +y axis clockwise
    angle = np.degrees(np.arctan2(dx, dy))
    return (angle + 360.0) % 360.0


def compute_track_bearings(x: np.ndarray, y: np.ndarray, speeds: np.ndarray | None = None) -> np.ndarray:
    """Compute local instantaneous vehicle heading for each point in trajectory.
    
    If vehicle is stationary/low speed (< 1.0 m/s), heading defaults to the closest valid heading.
    """
    n = len(x)
    bearings = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return bearings

    # Forward difference for first, backward difference for last, central difference for middle
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)

    dx[0] = x[1] - x[0]
    dy[0] = y[1] - y[0]
    dx[-1] = x[-1] - x[-2]
    dy[-1] = y[-1] - y[-2]

    if n > 2:
        dx[1:-1] = x[2:] - x[:-2]
        dy[1:-1] = y[2:] - y[:-2]

    raw_bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    # If speed gating is provided, forward-fill heading during stops
    if speeds is not None:
        last_valid = raw_bearings[0]
        for i in range(n):
            if speeds[i] >= 1.0:
                last_valid = raw_bearings[i]
            bearings[i] = last_valid
    else:
        bearings = raw_bearings

    return bearings
