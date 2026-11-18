import numpy as np
from pyproj import Transformer


def make_projector(lon, lat):
    """Metric CRS (UTM zone of the data centre). Returns (transformer, epsg)."""
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    zone = int((lon0 + 180) // 6) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True), epsg


def bearing_deg(dx, dy):
    """Compass bearing (0=north, clockwise) of a vector in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angdiff(a, b):
    """Smallest absolute angle between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))
