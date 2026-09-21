"""Local metric coordinates and robust visual-to-GPS similarity alignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Alignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    residuals: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points) @ self.rotation.T) + self.translation

    def metrics(self) -> dict[str, float | int]:
        errors = self.residuals[self.inliers]
        return {
            "observations": len(errors),
            "mean_residual_m": float(np.mean(errors)),
            "median_residual_m": float(np.median(errors)),
            "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "max_residual_m": float(np.max(errors)),
            "scale": self.scale,
        }


def wgs84_to_enu(latitudes: np.ndarray, longitudes: np.ndarray, altitudes: np.ndarray,
                 origin: tuple[float, float, float]) -> np.ndarray:
    """Transform WGS84 degrees/meters to a local East-North-Up metric frame."""
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("Geospatial alignment requires pip install '.[geospatial]'") from exc
    lat0, lon0, alt0 = origin
    ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = ecef.transform(longitudes, latitudes, altitudes)
    x0, y0, z0 = ecef.transform(lon0, lat0, alt0)
    radians = np.deg2rad([lat0, lon0])
    slat, slon = np.sin(radians)
    clat, clon = np.cos(radians)
    rotation = np.array([
        [-slon, clon, 0],
        [-slat * clon, -slat * slon, clat],
        [clat * clon, clat * slon, slat],
    ])
    delta = np.stack((np.asarray(x) - x0, np.asarray(y) - y0, np.asarray(z) - z0), axis=-1)
    return delta @ rotation.T


def similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares uniform scale, proper rotation, translation."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Source and target must be matching Nx3 arrays")
    if len(source) < 3:
        raise ValueError("At least three positions are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    variance = np.sum(centered_source ** 2) / len(source)
    if variance < 1e-12 or np.linalg.matrix_rank(centered_source) < 2:
        raise ValueError("Camera positions have insufficient spatial variation")
    covariance = centered_target.T @ centered_source / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if scale <= 0:
        raise ValueError("Estimated scale is nonpositive")
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def robust_alignment(source: np.ndarray, target: np.ndarray, threshold_m: float = 5.0,
                     trials: int = 500, seed: int = 0) -> Alignment:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) < 3 or source.shape != target.shape or source.shape[1:] != (3,):
        raise ValueError("At least three matching 3D camera positions are required")
    if threshold_m <= 0 or trials < 1:
        raise ValueError("RANSAC threshold and trial count must be positive")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), dtype=bool)
    best_error = float("inf")
    for _ in range(trials):
        subset = rng.choice(len(source), size=3, replace=False)
        try:
            scale, rotation, translation = similarity(source[subset], target[subset])
        except ValueError:
            continue
        residuals = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
        inliers = residuals <= threshold_m
        error = float(np.sum(residuals[inliers])) if inliers.any() else float("inf")
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and error < best_error):
            best, best_error = inliers, error
    if best.sum() < 3:
        raise ValueError("GPS alignment found fewer than three inliers")
    scale, rotation, translation = similarity(source[best], target[best])
    residuals = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    return Alignment(scale, rotation, translation, best, residuals)


def enu_to_wgs84(points: np.ndarray, origin: tuple[float, float, float]) -> np.ndarray:
    """Transform local East-North-Up meters to WGS84 lon/lat/ellipsoidal height."""
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("Geospatial export requires pip install '.[geospatial]'") from exc
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ENU points must be an Nx3 array")
    lat0, lon0, alt0 = origin
    ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    inverse = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    x0, y0, z0 = ecef.transform(lon0, lat0, alt0)
    lat, lon = np.deg2rad([lat0, lon0])
    slat, slon = np.sin(lat), np.sin(lon)
    clat, clon = np.cos(lat), np.cos(lon)
    rotation = np.array([
        [-slon, clon, 0],
        [-slat * clon, -slat * slon, clat],
        [clat * clon, clat * slon, slat],
    ])
    delta = points @ rotation
    longitude, latitude, altitude = inverse.transform(
        delta[:, 0] + x0, delta[:, 1] + y0, delta[:, 2] + z0)
    return np.column_stack((longitude, latitude, altitude))
