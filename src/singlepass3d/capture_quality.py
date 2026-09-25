"""Fast preflight diagnostics for single-pass reconstruction footage."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from .core import ensure_output


def assess_capture(images: list[Path], output: Path, max_pairs: int = 24) -> dict:
    """Measure feature support, image motion, and planar/pure-rotation risk."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Capture diagnostics require pip install '.[video]'") from exc
    if len(images) < 3:
        raise ValueError("Capture diagnostics require at least three frames")
    indices = np.linspace(0, len(images) - 2, min(max_pairs, len(images) - 1), dtype=int)
    detector = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows: list[dict] = []
    for index in sorted({int(i) for i in indices}):
        left = cv2.imread(str(images[index]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(images[index + 1]), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            continue
        scale = min(1.0, 1280.0 / max(left.shape))
        if scale < 1:
            left = cv2.resize(left, None, fx=scale, fy=scale)
            right = cv2.resize(right, None, fx=scale, fy=scale)
        key_a, desc_a = detector.detectAndCompute(left, None)
        key_b, desc_b = detector.detectAndCompute(right, None)
        if desc_a is None or desc_b is None:
            rows.append({"left": images[index].name, "right": images[index + 1].name, "matches": 0})
            continue
        pairs = matcher.knnMatch(desc_a, desc_b, k=2)
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        row = {"left": images[index].name, "right": images[index + 1].name, "matches": len(good)}
        if len(good) >= 12:
            source = np.float32([key_a[m.queryIdx].pt for m in good])
            target = np.float32([key_b[m.trainIdx].pt for m in good])
            flow = np.linalg.norm(target - source, axis=1)
            diagonal = float(np.hypot(left.shape[1], left.shape[0]))
            row["median_flow_fraction"] = float(np.median(flow) / diagonal)
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
            row["homography_inlier_fraction"] = float(mask.mean()) if mask is not None else None
        rows.append(row)
    usable = [row for row in rows if row.get("matches", 0) >= 12]
    flows = [row["median_flow_fraction"] for row in usable if "median_flow_fraction" in row]
    homographies = [row["homography_inlier_fraction"] for row in usable if row.get("homography_inlier_fraction") is not None]
    median_matches = median([row["matches"] for row in rows]) if rows else 0
    median_flow = median(flows) if flows else 0.0
    homography_dominance = median(homographies) if homographies else 1.0
    high_homography_fraction = (
        sum(value > 0.90 for value in homographies) / len(homographies)
        if homographies else 1.0
    )
    warnings = []
    if len(usable) < max(2, len(rows) // 3):
        warnings.append("Too few frame pairs have stable visual correspondences.")
    if median_matches < 80:
        warnings.append("Low feature support may prevent reliable camera registration.")
    if median_flow < 0.003:
        warnings.append("Camera motion is very small; add forward or sideways translation.")
    if homography_dominance > 0.90 or high_homography_fraction >= 0.20:
        warnings.append("A substantial part of the flight is homography-dominated, consistent with panning, pure rotation, or a mostly planar scene; recovered depth may be warped.")
    result = {"assessment": "warning" if warnings else "suitable", "pairs_tested": len(rows), "usable_pairs": len(usable), "median_good_matches": float(median_matches), "median_flow_fraction_of_diagonal": float(median_flow), "median_homography_inlier_fraction": float(homography_dominance), "high_homography_pair_fraction": float(high_homography_fraction), "warnings": warnings, "interpretation": "Heuristic capture preflight only. Metric validity requires GPS alignment and independent checkpoints.", "pairs": rows}
    destination = ensure_output(output) / "capture_quality.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

