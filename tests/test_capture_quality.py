import json

import cv2
import numpy as np

from singlepass3d.capture_quality import assess_capture


def test_capture_quality_writes_factual_result(tmp_path):
    images = []
    rng = np.random.default_rng(4)
    base = (rng.random((360, 480)) * 255).astype("uint8")
    for index, shift in enumerate((0, 8, 16, 24)):
        image = np.roll(base, shift, axis=1)
        path = tmp_path / f"frame_{index:03d}.jpg"
        cv2.imwrite(str(path), image)
        images.append(path)
    result = assess_capture(images, tmp_path / "quality")
    stored = json.loads((tmp_path / "quality/capture_quality.json").read_text())
    assert result == stored
    assert result["pairs_tested"] == 3
    assert result["median_good_matches"] > 80
    assert result["assessment"] in {"suitable", "warning"}
