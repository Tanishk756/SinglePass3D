import json
import sys
import types

import numpy as np

from singlepass3d.masking import create_masks


class FakeSegmenter:
    model_id = "fake"
    def masks(self, images):
        return [np.array([[0, 1], [0, 0]], dtype=np.uint8) for _ in images]

def test_masks_are_colmap_usable_pixels(tmp_path, monkeypatch):
    written = {}
    fake_cv2 = types.SimpleNamespace(
        imwrite=lambda path, data: written.setdefault(path, data) is data)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    artifacts, manifest = create_masks(images, tmp_path / "masks", FakeSegmenter(), 1)
    assert [p.name for p in artifacts] == ["a.jpg.png", "b.jpg.png"]
    assert written[str(artifacts[0])].tolist() == [[255, 0], [255, 255]]
    data = json.loads(manifest.read_text())
    assert data["masks"][0]["dynamic_pixels"] == 1
