import json

import numpy as np

from singlepass3d.geoexport import align_sparse, write_ply


def test_ply_export(tmp_path):
    source = tmp_path / "points3D.txt"
    source.write_text("# points\n1 1 2 3 255 128 0 0.1 1 0\n", encoding="utf-8")
    output = tmp_path / "points.ply"
    assert write_ply(source, output, lambda points: points * 2) == 1
    assert "2.0 4.0 6.0 255 128 0" in output.read_text(encoding="ascii")


def test_align_artifacts(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    lines = ["# images"]
    for index in range(1, 5):
        lines += [f"{index} 1 0 0 0 {-index} {-index * index} 0 1 frame_{index}.jpg", "0 0 -1"]
    (model / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (model / "points3D.txt").write_text("1 0 0 0 1 2 3 0.1\n", encoding="utf-8")
    sync = tmp_path / "sync.csv"
    sync.write_text("filename,matched,latitude,longitude,altitude\n" +
                    "".join(f"frame_{i}.jpg,True,{i},{i * i},0\n" for i in range(1, 5)),
                    encoding="utf-8")
    monkeypatch.setattr("singlepass3d.geoexport.wgs84_to_enu",
                        lambda lat, lon, alt, origin: np.column_stack((lat * 2, lon * 2, lat * 0)))
    monkeypatch.setattr("singlepass3d.geoexport.enu_to_wgs84",
                        lambda points, origin: points)
    metrics = align_sparse(model, sync, tmp_path / "out")
    assert metrics["observations"] == 4
    assert metrics["sfm_observations"] == 0
    assert metrics["sparse_points"] == 1
    assert (tmp_path / "out/trajectory.geojson").exists()
    assert json.loads((tmp_path / "out/reference.json").read_text())["scale"] == 2
