import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from singlepass3d.viewer import create_handler, mission_summary


def test_summary_and_safe_server(tmp_path):
    mission = tmp_path / "mission"
    (mission / "geospatial").mkdir(parents=True)
    (mission / "geospatial/sparse_georeferenced.ply").write_text("ply", encoding="ascii")
    viewer = tmp_path / "viewer"; viewer.mkdir()
    (viewer / "index.html").write_text("viewer", encoding="utf-8")
    summary = mission_summary(mission)
    assert "sparse" in summary["layers"]
    assert summary["reconstruction_level"] == "sparse_point_cloud"
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(mission, viewer))
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        base = "http://127.0.0.1:" + str(server.server_port)
        assert urllib.request.urlopen(base + "/", timeout=2).read() == b"viewer"
        assert json.load(urllib.request.urlopen(base + "/api/mission", timeout=2))["name"] == "mission"
        try:
            urllib.request.urlopen(base + "/mission/../viewer/index.html", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code in {403, 404}
        else:
            raise AssertionError("Traversal request unexpectedly succeeded")
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_packaged_viewer_includes_metric_measurement():
    asset = Path(__file__).parents[1] / "src/singlepass3d/viewer_assets/viewer.js"
    source = asset.read_text(encoding="utf-8-sig")
    assert "Measured distance:" in source
    assert "worldScale" in source


def test_viewer_does_not_invent_mesh_from_sparse_points():
    asset = Path(__file__).parents[1] / "src/singlepass3d/viewer_assets/viewer.js"
    source = asset.read_text(encoding="utf-8-sig")
    assert "auto-generate high-quality solid surface" not in source
    assert "SPARSE CAMERA GEOMETRY" in source
