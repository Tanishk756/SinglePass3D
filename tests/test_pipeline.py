import json

from singlepass3d.pipeline import build_report


def test_report_uses_stage_values(tmp_path):
    (tmp_path / "video_info.json").write_text(
        json.dumps({"fps": 30, "width": 1920, "height": 1080,
                    "frame_count": 60, "duration_seconds": 2}), encoding="utf-8")
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_manifest.csv").write_text(
        "selected,rejection_reason\nTrue,\nFalse,blur\n", encoding="utf-8")
    model = tmp_path / "reconstruction/text_model"
    model.mkdir(parents=True)
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame.jpg\n0 0 -1\n", encoding="utf-8")
    (model / "points3D.txt").write_text("1 0 0 0 0 0 0 0.1\n", encoding="utf-8")
    geo = tmp_path / "geospatial"
    geo.mkdir()
    (geo / "metrics.json").write_text(
        json.dumps({"rmse_m": 1.2, "observations": 4}), encoding="utf-8")
    page = build_report(tmp_path)
    report = json.loads((tmp_path / "reports/metrics.json").read_text())
    assert report["frame_processing"]["blur_rejections"] == 1
    assert report["sfm"]["sparse_points"] == 1
    assert report["gps_alignment"]["rmse_m"] == 1.2
    assert "not absolute survey accuracy" in page.read_text()
