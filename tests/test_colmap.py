from singlepass3d.colmap import _camera_center, read_sparse_text


def test_camera_center():
    assert _camera_center((1, 0, 0, 0), (1, 2, 3)) == (-1, -2, -3)


def test_sparse_model_reader(tmp_path):
    (tmp_path / "images.txt").write_text(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "1 1 0 0 0 1 2 3 1 frame_000.jpg\n"
        "10 20 3 30 40 -1\n"
        "2 1 0 0 0 2 3 4 1 frame_001.jpg\n"
        "10 20 3\n", encoding="utf-8")
    (tmp_path / "points3D.txt").write_text(
        "# points\n1 0 0 0 255 0 0 0.5 1 0\n", encoding="utf-8")
    result = read_sparse_text(tmp_path)
    assert len(result.poses) == 2
    assert result.poses[0].center == (-1, -2, -3)
    assert result.point_count == 1
    assert result.observations == 2
