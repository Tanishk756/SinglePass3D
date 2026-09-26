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
    assert result.mean_track_length == 1
    assert result.mean_reprojection_error_px == 0.5


def test_feature_extractor_receives_camera_and_mask_settings(tmp_path, monkeypatch):
    from singlepass3d.colmap import reconstruct_sparse
    images = tmp_path / "images"; images.mkdir()
    (images / "frame.jpg").write_bytes(b"image")
    masks = tmp_path / "masks"; masks.mkdir()
    output = tmp_path / "result"
    calls = []
    def fake_run(executable, arguments, log):
        calls.append(arguments)
        if arguments[0] == "mapper":
            model = output / "sparse/0"; model.mkdir(parents=True)
            (model / "images.bin").write_bytes(b"x")
        if arguments[0] == "model_converter":
            text = output / "text_model"
            (text / "images.txt").write_text("", encoding="utf-8")
            (text / "points3D.txt").write_text("", encoding="utf-8")
            (text / "cameras.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr("singlepass3d.colmap.run_colmap", fake_run)
    reconstruct_sparse(images, output, tmp_path / "colmap.exe", "PINHOLE", False, 5,
                       masks, "1000,1000,500,500", True)
    features = calls[0]
    assert features[features.index("--image_list_path") + 1].endswith("image_list.txt")
    assert features[features.index("--ImageReader.mask_path") + 1] == str(masks.resolve())
    assert features[features.index("--ImageReader.camera_params") + 1] == "1000,1000,500,500"
    assert features[features.index("--ImageReader.single_camera") + 1] == "1"
    assert features[features.index("--FeatureExtraction.use_gpu") + 1] == "0"
    matcher = calls[1]
    assert matcher[matcher.index("--FeatureMatching.use_gpu") + 1] == "0"
    mapper = calls[2]
    assert mapper[mapper.index("--Mapper.num_threads") + 1] == "-1"
    assert mapper[mapper.index("--Mapper.ba_use_gpu") + 1] == "0"


def test_sparse_reader_reports_triangulation_angle(tmp_path):
    (tmp_path / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 left.jpg\n0 0 -1\n"
        "2 1 0 0 0 -2 0 0 1 right.jpg\n0 0 -1\n", encoding="utf-8")
    (tmp_path / "points3D.txt").write_text(
        "1 1 0 10 255 255 255 0.2 1 0 2 0\n", encoding="utf-8")
    result = read_sparse_text(tmp_path)
    assert result.median_triangulation_angle_deg is not None
    assert 11 < result.median_triangulation_angle_deg < 12
    assert result.low_angle_point_fraction == 0
