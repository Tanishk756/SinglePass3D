import pytest

from singlepass3d.mvs import ply_vertex_count, reconstruct_dense


def test_ply_vertex_count(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_bytes(b"ply\nformat binary_little_endian 1.0\n"
                     b"element vertex 123\nproperty float x\nend_header\n")
    assert ply_vertex_count(path) == 123
    path.write_text("bad", encoding="ascii")
    with pytest.raises(ValueError):
        ply_vertex_count(path)


def test_mvs_command_sequence(tmp_path, monkeypatch):
    images = tmp_path / "frames"
    images.mkdir()
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "images.bin").write_bytes(b"model")
    calls = []
    def fake_run(executable, arguments, log):
        calls.append(arguments[0])
        if arguments[0] == "stereo_fusion":
            (tmp_path / "dense/fused.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n",
                encoding="ascii")
    monkeypatch.setattr("singlepass3d.mvs.run_colmap", fake_run)
    cloud = reconstruct_dense(images, sparse, tmp_path / "dense", tmp_path / "colmap.exe")
    assert cloud.is_file()
    assert calls == ["image_undistorter", "patch_match_stereo", "stereo_fusion"]
