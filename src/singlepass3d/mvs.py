"""COLMAP multi-view stereo and dense PLY inspection."""
from __future__ import annotations

from pathlib import Path

from .colmap import run_colmap
from .core import PipelineError, ensure_output


def ply_vertex_count(path: Path) -> int:
    """Read the actual vertex count from an ASCII or binary PLY header."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        count = None
        for _ in range(1000):
            line = stream.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            if line.startswith(b"element vertex "):
                count = int(line.split()[2])
            if line.strip() == b"end_header":
                break
        else:
            raise ValueError("PLY header is too long")
    if count is None:
        raise ValueError("PLY header has no vertex element")
    return count


def reconstruct_dense(images: Path, sparse_model: Path, output: Path,
                      executable: Path) -> Path:
    """Undistort, estimate multi-view depths and fuse observed dense points."""
    images = images.resolve(strict=True)
    sparse_model = sparse_model.resolve(strict=True)
    output = ensure_output(output)
    if not (sparse_model / "images.bin").is_file():
        raise PipelineError("dense", "COLMAP binary sparse model is missing",
                            "Run sparse reconstruction and pass its numbered model directory")
    run_colmap(executable, ["image_undistorter", "--image_path", str(images),
                            "--input_path", str(sparse_model),
                            "--output_path", str(output), "--output_type", "COLMAP"],
               output / "image_undistorter.log")
    run_colmap(executable, ["patch_match_stereo", "--workspace_path", str(output),
                            "--workspace_format", "COLMAP",
                            "--PatchMatchStereo.geom_consistency", "true"],
               output / "patch_match_stereo.log")
    cloud = output / "fused.ply"
    run_colmap(executable, ["stereo_fusion", "--workspace_path", str(output),
                            "--workspace_format", "COLMAP", "--input_type", "geometric",
                            "--output_path", str(cloud)],
               output / "stereo_fusion.log")
    if not cloud.is_file() or ply_vertex_count(cloud) == 0:
        raise PipelineError("dense", "Stereo fusion produced no points",
                            "Inspect overlap, image texture and COLMAP logs")
    return cloud
