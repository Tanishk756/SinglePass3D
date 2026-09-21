# SinglePass3D

Author: Tanishk Singhal

SinglePass3D is a Windows-first research prototype for reconstructing observed geometry from a single drone flight. This directory is isolated from the parent React application.

## Windows setup

Install Python 3.11, Git, and optionally FFmpeg, COLMAP, OpenCV, and NVIDIA drivers. In PowerShell:

    .\setup.ps1 -IncludeVideo
    .\run.ps1 doctor
    .\test.ps1

The base install needs no GPU, COLMAP, or model weights. Optional extras are `video`, `geospatial`, `reconstruction`, and `dev`. CUDA-enabled PyTorch requires a separate install matched to the local NVIDIA driver.

## Commands

    .\run.ps1 inspect-video ".\data\input\mission.mp4"
    .\run.ps1 extract-frames ".\data\input\mission.mp4" --output ".\data\output\mission01" --config ".\configs\default.yaml"

Frame extraction creates `frames/frame_manifest.csv`, selected JPEGs, `checkpoints/extract_frames.json`, and `logs/pipeline.log` in the output mission. The extraction checkpoint is invalidated by source size/mtime, configuration hash, or stage version.

## Current architecture and limits

Implemented: configuration, diagnostics, stage checkpoints, video inspection, and streaming frame extraction with blur/exposure filtering. GPS synchronization, COLMAP reconstruction, metric alignment, dense geometry, mesh, and viewer are pending. No geometry or accuracy claim is produced by the current CLI.

Single-pass video has occlusions and limited parallax. Motion blur, weak or repeated texture, moving objects, shadows, reflective surfaces, insufficient overlap, GPS uncertainty, and monocular depth ambiguity can degrade results. Observed geometry and future inferred geometry must be labeled separately. Metric accuracy requires actual residual measurements, not an assumed GPS precision.

The canonical telemetry format planned for the next stage requires timestamp, latitude, longitude, and altitude; optional fields are roll, pitch, yaw, and velocity components. Timestamps must share a documented clock origin with video. Use a local metric coordinate frame before geometric alignment.

Troubleshooting: `doctor` reports optional tools as WARN and blocking Python/output failures as FAIL. If OpenCV is absent, install `.[video]`. If a video cannot be read, inspect codec support and verify the file is MP4 or MOV.

## Telemetry and synchronization

CSV files need `timestamp,latitude,longitude,altitude` headers. JSON files contain an array of objects with the same fields. Timestamps accept Unix seconds or timezone-aware ISO-8601, including `Z`. Latitude and longitude are WGS84 degrees; altitude is in meters in the source datum and must be documented by the operator. Optional roll, pitch, yaw, and velocity components are supported.

    .\run.ps1 telemetry ".\data\telemetry\mission.csv"
    .\run.ps1 sync-telemetry --manifest ".\data\output\mission01\frames\frame_manifest.csv" --telemetry ".\data\telemetry\mission.csv" --start-time "2026-01-01T10:00:00Z" --output ".\data\output\mission01\telemetry\frame_telemetry.csv"

The video start time is required because a video frame timestamp is relative to the clip, while telemetry uses UTC. Unmatched selected frames remain in the synchronization CSV with empty position fields.

## Sparse reconstruction

Install COLMAP for Windows and add `colmap.exe` to PATH, set `COLMAP_EXE`, or set `colmap.executable` in the YAML config. Then run:

    .\run.ps1 reconstruct-sparse --images ".\data\output\mission01\frames" --output ".\data\output\mission01" --config ".\configs\default.yaml"

This executes feature extraction, sequential matching, incremental mapping, and text model conversion. Reported registered images, sparse points, and observations are parsed from COLMAP output. The stage checkpoint tracks selected image identities and configuration. No reconstruction was run in the development environment because COLMAP is absent.

## Metric alignment

With `.[geospatial]` installed, use registered COLMAP poses and synchronized frame GPS:

    .\run.ps1 align-sparse --model ".\data\output\mission01\reconstruction\text_model" --synchronized ".\data\output\mission01\telemetry\frame_telemetry.csv" --output ".\data\output\mission01"

The output includes a metric ENU PLY, camera poses, reference transform, and computed alignment residuals. A path with nearly collinear camera positions cannot constrain the full 3D orientation and is rejected. The CSV altitude datum must match the interpretation used for the local reference. Residuals describe fit to supplied GPS, not absolute survey accuracy.

## One-command sparse milestone

After installing `.[video,geospatial]` and COLMAP, provide a video start time on the same UTC clock as the telemetry:

    .\run.ps1 reconstruct --video ".\data\input\mission.mp4" --telemetry ".\data\telemetry\mission.csv" --output ".\data\output\mission01" --start-time "2026-01-01T10:00:00Z"

This runs the sparse milestone and writes `manifest.json`, checkpoints, logs, selected frames, frame telemetry, COLMAP model, georeferenced sparse PLY, camera poses, reference transform, and an HTML/JSON report. Re-running skips compatible completed stages. `report <mission>` regenerates the report from existing stage artifacts.

The dense reconstruction, optional AI depth, dynamic masking, mesh, GLB, and interactive viewer are not implemented yet. They must not be inferred from the sparse output.

## Optional inferred depth

Install `.[ai,geospatial]` and run `.\run.ps1 estimate-depth --images ".\data\output\mission01\frames" --output ".\data\output\mission01\reconstruction\depth"`. The [Depth Anything V2 Small model](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) downloads on first use. Output `.npz` files contain relative depth maps tagged `inferred`; they are not metric geometry. The backend uses CUDA when available and falls back to CPU. No model weights are bundled.

## Optional observed dense reconstruction

After a successful sparse model, run:

    .\run.ps1 reconstruct-dense --images ".\data\output\mission01\frames" --sparse-model ".\data\output\mission01\reconstruction\sparse\0" --output ".\data\output\mission01"

COLMAP performs image undistortion, geometric-consistency PatchMatch, and stereo fusion. The output reconstruction/dense/fused.ply contains observed dense points in the COLMAP reconstruction frame, not yet georeferenced meters. The printed count is read from its PLY header. Dense reconstruction has not been exercised locally because COLMAP is absent.

## Optional metric dense point cloud

Install the pointcloud extra and process the COLMAP fused PLY with the sparse GPS transform:

    .\run.ps1 process-cloud --input ".\data\output\mission01\reconstruction\dense\fused.ply" --reference ".\data\output\mission01\geospatial\reference.json" --output ".\data\output\mission01\pointcloud"

This creates raw.ply and processed.ply in local ENU meters, plus metrics.json with actual counts and bounds. The raw transformed cloud is retained separately. Open3D processing has not been exercised locally because Open3D is absent.

## Optional mesh

Install the pointcloud and mesh extras, then run:

    .\run.ps1 mesh --input ".\data\output\mission01\pointcloud\processed.ply" --output ".\data\output\mission01\mesh"

This writes an OBJ and GLB from a Poisson surface interpolated between observed dense points. The surface is labeled inferred, and no texture is claimed. Meshing requires sufficient input points and has not been exercised locally because Open3D and trimesh are absent.
