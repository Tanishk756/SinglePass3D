# SinglePass3D

**Author:** Tanishk Singhal

SinglePass3D is a Windows-first research prototype that turns a single-pass drone video and timestamped GPS telemetry into observed sparse or dense geometry, metric local coordinates, reports, and optional mesh products. It never treats unseen or monocularly inferred surfaces as measured geometry.

## Windows setup

Required: Windows 11, PowerShell, Git, and Python 3.11. Install COLMAP separately for sparse or dense reconstruction.

    .\setup.ps1 -IncludeVideo -IncludeGeospatial
    .\run.ps1 doctor
    .\test.ps1

For all optional Python stages use .\setup.ps1 -Full. Full setup installs large AI and point-cloud packages but does not install COLMAP or CUDA drivers. Package groups are dev, video, geospatial, ai, pointcloud, mesh, and segmentation.

## Quick start

### Local upload application

The Windows application accepts MP4 or MOV video, optional GPS telemetry, a speed profile, and sparse or dense output. Without telemetry it produces observed geometry in arbitrary COLMAP units. With synchronized telemetry it produces a metric local ENU reconstruction.

    .\run-app.ps1

Open `http://localhost:8501`, upload a video, choose **Fast** and **Sparse preview**, and start reconstruction. The page reports completed checkpoints and exposes the processing log. When complete, download the PLY, GLB, or metrics and open the interactive 3D viewer. Reconstruction is batch processing rather than live real-time 3D; duration depends on video length, selected frames, output mode, and scene quality.

For a GPU-enabled full installation on an NVIDIA Windows system:

    .\setup.ps1 -Full -IncludeCuda

The first dynamic-mask or inferred-depth run downloads its selected model weights. No model weights are committed to this repository.

### Command line

    .\run.ps1 reconstruct --video ".\data\input\mission.mp4" --telemetry ".\data\telemetry\mission.csv" --output ".\data\output\mission01" --start-time "2026-01-01T10:00:00Z"

For a video-only visual reconstruction:

    .\run.ps1 reconstruct-video --video ".\data\input\mission.mp4" --output ".\data\output\visual-test" --config ".\configs\fast.yaml"

Add --full to continue through COLMAP MVS, metric point-cloud filtering, and Poisson OBJ or GLB export. Optional inferred depth and dynamic masks are controlled in YAML. Every expensive stage writes a versioned checkpoint and skips compatible completed work.

View completed output with .\run.ps1 viewer ".\data\output\mission01". The local WebGL viewer supports observed points, GPS and visual trajectories, orbit, zoom, visibility controls, and metric point-to-point measurement.

## Pipeline

~~~mermaid
flowchart LR
  V[Video] --> F[Streaming frame selection]
  T[GPS telemetry] --> S[UTC synchronization]
  F --> S
  F --> M[Optional dynamic masks]
  M --> C[COLMAP sequential SfM]
  S --> A[Robust GPS alignment]
  C --> A
  A --> P[Georeferenced sparse PLY]
  C --> D[COLMAP MVS]
  D --> Q[Metric filtered cloud]
  Q --> G[Inferred Poisson surface OBJ and GLB]
  P --> R[Computed report and viewer]
~~~

Classical multi-view geometry produces observed sparse and dense points. Depth Anything produces relative inferred depth maps. Poisson meshing interpolates an inferred surface between observed points.

## Telemetry

CSV and JSON inputs require timestamp, latitude, longitude, and altitude. Optional fields are roll, pitch, yaw, velocity_x, velocity_y, and velocity_z. Timestamps are Unix seconds or timezone-aware ISO-8601. Latitude and longitude are WGS84. Altitude is currently interpreted as WGS84 ellipsoidal meters; convert orthometric heights first. Video frame zero requires an explicit UTC time.

## Camera and COLMAP

Set camera.model and optional comma-separated camera.parameters in YAML for supplied calibration. With no parameters, COLMAP estimates intrinsics using the selected camera model. COLMAP is discovered from config, COLMAP_EXE, PATH, or common Windows locations. Commands use argument arrays and video-aware sequential matching.

## Outputs

A mission contains manifest.json, checkpoints, logs, frames, synchronized telemetry, COLMAP models, geospatial trajectory and reference files, observed PLY clouds, optional inferred depth, optional mesh or GLB, and JSON or HTML metrics. The report includes only computed values such as accepted frames, registered cameras, sparse observations, track length, reprojection error, alignment residuals, and point or triangle counts.

The pipeline applies a minimum reconstruction quality gate before accepting an output or starting dense processing. By default it requires at least eight registered images, 60% registration, and 5,000 sparse points. These thresholds reject obvious fragments; passing them does not establish survey accuracy. Panoramic rotation from one position cannot provide the parallax required for reliable 3D geometry.

## Accuracy and limitations

Alignment residuals quantify agreement with supplied GPS; they are not independent absolute accuracy. Defensible absolute accuracy needs ground control or surveyed checkpoints. Single-pass occlusion, limited parallax, collinear flight, blur, poor overlap, weak or repeated texture, moving objects, shadows, reflectivity, clock offset, antenna lever arm, GPS uncertainty, and altitude datum errors can degrade results. See docs/accuracy.md and docs/limitations.md.

## Troubleshooting

Doctor returns zero for PASS or WARN and nonzero for blocking failures. Missing optional GPU, CUDA, COLMAP, Open3D, or PyTorch components are warnings. Python other than 3.11 and an unwritable output directory are failures. Use python -m singlepass3d.cli --help for individual stage commands.
