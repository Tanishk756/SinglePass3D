# Reproducible Zurich MAV reference demonstration

Source: University of Zurich Robotics and Perception Group, **Zurich Urban Micro
Aerial Vehicle Dataset**. The source provides synchronized 1920×1080 MAV images,
GPS, IMU, barometric data, calibration, and photogrammetric ground truth. Cite Majdik,
Till, and Scaramuzza, *The Zurich Urban Micro Aerial Vehicle Dataset*, IJRR, 2017.

The downloaded sample is stored locally under `data/reference/zurich-mav/source`.
The generated test inputs are:

- `data/reference/zurich-mav/zurich_mav_demo.mp4`
- `data/reference/zurich-mav/zurich_mav_telemetry.csv`
- frame-zero UTC: `2024-01-01T00:00:00+00:00`

The UTC date is an explicit synthetic anchor because the source subset supplies a
monotonic flight clock rather than civil UTC. Relative timing, coordinates, altitude,
and velocity are copied from the source GPS log. Do not interpret that date as capture date.

The completed mission is `data/output/zurich-mav-sih26158`. It registered 35/35
selected frames, fused 1,381,333 observed points, retained 83,817 filtered points,
and generated a 547,755-triangle vertex-colored mesh. GPS-fit RMSE was 0.748 m.
This is an alignment residual against onboard GPS, not independent absolute accuracy.

Run again from PowerShell:

```powershell
cd C:\SinglePass3D
$env:COLMAP_EXE = 'C:\Tools\COLMAP-4.2.0\bin\colmap.exe'
.\.venv\Scripts\python.exe -m singlepass3d.cli reconstruct `
  --video .\data\reference\zurich-mav\zurich_mav_demo.mp4 `
  --telemetry .\data\reference\zurich-mav\zurich_mav_telemetry.csv `
  --start-time '2024-01-01T00:00:00+00:00' `
  --config .\configs\sih26158.yaml `
  --output .\data\output\zurich-mav-sih26158-copy `
  --full
```

Open the completed result:

```powershell
.\.venv\Scripts\python.exe -m singlepass3d.cli viewer `
  .\data\output\zurich-mav-sih26158
```
