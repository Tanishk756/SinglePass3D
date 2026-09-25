# SIH26158 requirements and verification

## Source record

**Title:** Single-Pass Drone Video to Accurate 3D Model Generation System  
**Organization:** National Technical Research Organisation (NTRO)  
**Category:** Software  
**Theme:** Robotics and Drones

The SIH portal record was captured on 4 September 2026. Its public text ends with the literal placeholder “Add 'Desired Output' and 'Evaluation Criteria' table here”. The linked Google Drive attachment must therefore be obtained from the organizer before final scoring claims can be made. The official challenge dataset is described as “Will be provided real time.”

## Traceability

| Challenge requirement | Implementation | Verification evidence | Status |
|---|---|---|---|
| 1080p/4K drone video | FFmpeg/OpenCV inspection and selectable frame extraction | video_info.json, frame_manifest.csv | Implemented |
| GPS coordinates and flight metadata | CSV/JSON ingestion, timestamp interpolation | frame_telemetry.csv | Implemented |
| Georeferenced metric 3D | Robust similarity alignment into local ENU metres | reference.json, alignment metrics | Implemented; checkpoint validation required |
| Terrain, structures, roofs, roads, vegetation, obstacles | Photogrammetric observed cloud and inferred Poisson surface | PLY, OBJ and GLB products | Geometry implemented; semantic labels pending |
| Textured mesh or point cloud | Colored point cloud and vertex-colored mesh | processed.ply, scene.obj, scene.glb | Implemented as vertex color; UV texture atlas pending |
| Dynamic objects | YOLO segmentation masks before feature extraction | masks and mask manifest | Implemented in SIH profile |
| Motion blur/compression/illumination | Frame rejection plus capture preflight | frame manifest, capture_quality.json | Implemented |
| Limited angles/occlusion | Capture warnings; dense MVS; optional learned depth | diagnostics and relative depth maps | Partial; inferred surfaces are identified |
| Visualization and measurement | Local WebGL viewer and metric point picking | viewer | Implemented |
| Real or near-real time | Stage checkpoints and timings | checkpoint JSON, report | Batch near-real-time prototype; target latency must be benchmarked |
| Metric accuracy without extensive GCPs | GPS scale/alignment with robust outlier rejection | GPS residual metrics | Implemented; absolute accuracy needs checkpoints/RTK |
| Optional IMU, barometer, intrinsics, RTK/PPK | Intrinsics and generic altitude fields accepted | config and telemetry parser | Intrinsics/RTK coordinates supported; orientation fusion pending |

## Acceptance gates for demonstrations

1. Use **SIH26158 metric mode**, which requires video, telemetry, and frame-zero UTC.
2. Capture must have translational motion and sufficient texture. Treat a preflight homography warning as a reason to recapture.
3. At least 70% of selected images and at least 12 images must register.
4. Sparse reconstruction must contain at least 8,000 points for the SIH profile.
5. Report GPS fit residuals and independent checkpoint RMSE separately.
6. Label outputs as observed, interpolated, or AI inferred. Never present inferred geometry as surveyed fact.
7. Benchmark processing time on the target 1080p and 4K datasets.

These internal gates are engineering safeguards, not organizer-published scoring criteria.
