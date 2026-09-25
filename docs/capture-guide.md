# Drone capture guide

A monocular video can recover stable 3D only when the camera translates enough to create parallax. A smooth pan from one position can look excellent as video while remaining geometrically ambiguous.

- Fly forward or sideways past the target; avoid hovering and rotating in place.
- Keep the target visible across many frames and maintain roughly 70–85% overlap.
- Use a fast shutter, locked focus/exposure where possible, and avoid digital zoom.
- Record the original 1080p/4K file without messaging-app recompression.
- Export GPS at the highest rate available. Include UTC timestamp, latitude, longitude, ellipsoidal or clearly identified altitude, and accuracy fields when available.
- Provide calibrated intrinsics, rolling-shutter readout time, IMU, RTK/PPK, and GCP checkpoints when available.
- For buildings, use an oblique path that sees roofs and facades. A single nadir strip cannot observe hidden facades or occluded surfaces.

The capture preflight flags weak correspondences, negligible image motion, and homography-dominated footage before expensive dense processing.
