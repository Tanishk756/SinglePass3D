# Accuracy methodology

The similarity fit estimates uniform scale, rotation, and translation from COLMAP camera centers and matched GPS positions in local ENU meters. RANSAC rejects observations above a configured residual threshold. The report computes mean, median, RMSE, and maximum residual from inliers.

These values describe internal agreement with supplied GPS, not absolute 3D accuracy. GPS noise, altitude datum, camera-to-GPS lever arm, clock offset, and weak camera geometry can bias scale and position. The current implementation does not estimate these uncertainties independently. Ground control points or independent surveyed checkpoints are needed for defensible absolute accuracy claims.

The current coordinate converter treats input altitude as WGS84 ellipsoidal height. Orthometric heights need geoid conversion before use; otherwise absolute height is biased.
