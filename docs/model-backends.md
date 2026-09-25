# Reconstruction backend strategy

## Production baseline

COLMAP remains the observed-geometry baseline because it exposes track, camera, reprojection, and bundle-adjustment outputs that can be independently inspected. Depth Anything and YOLO are optional inference and masking stages; their output is never promoted to measured geometry.

## High-end research candidates

- **VGGT** predicts cameras, depth, point maps, and tracks from multiple views. Its official implementation and weights are suitable for an experimental backend, but memory use depends strongly on image count and resolution. The current RTX 2060 has 6 GB VRAM, so practical local evaluation requires small batches, reduced resolution, or a larger GPU.
- **MASt3R / MASt3R-SfM** provides learned matching and unconstrained SfM, including difficult view sets. Its dependency and checkpoint licenses must be reviewed for the intended use; research code or weights may restrict commercial deployment.
- **DUSt3R** is useful for research comparisons but its official repository uses a noncommercial Creative Commons license, so it is not a default production dependency.

An advanced backend must implement the same artifact contract: camera poses, confidence, observed or inferred classification, coordinate-frame declaration, source provenance, and measured evaluation. Foundation-model output should be refined with geometric optimization and validated against field control.

