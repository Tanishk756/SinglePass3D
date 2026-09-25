# Architecture

`singlepass3d.cli` drives mission stages through `core.run_stage`. Each stage records version, schema, configuration hash, source identities, and artifact paths. The cache accepts a checkpoint only when all match and artifacts exist.

```mermaid
flowchart LR
  V[Video] --> I[Inspect]
  I --> F[Select frames]
  T[UTC GPS telemetry] --> S[Sync]
  F --> S
  F --> M[Optional dynamic masks]
  M --> C[COLMAP SfM]
  C --> A[Similarity alignment]
  S --> A
  A --> R[Metric sparse PLY and report]
  C --> D[COLMAP MVS]
  D --> P[Filtered dense cloud]
  P --> G[Poisson mesh and GLB]
```

The sparse and fused clouds are observed geometry from COLMAP. GPS provides a local ENU reference and scale through matched camera centers. Depth Anything produces separately labeled relative inferred depth. Poisson output is an inferred surface between observed samples. Modules can run from any Windows project directory.
