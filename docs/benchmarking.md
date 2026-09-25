# Benchmark and acceptance protocol

Industry readiness is established with repeatable field measurements, not screenshots. Keep evaluation missions outside Git and retain source hashes with every report.

## Capture sets

Evaluate at least: urban structures, vegetation, roads, weak texture, repeated facades, low light, moving traffic, altitude changes, and a deliberately invalid yaw-only panorama. Each valid mission needs surveyed checkpoints withheld from alignment.

## Required comparisons

Run Fast, Default, and Quality profiles from identical inputs. Record accepted frames, registered fraction, sparse and dense point counts, track length, reprojection error, runtime, peak memory, alignment residuals, and withheld-checkpoint horizontal, vertical, and 3D errors. Compare against an independently configured COLMAP reference run.

## Acceptance

A run must pass the configured registration and sparse-point gate before dense processing. Metric claims require synchronized telemetry and independent ground truth. Report median, 95th percentile, worst-case error, failure rate, and hardware. Never use GPS alignment residual as independent accuracy.

## Reproducibility

Archive the YAML configuration, software tag, GPU driver, COLMAP version, input identifiers, logs, checkpoints, and exported metrics. Repeat each scenario after dependency or algorithm changes.

