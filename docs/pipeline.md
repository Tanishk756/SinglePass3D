# Pipeline and checkpoints

`reconstruct` runs inspection, frame extraction, telemetry synchronization, COLMAP feature extraction, sequential matching, incremental mapping, model conversion, GPS alignment, and report generation. Stage files live below one mission directory. Logs record stage failures and retain COLMAP command output.

Source identifiers include normalized path, file size, modification time, and hashes of the first and last 64 KiB. This is a lightweight change detector; it is not a full cryptographic guarantee for large video files. Configuration changes invalidate affected stages. On failure, correct the input or dependency and rerun the same command.

No heavy stage is silently replaced with simulated output.
