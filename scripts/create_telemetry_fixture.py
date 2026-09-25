"""Create deterministic telemetry for local pipeline experiments."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("At least two samples are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "latitude",
                                                     "longitude", "altitude"])
        writer.writeheader()
        for index in range(args.samples):
            writer.writerow({"timestamp": (start + timedelta(seconds=index)).isoformat(),
                             "latitude": 12.9716 + index * 0.00001,
                             "longitude": 77.5946 + index * 0.00001,
                             "altitude": 100 + index * 0.1})
if __name__ == "__main__":
    main()
