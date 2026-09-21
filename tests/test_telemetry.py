import csv
import json

import pytest

from singlepass3d.telemetry import (
    load_telemetry,
    parse_timestamp,
    sample_at,
    synchronize,
)


def test_parse_csv_json_and_validation(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    csv_path.write_text(
        "timestamp,latitude,longitude,altitude\n"
        "2026-01-01T00:00:01Z,12,77,101\n"
        "2026-01-01T00:00:00Z,12,77,100\n", encoding="utf-8")
    samples = load_telemetry(csv_path)
    assert len(samples) == 2
    assert samples[0].altitude == 100
    json_path = tmp_path / "telemetry.json"
    json_path.write_text(json.dumps([item.model_dump() for item in samples]), encoding="utf-8")
    assert load_telemetry(json_path) == samples
    assert parse_timestamp("2026-01-01T00:00:00Z") == samples[0].timestamp
    with pytest.raises(ValueError):
        parse_timestamp("2026-01-01T00:00:00")
    csv_path.write_text("timestamp,latitude,longitude,altitude\n0,100,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_telemetry(csv_path)


def test_interpolation_and_tolerance(tmp_path):
    path = tmp_path / "gps.csv"
    path.write_text("timestamp,latitude,longitude,altitude\n0,0,0,100\n2,2,4,110\n",
                    encoding="utf-8")
    samples = load_telemetry(path)
    middle = sample_at(samples, 1, "interpolate", 2)
    assert middle is not None
    assert (middle.latitude, middle.longitude, middle.altitude) == (1, 2, 105)
    assert sample_at(samples, 1, "nearest", 0.5) is None
    assert sample_at(samples, 8, "interpolate", 2) is None


def test_frame_sync(tmp_path):
    path = tmp_path / "gps.csv"
    path.write_text("timestamp,latitude,longitude,altitude\n100,0,0,100\n102,2,4,110\n",
                    encoding="utf-8")
    manifest = tmp_path / "frame_manifest.csv"
    manifest.write_text("frame_id,filename,timestamp,selected\n0,a.jpg,1,True\n"
                        "1,b.jpg,10,True\n2,,11,False\n", encoding="utf-8")
    output, missing = synchronize(manifest, load_telemetry(path), 100, tmp_path / "sync.csv",
                                  "interpolate", 2)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert missing == 1
    assert len(rows) == 2
    assert float(rows[0]["altitude"]) == 105
    assert rows[1]["matched"] == "False"


def test_blank_optional_csv_value(tmp_path):
    path = tmp_path / "telemetry.csv"
    path.write_text("timestamp,latitude,longitude,altitude,roll\n0,1,2,3,\n",
                    encoding="utf-8")
    assert load_telemetry(path)[0].roll is None
