from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from singlepass3d.app_runtime import (
    TELEMETRY_EXTENSIONS,
    VIDEO_EXTENSIONS,
    job_status,
    reconstruction_command,
    safe_stem,
    save_upload_stream,
    start_job,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/output"

st.set_page_config(page_title="SinglePass3D", page_icon=":material/view_in_ar:", layout="wide")
st.session_state.setdefault("mission", None)
st.session_state.setdefault("viewer_pid", None)

st.title("SinglePass3D", icon=":material/view_in_ar:")
st.caption("Single-pass video reconstruction on this Windows workstation")

with st.sidebar:
    st.header("System", icon=":material/memory:")
    st.success("COLMAP, FFmpeg, OpenCV and Open3D installed")
    st.caption("NVIDIA GeForce RTX 2060 detected")
    st.warning("Reconstruction is batch processing. Results update by stage; it is not live real-time 3D.")

with st.container(border=True):
    st.subheader("New reconstruction", icon=":material/upload_file:")
    video = st.file_uploader("Video", type=sorted(x.lstrip(".") for x in VIDEO_EXTENSIONS),
                             help="MP4 or MOV with overlap and limited motion blur")
    telemetry = st.file_uploader(
        "GPS telemetry (optional)", type=sorted(x.lstrip(".") for x in TELEMETRY_EXTENSIONS),
        help="CSV or JSON with timestamp, latitude, longitude and altitude")
    with st.form("reconstruction"):
        mission_mode = st.segmented_control(
            "Mission mode", ["SIH26158 metric", "Research video only"],
            default="SIH26158 metric",
            help="SIH26158 mode requires GPS/flight telemetry and produces metric ENU output.")
        row = st.container(horizontal=True)
        profile = row.segmented_control(
            "Quality profile", ["Fast", "Default", "Quality", "SIH26158"],
            default="SIH26158", key="profile")
        product = row.segmented_control("Output", ["Sparse preview", "Dense + mesh"],
                                        default="Sparse preview", key="product")
        mission_name = st.text_input("Mission name", value="test-mission")
        start_time = st.text_input(
            "Video frame-zero UTC time", placeholder="2026-09-25T10:30:00Z",
            help="Required only when GPS telemetry is uploaded")
        submitted = st.form_submit_button("Start reconstruction", type="primary",
                                          icon=":material/play_arrow:", disabled=video is None)

if submitted and video is not None:
    try:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        mission = OUTPUT / f"{safe_stem(mission_name)}-{stamp}"
        input_dir = mission / "input"
        video_path = save_upload_stream(video, video.name, input_dir, VIDEO_EXTENSIONS)
        telemetry_path = None
        if telemetry is not None:
            telemetry_path = save_upload_stream(telemetry, telemetry.name, input_dir,
                                                TELEMETRY_EXTENSIONS)
        if mission_mode == "SIH26158 metric" and telemetry_path is None:
            raise ValueError("SIH26158 metric mode requires GPS/flight telemetry.")
        if mission_mode == "SIH26158 metric" and not start_time.strip():
            raise ValueError("SIH26158 metric mode requires video frame-zero UTC time.")
        if profile == "SIH26158" and telemetry_path is None:
            raise ValueError("The SIH26158 profile requires GPS/flight telemetry.")
        command = reconstruction_command(
            video_path, mission, ROOT / "configs" / f"{profile.lower()}.yaml",
            product == "Dense + mesh" or profile == "SIH26158",
            telemetry_path, start_time.strip() or None)
        start_job(command, mission)
        st.session_state.mission = str(mission)
        st.success("Reconstruction started. This page updates automatically.")
    except (OSError, ValueError) as exc:
        st.error(str(exc), icon=":material/error:")


@st.fragment(run_every="2s" if st.session_state.mission else None)
def monitor() -> None:
    if not st.session_state.mission:
        st.info("Upload a video to start. GPS telemetry is optional for a visual test.",
                icon=":material/info:")
        return
    mission = Path(st.session_state.mission)
    status = job_status(mission)
    st.subheader("Processing", icon=":material/progress_activity:")
    st.caption(str(mission))
    stages = status["checkpoints"]
    st.progress(min(len(stages) / 7, 1.0), text=(stages[-1] if stages else "Starting"))
    if status["running"]:
        st.badge("Running", color="blue", icon=":material/pending:")
    elif status["complete"]:
        st.success("Reconstruction complete", icon=":material/check_circle:")
    elif status["failed"]:
        st.error("Reconstruction failed. Open the log below for the exact cause.",
                 icon=":material/error:")
    if stages:
        st.caption("Completed stages: " + " → ".join(stages))
    if status["complete"]:
        metrics_path = mission / "reports/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json(metrics, expanded=1)
        downloads = st.container(horizontal=True)
        downloads.download_button("Download metrics", metrics_path.read_bytes(),
                                  file_name="metrics.json", mime="application/json",
                                  icon=":material/download:")
        for candidate in (mission / "mesh/scene.glb", mission / "pointcloud/processed.ply",
                          mission / "geometry/sparse_observed.ply",
                          mission / "geospatial/sparse_georeferenced.ply"):
            if candidate.is_file():
                downloads.download_button(
                    f"Download {candidate.suffix.upper().lstrip('.')}", candidate.read_bytes(),
                    file_name=candidate.name, mime="application/octet-stream",
                    icon=":material/download:", key=str(candidate))
        if st.button("Open interactive 3D viewer", icon=":material/3d_rotation:"):
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = subprocess.Popen(
                [sys.executable, "-m", "singlepass3d.cli", "viewer", str(mission),
                 "--port", "8765", "--no-browser"], cwd=ROOT, creationflags=flags)
            st.session_state.viewer_pid = process.pid
        if st.session_state.viewer_pid:
            st.link_button("Launch viewer", "http://127.0.0.1:8765",
                           icon=":material/open_in_new:")
    with st.expander("Processing log", icon=":material/description:"):
        st.code(status["log"] or "No log output yet", language="text")


monitor()
