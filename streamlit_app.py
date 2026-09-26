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
    write_runtime_config,
)
from singlepass3d.video import capture_stream

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/output"
PROFILE_FILES = {
    "Fast": "fast.yaml",
    "Balanced": "default.yaml",
    "Quality": "quality.yaml",
    "Advanced": "advanced.yaml",
}
STAGES = {
    "inspect_video": "Inspecting source",
    "extract_frames": "Selecting frames",
    "sync_telemetry": "Synchronizing telemetry",
    "capture_preflight": "Checking parallax",
    "dynamic_masks": "Masking motion",
    "sparse": "Solving cameras",
    "align_sparse": "Aligning metric frame",
    "relative_depth": "Estimating depth",
    "dense": "Fusing dense geometry",
    "pointcloud": "Cleaning point cloud",
    "mesh": "Building surface",
    "report": "Finalizing report",
}

st.set_page_config(
    page_title="SinglePass3D",
    page_icon=":material/view_in_ar:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.setdefault("mission", None)
st.session_state.setdefault("viewer_pid", None)

st.markdown("""
<style>
.stApp{background:#080c13;color:#edf4ff}
[data-testid="stSidebar"]{background:#0d1420;border-right:1px solid #223047}
[data-testid="stHeader"]{background:rgba(8,12,19,.85)}
.block-container{max-width:1320px;padding-top:2.2rem}
.title{display:flex;align-items:center;gap:.8rem;margin-bottom:.25rem}
.mark{width:36px;height:36px;border:1px solid #4de8ff;border-radius:10px;
display:grid;place-items:center;color:#4de8ff;box-shadow:0 0 24px #4de8ff22}
.title h1{font-size:2rem;letter-spacing:-.04em;margin:0}
.subtitle{color:#8fa3bd;margin:0 0 2rem 3rem}
.step{font:700 .72rem ui-monospace;color:#4de8ff;letter-spacing:.12em}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#223047;border-radius:16px}
.stButton button,.stDownloadButton button{border-radius:10px;font-weight:700}
.stProgress>div>div>div{background:linear-gradient(90deg,#4de8ff,#8a7dff)}
.truth{padding:.8rem 1rem;border-left:2px solid #4de8ff;background:#101a28;
color:#9fb1c8;font-size:.8rem;border-radius:0 8px 8px 0}
</style>
<div class="title"><div class="mark">3D</div><h1>SinglePass3D</h1></div>
<p class="subtitle">Video and live-feed reconstruction workspace</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Mission")
    st.caption("One source. One configuration. One complete pipeline.")
    st.divider()
    st.markdown("**Accuracy policy**")
    st.caption(
        "A mission is accepted only when camera registration, reprojection error, "
        "triangulation angle, and metric alignment pass configured gates."
    )
    st.markdown(
        "<div class='truth'>Point clouds are observed geometry. Filled surfaces and "
        "monocular depth are identified as inferred products.</div>",
        unsafe_allow_html=True,
    )

source_col, settings_col = st.columns([1.05, .95], gap="large")

with source_col, st.container(border=True):
    st.markdown("<span class='step'>01 / SOURCE</span>", unsafe_allow_html=True)
    source_mode = st.segmented_control(
        "Input", ["Video upload", "Live feed"], default="Video upload"
    )
    video = None
    live_source = ""
    live_duration = 15
    if source_mode == "Video upload":
        video = st.file_uploader(
            "Video",
            type=sorted(value.lstrip(".") for value in VIDEO_EXTENSIONS),
            help="Use an original MP4 or MOV with translation and stable overlap.",
        )
        if video is not None:
            st.video(video)
            st.caption(f"{video.name} | {video.size / 1_048_576:.1f} MB")
    else:
        live_source = st.text_input(
            "Camera or stream source",
            placeholder="0 or rtsp://camera/stream",
            help="Use a webcam index, RTSP URL, or HTTP video stream.",
        )
        live_duration = st.number_input(
            "Capture window (seconds)", min_value=2, max_value=3600, value=15
        )
        st.info(
            "Frames are captured for the selected window, then jointly optimized. "
            "This preserves bundle-adjustment accuracy.",
            icon=":material/videocam:",
        )
    telemetry = st.file_uploader(
        "Position telemetry",
        type=sorted(value.lstrip(".") for value in TELEMETRY_EXTENSIONS),
        help="Optional for visual 3D. Required for georeferenced meters.",
    )

with settings_col, st.container(border=True):
    st.markdown("<span class='step'>02 / CALIBRATION</span>", unsafe_allow_html=True)
    coordinate_mode = st.segmented_control(
        "Coordinates",
        ["Metric + georeferenced", "Visual scale"],
        default="Metric + georeferenced",
    )
    profile = st.select_slider(
        "Processing", options=list(PROFILE_FILES), value="Advanced"
    )
    output_product = st.segmented_control(
        "Output",
        ["Sparse preview", "Dense cloud + mesh"],
        default="Dense cloud + mesh",
    )
    mission_name = st.text_input("Mission name", value="mission")
    start_time = st.text_input(
        "Frame-zero UTC",
        placeholder="2026-09-26T10:30:00+05:30",
        help="Required when position telemetry is supplied.",
    )
    with st.expander("Camera and altitude calibration"):
        camera_model = st.selectbox(
            "Camera model",
            [
                "SIMPLE_RADIAL",
                "PINHOLE",
                "SIMPLE_PINHOLE",
                "RADIAL",
                "OPENCV",
                "FULL_OPENCV",
                "OPENCV_FISHEYE",
            ],
        )
        camera_parameters = st.text_input(
            "Calibrated parameters",
            placeholder="Leave empty to estimate, or enter model parameters",
            help="Comma-separated COLMAP parameters in the selected model's order.",
        )
        altitude_datum = st.radio(
            "Altitude datum", ["Ellipsoidal", "Orthometric"], horizontal=True
        )
        geoid_separation = None
        if altitude_datum == "Orthometric":
            geoid_separation = st.number_input(
                "Geoid separation N (m)",
                value=0.0,
                help="Ellipsoidal height h = orthometric height H + N.",
            )
    disabled = source_mode == "Video upload" and video is None
    submitted = st.button(
        "Generate 3D model",
        type="primary",
        icon=":material/play_arrow:",
        disabled=disabled,
        use_container_width=True,
    )

if submitted:
    try:
        if source_mode == "Live feed" and not live_source.strip():
            raise ValueError("Enter a webcam index or stream URL.")
        if coordinate_mode == "Metric + georeferenced" and telemetry is None:
            raise ValueError("Metric coordinates require position telemetry.")
        if telemetry is not None and not start_time.strip():
            raise ValueError("Telemetry synchronization requires frame-zero UTC.")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        mission = OUTPUT / f"{safe_stem(mission_name)}-{stamp}"
        input_dir = mission / "input"
        if source_mode == "Video upload":
            video_path = save_upload_stream(
                video, video.name, input_dir, VIDEO_EXTENSIONS
            )
        else:
            with st.spinner("Capturing live source..."):
                video_path = capture_stream(
                    live_source,
                    input_dir / "live-capture.mp4",
                    float(live_duration),
                    max_dimension=None,
                )
        telemetry_path = None
        if telemetry is not None:
            telemetry_path = save_upload_stream(
                telemetry, telemetry.name, input_dir, TELEMETRY_EXTENSIONS
            )
        config_path = write_runtime_config(
            ROOT / "configs" / PROFILE_FILES[profile],
            mission / "mission-config.yaml",
            camera_model,
            camera_parameters or None,
            altitude_datum.lower(),
            float(geoid_separation) if geoid_separation is not None else None,
        )
        full = output_product == "Dense cloud + mesh" or profile == "Advanced"
        command = reconstruction_command(
            video_path,
            mission,
            config_path,
            full,
            telemetry_path,
            start_time.strip() or None,
        )
        start_job(command, mission)
        st.session_state.mission = str(mission)
        st.toast("Mission started", icon=":material/rocket_launch:")
    except (OSError, ValueError) as exc:
        st.error(str(exc), icon=":material/error:")


@st.fragment(run_every="2s" if st.session_state.mission else None)
def mission_status() -> None:
    st.markdown("---")
    st.markdown("<span class='step'>03 / RECONSTRUCTION</span>", unsafe_allow_html=True)
    if not st.session_state.mission:
        st.caption("Configure a source and start the mission.")
        return
    mission = Path(st.session_state.mission)
    status = job_status(mission)
    completed = status["checkpoints"]
    current = STAGES.get(completed[-1], completed[-1]) if completed else "Starting"
    st.progress(min(len(completed) / len(STAGES), 1.0), text=current)
    state_col, stage_col, name_col = st.columns(3)
    state_col.metric(
        "State",
        "Running" if status["running"] else
        "Complete" if status["complete"] else "Stopped",
    )
    stage_col.metric("Stages", f"{len(completed)} / {len(STAGES)}")
    name_col.metric("Mission", mission.name)

    metrics_path = mission / "reports/metrics.json"
    if status["failed"]:
        st.error("The mission did not pass. Review the engineering log.", icon=":material/error:")
    if status["complete"] and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sfm = metrics.get("sfm", {})
        alignment = metrics.get("gps_alignment", {})
        dense = metrics.get("dense", {})
        mesh = metrics.get("mesh", {})
        st.success("Quality gates passed", icon=":material/check_circle:")
        a, b, c, d = st.columns(4)
        a.metric("Registered cameras", sfm.get("registered_images", "n/a"))
        reprojection = sfm.get("mean_reprojection_error_px")
        b.metric(
            "Reprojection",
            f"{reprojection:.3f} px" if isinstance(reprojection, (int, float)) else "n/a",
        )
        angle = sfm.get("median_triangulation_angle_deg")
        c.metric(
            "Triangulation",
            f"{angle:.2f} deg" if isinstance(angle, (int, float)) else "n/a",
        )
        rmse = alignment.get("rmse_m")
        d.metric(
            "GPS fit",
            f"{rmse:.2f} m" if isinstance(rmse, (int, float)) else "visual scale",
        )
        a, b, c = st.columns(3)
        a.metric("Sparse points", f"{sfm.get('sparse_points', 0):,}")
        b.metric("Filtered points", f"{dense.get('processed_points', 0):,}")
        c.metric("Mesh triangles", f"{mesh.get('triangles', 0):,}")

        downloads = st.container(horizontal=True)
        downloads.download_button(
            "Metrics",
            metrics_path.read_bytes(),
            file_name="metrics.json",
            mime="application/json",
            icon=":material/download:",
        )
        for artifact in (
            mission / "pointcloud/processed.ply",
            mission / "mesh/scene.glb",
            mission / "mesh/scene.obj",
            mission / "geospatial/trajectory.geojson",
        ):
            if artifact.is_file():
                downloads.download_button(
                    artifact.name,
                    artifact.read_bytes(),
                    file_name=artifact.name,
                    mime="application/octet-stream",
                    icon=":material/download:",
                    key=str(artifact),
                )
        if st.button("Open 3D viewer", icon=":material/3d_rotation:"):
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "singlepass3d.cli",
                    "viewer",
                    str(mission),
                    "--port",
                    "8765",
                    "--no-browser",
                ],
                cwd=ROOT,
                creationflags=flags,
            )
            st.session_state.viewer_pid = process.pid
        if st.session_state.viewer_pid:
            st.link_button("Launch viewer", "http://127.0.0.1:8765")
    with st.expander("Engineering log"):
        st.code(status["log"] or "Worker is starting...", language="text")


mission_status()
