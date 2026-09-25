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
PROFILE_FILES = {
    "Fast": "fast.yaml",
    "Balanced": "default.yaml",
    "Quality": "quality.yaml",
    "Advanced": "advanced.yaml",
}
STAGE_LABELS = {
    "inspect_video": "Inspect",
    "extract_frames": "Frames",
    "sync_telemetry": "Sync",
    "capture_preflight": "Preflight",
    "dynamic_masks": "Mask",
    "sparse": "Sparse 3D",
    "align_sparse": "Georeference",
    "relative_depth": "AI depth",
    "dense": "Dense 3D",
    "pointcloud": "Clean cloud",
    "mesh": "Mesh",
    "report": "Report",
}

st.set_page_config(
    page_title="SinglePass3D Studio",
    page_icon=":material/deployed_code:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.setdefault("mission", None)
st.session_state.setdefault("viewer_pid", None)

st.markdown("""
<style>
:root { --ink:#e8f0ff; --muted:#91a4bf; --cyan:#4de8ff; --violet:#8b7cff; }
.stApp { background:
  radial-gradient(circle at 78% 0%, rgba(64,78,180,.18), transparent 34rem),
  radial-gradient(circle at 10% 20%, rgba(0,183,210,.10), transparent 30rem),
  #070b13; color:var(--ink); }
[data-testid="stSidebar"] { background:#0c1220; border-right:1px solid #1f2d43; }
[data-testid="stHeader"] { background:rgba(7,11,19,.72); }
.block-container { max-width:1500px; padding-top:2rem; }
.hero { padding:2.4rem 2.5rem; border:1px solid rgba(111,153,213,.25);
  border-radius:24px; background:linear-gradient(130deg,rgba(19,34,58,.88),rgba(10,15,27,.72));
  box-shadow:0 24px 80px rgba(0,0,0,.28); margin-bottom:1.4rem; }
.eyebrow { color:var(--cyan); letter-spacing:.16em; font-size:.72rem; font-weight:800; }
.hero h1 { font-size:clamp(2.5rem,5vw,5.2rem); line-height:.96; margin:.55rem 0 1rem;
  letter-spacing:-.055em; background:linear-gradient(90deg,#fff 5%,#9befff 52%,#a89fff);
  -webkit-background-clip:text; color:transparent; }
.hero p { color:#a9bad0; font-size:1.05rem; max-width:760px; line-height:1.65; }
.pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.35rem; }
.pill { border:1px solid #2b4565; background:#101b2b; border-radius:99px;
  color:#bcd2eb; padding:.42rem .72rem; font-size:.78rem; }
.metric-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; margin:1rem 0 1.6rem; }
.metric-box { border:1px solid #20324a; border-radius:16px; padding:1rem 1.1rem;
  background:rgba(12,20,34,.76); }
.metric-box b { display:block; font-size:1.35rem; color:#fff; }
.metric-box span { color:#7f94af; font-size:.75rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#20324a; border-radius:18px; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:700; }
.stProgress > div > div > div { background:linear-gradient(90deg,var(--cyan),var(--violet)); }
.stage-line { color:#91a4bf; font-family:ui-monospace,monospace; font-size:.78rem; }
@media(max-width:800px){.metric-strip{grid-template-columns:repeat(2,1fr)}.hero{padding:1.5rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="hero">
  <div class="eyebrow">VIDEO → MEASURABLE 3D</div>
  <h1>SinglePass3D</h1>
  <p>Turn one moving-camera flight into georeferenced point clouds, camera trajectories,
  AI-assisted depth evidence, and exportable 3D surfaces—all on your workstation.</p>
  <div class="pills">
    <span class="pill">GPU accelerated</span><span class="pill">Metric ENU output</span>
    <span class="pill">Dynamic-object masking</span><span class="pill">Capture diagnostics</span>
    <span class="pill">PLY · OBJ · GLB</span>
  </div>
</section>
<div class="metric-strip">
  <div class="metric-box"><b>1.38M</b><span>REFERENCE FUSED POINTS</span></div>
  <div class="metric-box"><b>35 / 35</b><span>REGISTERED FRAMES</span></div>
  <div class="metric-box"><b>0.89 px</b><span>REPROJECTION ERROR</span></div>
  <div class="metric-box"><b>547K</b><span>MESH TRIANGLES</span></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Reconstruction console")
    st.caption("Local processing · files stay on this machine")
    st.success("Reconstruction stack ready", icon=":material/check_circle:")
    st.markdown("**Hardware**  \nNVIDIA GeForce RTX 2060")
    st.markdown("**Engine**  \nCOLMAP · OpenCV · Open3D · PyTorch")
    st.divider()
    st.markdown("#### Output truth")
    st.caption("Observed clouds, inferred surfaces, and alignment residuals are labeled separately.")
    st.info("Use telemetry for meter-scale output. Video-only output remains in arbitrary units.",
            icon=":material/info:")

left, right = st.columns([1.08, .92], gap="large")
with left:  # noqa: SIM117
    with st.container(border=True):
        st.subheader("01 · Add flight data")
        video = st.file_uploader(
            "Video", type=sorted(x.lstrip(".") for x in VIDEO_EXTENSIONS),
            help="Original MP4 or MOV with forward/sideways translation and stable overlap")
        if video is not None:
            st.video(video)
            st.caption(f"{video.name} · {video.size / 1_048_576:.1f} MB")
        telemetry = st.file_uploader(
            "Position telemetry", type=sorted(x.lstrip(".") for x in TELEMETRY_EXTENSIONS),
            help="CSV or JSON with timestamp, latitude, longitude and altitude")
        with st.form("reconstruction"):
            mode = st.segmented_control(
                "Coordinate mode", ["Metric + georeferenced", "Visual reconstruction"],
                default="Metric + georeferenced")
            profile = st.segmented_control(
                "Processing profile", list(PROFILE_FILES), default="Advanced", key="profile")
            product = st.segmented_control(
                "Product", ["Sparse preview", "Dense cloud + mesh"],
                default="Dense cloud + mesh", key="product")
            mission_name = st.text_input("Mission name", value="flight")
            start_time = st.text_input(
                "Video frame-zero UTC", placeholder="2026-09-25T10:30:00Z",
                help="Required for telemetry synchronization")
            submitted = st.form_submit_button(
                "Build reconstruction", type="primary", icon=":material/play_arrow:",
                disabled=video is None, use_container_width=True)

with right:
    with st.container(border=True):
        st.subheader("02 · Processing plan")
        plan = [
            ("Capture intelligence", "Blur, exposure, feature motion and parallax risk"),
            ("Visual geometry", "SIFT matching, camera registration and bundle adjustment"),
            ("Metric alignment", "Robust GPS synchronization and local ENU transform"),
            ("Dense scene", "Multi-view stereo, filtering and AI depth evidence"),
            ("3D delivery", "Point cloud, colored mesh, report and measurement viewer"),
        ]
        for index, (title, detail) in enumerate(plan, 1):
            st.markdown(f"**{index:02d}  {title}**  \n<span style='color:#8296b2'>{detail}</span>",
                        unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Live experience")
        st.markdown("The public showcase includes a browser-based **live 2D vision preview** "
                    "while video plays. Full 3D uses GPU batch stages and streams progress here.")
        st.link_button("Open public showcase", "https://tanishk756.github.io/SinglePass3D/",
                       icon=":material/open_in_new:", use_container_width=True)

if submitted and video is not None:
    try:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        mission = OUTPUT / f"{safe_stem(mission_name)}-{stamp}"
        input_dir = mission / "input"
        video_path = save_upload_stream(video, video.name, input_dir, VIDEO_EXTENSIONS)
        telemetry_path = None
        if telemetry is not None:
            telemetry_path = save_upload_stream(
                telemetry, telemetry.name, input_dir, TELEMETRY_EXTENSIONS)
        metric_mode = mode == "Metric + georeferenced"
        if metric_mode and telemetry_path is None:
            raise ValueError("Metric mode requires position telemetry.")
        if metric_mode and not start_time.strip():
            raise ValueError("Metric mode requires video frame-zero UTC.")
        if profile == "Advanced" and telemetry_path is None:
            raise ValueError("Advanced profile requires telemetry for validated metric output.")
        command = reconstruction_command(
            video_path, mission, ROOT / "configs" / PROFILE_FILES[profile],
            product == "Dense cloud + mesh" or profile == "Advanced",
            telemetry_path, start_time.strip() or None)
        start_job(command, mission)
        st.session_state.mission = str(mission)
        st.toast("Reconstruction started", icon=":material/rocket_launch:")
    except (OSError, ValueError) as exc:
        st.error(str(exc), icon=":material/error:")


@st.fragment(run_every="2s" if st.session_state.mission else None)
def monitor() -> None:
    st.markdown("---")
    st.subheader("Mission telemetry", icon=":material/monitoring:")
    if not st.session_state.mission:
        st.caption("A mission timeline will appear here after upload.")
        return
    mission = Path(st.session_state.mission)
    status = job_status(mission)
    stages = status["checkpoints"]
    stage_text = "  →  ".join(STAGE_LABELS.get(stage, stage) for stage in stages)
    total = 12
    st.progress(min(len(stages) / total, 1.0),
                text=STAGE_LABELS.get(stages[-1], stages[-1]) if stages else "Starting")
    st.markdown(f"<div class='stage-line'>{stage_text or 'Waiting for worker'}</div>",
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Completed stages", f"{len(stages)} / {total}")
    b.metric("State", "Running" if status["running"] else
             "Complete" if status["complete"] else "Needs attention")
    b.caption(mission.name)
    metrics_path = mission / "reports/metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sfm = metrics.get("sfm", {})
        c.metric("Registered cameras", sfm.get("registered_images", "—"))
    if status["failed"]:
        st.error("Processing stopped. The diagnostic log below contains the exact cause.",
                 icon=":material/error:")
    if status["complete"] and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sfm, dense, mesh = metrics.get("sfm", {}), metrics.get("dense", {}), metrics.get("mesh", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sparse points", f"{sfm.get('sparse_points', 0):,}")
        m2.metric("Dense points", f"{dense.get('processed_points', 0):,}")
        m3.metric("Triangles", f"{mesh.get('triangles', 0):,}")
        error = sfm.get("mean_reprojection_error_px")
        m4.metric("Reprojection", f"{error:.2f} px" if isinstance(error, (int, float)) else "—")
        downloads = st.container(horizontal=True)
        downloads.download_button("Metrics JSON", metrics_path.read_bytes(),
                                  file_name="metrics.json", mime="application/json",
                                  icon=":material/download:")
        for candidate in (mission / "mesh/scene.glb", mission / "mesh/scene.obj",
                          mission / "pointcloud/processed.ply",
                          mission / "geospatial/sparse_georeferenced.ply",
                          mission / "geometry/sparse_observed.ply"):
            if candidate.is_file():
                downloads.download_button(
                    candidate.name, candidate.read_bytes(), file_name=candidate.name,
                    mime="application/octet-stream", icon=":material/download:",
                    key=str(candidate))
        if st.button("Start local 3D viewer", icon=":material/3d_rotation:"):
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = subprocess.Popen(
                [sys.executable, "-m", "singlepass3d.cli", "viewer", str(mission),
                 "--port", "8765", "--no-browser"], cwd=ROOT, creationflags=flags)
            st.session_state.viewer_pid = process.pid
        if st.session_state.viewer_pid:
            st.link_button("Open metric 3D viewer", "http://127.0.0.1:8765",
                           icon=":material/open_in_new:")
    with st.expander("Engineering log", icon=":material/terminal:"):
        st.code(status["log"] or "Worker is starting…", language="text")


monitor()
