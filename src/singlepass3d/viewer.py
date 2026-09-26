"""Safe local HTTP server for mission artifacts and the built-in WebGL viewer."""
from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def mission_summary(mission: Path) -> dict:
    mission = mission.resolve(strict=True)
    if not mission.is_dir():
        raise ValueError("Mission path must be a directory")
    candidates = {
        "sparse": (
            mission / "geospatial/sparse_georeferenced.ply"
            if (mission / "geospatial/sparse_georeferenced.ply").is_file()
            else mission / "geometry/sparse_observed.ply"
        ),
        "raw": mission / "pointcloud/raw.ply",
        "processed": mission / "pointcloud/processed.ply",
        "mesh_ply": mission / "mesh/scene.ply",
        "mesh_obj": mission / "mesh/scene.obj",
        "mesh": mission / "mesh/scene.glb",
        "trajectory": mission / "geospatial/trajectory.geojson",
        "camera_poses": mission / "geospatial/camera_poses.json",
        "metrics": mission / "reports/metrics.json",
    }
    video_files = (
        list((mission / "input").glob("*.mp4"))
        + list((mission / "input").glob("*.mov"))
        + list(mission.glob("*.mp4"))
        + list(mission.glob("*.mov"))
    )
    if video_files:
        candidates["video"] = video_files[0]
    manifest = mission / "manifest.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    available = {name for name, path in candidates.items() if path.is_file()}
    reconstruction_level = (
        "validated_mesh" if "mesh_ply" in available or "mesh_obj" in available
        else "dense_point_cloud" if "processed" in available or "raw" in available
        else "sparse_point_cloud"
    )
    return {
        "name": mission.name,
        "coordinate_frame": metadata.get("coordinate_frame"),
        "metric_scale": metadata.get("metric_scale", True),
        "reconstruction_level": reconstruction_level,
        "layers": {
            name: "/mission/" + path.relative_to(mission).as_posix()
            for name, path in candidates.items()
            if path.is_file()
        },
    }

def create_handler(default_mission: Path, viewer: Path):
    default_mission, viewer = default_mission.resolve(strict=True), viewer.resolve(strict=True)

    def resolve_active_mission(req_query: str = "") -> Path:
        if req_query:
            from urllib.parse import parse_qs
            qs = parse_qs(req_query)
            if qs.get("mission"):
                cand = default_mission.parent / qs["mission"][0]
                if cand.is_dir():
                    return cand.resolve()
        pointer = default_mission.parent / "active_mission.txt"
        if pointer.is_file():
            try:
                target = Path(pointer.read_text(encoding="utf-8").strip())
                if target.is_dir():
                    return target.resolve()
            except OSError:
                return default_mission
        return default_mission

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            request = parsed.path
            active_mission = resolve_active_mission(parsed.query)

            if request == "/api/mission":
                summary = mission_summary(active_mission)
                payload = json.dumps(summary).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Content-Length", str(len(payload))); self.end_headers()
                self.wfile.write(payload); return

            if request == "/api/missions":
                missions_list = []
                for d in sorted(default_mission.parent.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
                    if d.is_dir() and not d.name.startswith("."):
                        has_ply = (d / "pointcloud/processed.ply").is_file() or (d / "geometry/sparse_observed.ply").is_file()
                        missions_list.append({
                            "name": d.name,
                            "has_3d": has_ply,
                            "is_active": d.resolve() == active_mission.resolve(),
                        })
                payload = json.dumps(missions_list).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload))); self.end_headers()
                self.wfile.write(payload); return

            if request == "/":
                target = viewer / "index.html"
            elif request.startswith("/mission/"):
                subpath = Path(unquote(request[len("/mission/"):]))
                target = (active_mission / subpath).resolve()
                if not target.is_relative_to(active_mission):
                    target = (default_mission / subpath).resolve()
                    if not target.is_relative_to(default_mission):
                        self.send_error(403); return
            else:
                target = (viewer / Path(unquote(request.lstrip("/")))).resolve()
                if not target.is_relative_to(viewer):
                    self.send_error(403); return
            if not target.is_file():
                self.send_error(404); return
            content = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(content))); self.end_headers()
            self.wfile.write(content)
        def log_message(self, format: str, *args: object) -> None:
            return
    return Handler

def serve(mission: Path, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    viewer = Path(__file__).resolve().parent / "viewer_assets"
    if not (viewer / "index.html").is_file():
        raise FileNotFoundError("Built-in viewer assets are missing")
    server = ThreadingHTTPServer((host, port), create_handler(mission, viewer))
    url = "http://" + host + ":" + str(server.server_port) + "/"
    print("SinglePass3D viewer: " + url)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
