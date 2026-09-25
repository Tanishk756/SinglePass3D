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
    candidates = {"sparse": (mission / "geospatial/sparse_georeferenced.ply"
                             if (mission / "geospatial/sparse_georeferenced.ply").is_file()
                             else mission / "geometry/sparse_observed.ply"),
                  "raw": mission / "pointcloud/raw.ply",
                  "processed": mission / "pointcloud/processed.ply",
                  "mesh": mission / "mesh/scene.glb",
                  "trajectory": mission / "geospatial/trajectory.geojson",
                  "camera_poses": mission / "geospatial/camera_poses.json",
                  "metrics": mission / "reports/metrics.json"}
    manifest = mission / "manifest.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    return {"name": mission.name, "coordinate_frame": metadata.get("coordinate_frame"),
            "metric_scale": metadata.get("metric_scale", True),
            "layers": {name: "/mission/" + path.relative_to(mission).as_posix()
                       for name, path in candidates.items() if path.is_file()}}

def create_handler(mission: Path, viewer: Path):
    mission, viewer = mission.resolve(strict=True), viewer.resolve(strict=True)
    summary = mission_summary(mission)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlsplit(self.path).path
            if request == "/api/mission":
                payload = json.dumps(summary).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload))); self.end_headers()
                self.wfile.write(payload); return
            if request == "/":
                target = viewer / "index.html"
            elif request.startswith("/mission/"):
                target = (mission / Path(unquote(request[len("/mission/"):]))).resolve()
                if not target.is_relative_to(mission):
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
