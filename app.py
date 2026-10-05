"""Web interface for the tea leaf classifier.

    .venv\\Scripts\\python app.py

Opens http://127.0.0.1:8000 in your browser. Drop a leaf photo on the page (or
click it, or paste from the clipboard) and the page shows the verdict next to
the annotated image, the same drawing detect.py saves to test/.

Nothing extra to install: this uses Python's own http.server, so the only
requirements are the ones already needed for predict.py.
"""
import argparse
import base64
import io
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from PIL import Image

from common import ROOT
from predict import IMG_EXT

WEB = ROOT / "web"
MAX_UPLOAD = 25 * 1024 * 1024  # a phone photo is a few MB; this is plenty
PREVIEW_PX = 1100  # annotated images are sent back no larger than this

# Two dataset photos shown at the top of the page so the difference between a
# healthy and an affected leaf is on screen before anything is uploaded.
SHOWCASE = [
    {"verdict": "healthy", "name": "Healthy leaf",
     "original": "healthy/UNADJUSTEDNONRAW_thumb_247.jpg",
     "annotated": "test/healthy/healthy__UNADJUSTEDNONRAW_thumb_247.jpg",
     "note": "Even green across the blade. No brown or yellow patches, "
             "edges and midrib intact."},
    {"verdict": "spoiled", "name": "Affected leaf",
     "original": "brown blight/UNADJUSTEDNONRAW_thumb_15c.jpg",
     "annotated": "test/spoiled/brown_blight__UNADJUSTEDNONRAW_thumb_15c.jpg",
     "note": "Brown dead tissue with a pale centre and a dark rim. The orange "
             "boxes mark the discoloured areas."},
]

# ------------------------------------------------------------------ the model
_model = {"ready": threading.Event(), "error": None, "lock": threading.Lock()}


def load_model(model_path):
    """Load the classifier in the background so the page can open right away."""
    try:
        from predict import load
        model, size, classes, device = load(model_path)
        _model.update(model=model, size=size, classes=classes, device=device)
        print(f"Model ready on {device}.")
    except Exception as e:  # a missing or broken checkpoint, no GPU driver, ...
        _model["error"] = f"{type(e).__name__}: {e}"
        print(f"Could not load the model: {_model['error']}")
    finally:
        _model["ready"].set()


def run_prediction(data: bytes):
    from detect import detect
    _model["ready"].wait()
    if _model["error"]:
        raise RuntimeError(_model["error"])
    with _model["lock"]:  # one model instance, so one photo at a time
        annotated, verdict, conf, prob, frac = detect(
            _model["model"], _model["size"], _model["classes"], _model["device"],
            io.BytesIO(data))
    return {
        "verdict": verdict,
        "confidence": round(float(conf), 4),
        "p_healthy": round(float(prob[0]), 4),
        "p_spoiled": round(float(prob[1]), 4),
        "damaged_fraction": round(float(frac), 4) if verdict == "spoiled" else None,
        "annotated": as_data_url(annotated),
    }


def as_data_url(img: Image.Image, px: int = PREVIEW_PX) -> str:
    img = img.convert("RGB")
    if max(img.size) > px:
        img = img.copy()
        img.thumbnail((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- the server
def safe_project_path(rel: str):
    """Resolve a /img?p=... request inside the project folder, images only."""
    try:
        p = (ROOT / unquote(rel).lstrip("/\\")).resolve()
    except OSError:
        return None
    if ROOT not in p.parents or p.suffix.lower() not in IMG_EXT or not p.is_file():
        return None
    return p


class Handler(BaseHTTPRequestHandler):
    server_version = "TeaLeafUI/1.0"

    def log_message(self, fmt, *args):  # one line per upload is enough
        if self.command == "POST":
            print(f"  {fmt % args}")

    # -- helpers
    def send(self, code, body, ctype, extra=()):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, code, obj):
        self.send(code, json.dumps(obj), "application/json; charset=utf-8",
                  [("Cache-Control", "no-store")])

    def send_file(self, path: Path, cache="no-store"):
        if not path.is_file():
            return self.send_json(404, {"error": f"not found: {path.name}"})
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self.send(200, path.read_bytes(), ctype, [("Cache-Control", cache)])

    # -- routes
    def do_GET(self):
        route = urlparse(self.path)
        path, query = route.path, route.query

        if path in ("/", "/index.html"):
            return self.send_file(WEB / "index.html")

        if path == "/api/status":
            if not _model["ready"].is_set():
                return self.send_json(200, {"state": "loading"})
            if _model["error"]:
                return self.send_json(200, {"state": "error", "error": _model["error"]})
            return self.send_json(200, {"state": "ready", "device": _model["device"],
                                        "input_size": _model["size"]})

        if path == "/api/showcase":
            items = [dict(it, original="/img?p=" + quote(it["original"]),
                          annotated="/img?p=" + quote(it["annotated"]))
                     for it in SHOWCASE
                     if (ROOT / it["original"]).is_file() and (ROOT / it["annotated"]).is_file()]
            return self.send_json(200, {"items": items})

        if path == "/img":
            rel = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("p", "")
            target = safe_project_path(rel)
            if not target:
                return self.send_json(404, {"error": "no such image"})
            return self.send_file(target, cache="public, max-age=3600")

        if path.startswith("/web/"):
            target = (WEB / path[len("/web/"):]).resolve()
            if WEB in target.parents:
                return self.send_file(target)

        self.send_json(404, {"error": "no such page"})

    do_HEAD = do_GET

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send_json(404, {"error": "no such endpoint"})
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return self.send_json(400, {"error": "no image in the request"})
        if length > MAX_UPLOAD:
            return self.send_json(413, {"error": f"image larger than {MAX_UPLOAD // 2**20} MB"})
        data = self.rfile.read(length)
        try:
            Image.open(io.BytesIO(data)).verify()  # reject anything that is not an image
        except Exception:
            return self.send_json(400, {"error": "that file is not an image Pillow can read"})
        try:
            result = run_prediction(data)
        except Exception as e:
            return self.send_json(500, {"error": f"{type(e).__name__}: {e}"})
        print(f"  -> {result['verdict'].upper()} {result['confidence']:.1%}")
        self.send_json(200, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "tea_leaf_classifier.pt"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not (WEB / "index.html").is_file():
        raise SystemExit(f"Missing {WEB / 'index.html'}")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    threading.Thread(target=load_model, args=(args.model,), daemon=True).start()
    print(f"Tea leaf classifier running at {url}\nLoading the model... "
          f"(the page opens now; uploads wait until it is ready)\nPress Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
