"""A point-and-click web page for the story AI: feed stories, train, and write.

Run `python story.py app` and open the address it prints. It uses only Python's
built-in web server, so there is nothing extra to install.
"""

import json
import threading
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .checkpoint import read_checkpoint_info
from .data import count_new_stories, list_story_files, save_story
from .model import PRESETS
from .trainer import TrainingError, train
from .writer import NoModelError, StoryWriter

PAGE = Path(__file__).with_name("app.html")
MAX_BODY = 64 * 1024 * 1024
LENGTHS = {"short": 200, "medium": 600, "long": 1500}


class Busy(Exception):
    pass


class StoryApp:
    """Everything the web page can ask for, independent of HTTP."""

    def __init__(self, stories_dir, model_dir):
        self.stories_dir, self.model_dir = Path(stories_dir), Path(model_dir)
        self.lock = threading.Lock()
        self.log = deque(maxlen=500)
        self.training = False
        self.stop_requested = False
        self.progress = {}
        self.error = None
        self._writer, self._writer_stamp = None, None
        self._info, self._info_stamp = None, None

    # ------------------------------------------------------------------ stories

    def stories(self):
        files = list_story_files(self.stories_dir)
        return {
            "stories": [{"name": str(p.relative_to(self.stories_dir)), "words": w, "start": s}
                        for p, w, s in files],
            "total_words": sum(w for _, w, _ in files),
        }

    def add_story(self, text, name=None):
        text = (text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ValueError("That story is empty.")
        if count_new_stories(self.stories_dir, text) == 0:
            raise ValueError("That story is already in your collection.")
        path = save_story(self.stories_dir, text, name=name)
        return {"saved": path.name}

    def delete_story(self, name):
        path = (self.stories_dir / name).resolve()
        if path.suffix != ".txt" or self.stories_dir.resolve() not in path.parents or not path.is_file():
            raise ValueError("No such story.")
        path.unlink()
        return {"deleted": name}

    # ----------------------------------------------------------------- training

    def start_training(self, steps, size=None, fresh=False):
        steps = int(steps)
        if not 1 <= steps <= 1_000_000:
            raise ValueError("Steps must be between 1 and 1,000,000.")
        if size not in (None, *PRESETS):
            raise ValueError(f"Unknown size {size!r}.")
        with self.lock:
            if self.training:
                raise Busy("Already training.")
            self.training, self.stop_requested, self.error = True, False, None
            self.progress = {"step": 0, "steps": steps}
            self.log.clear()
        threading.Thread(target=self._train, args=(steps, size, fresh), daemon=True).start()
        return {"started": True}

    def _train(self, steps, size, fresh):
        try:
            train(self.stories_dir, self.model_dir, size=size, steps=steps, fresh=fresh,
                  log=self._log, should_stop=lambda: self.stop_requested,
                  on_progress=self._on_progress)
        except TrainingError as e:
            self.error = str(e)
            self._log(str(e))
        except Exception as e:  # show the problem on the page instead of dying silently
            self.error = f"Training crashed: {e}"
            self._log(traceback.format_exc())
        finally:
            with self.lock:
                self.progress["stopped"] = self.stop_requested
                self.training = False

    def stop_training(self):
        self.stop_requested = True
        return {"stopping": self.training}

    def _log(self, message):
        with self.lock:
            self.log.extend(str(message).split("\n"))

    def _on_progress(self, **info):
        with self.lock:
            self.progress.update(info)

    # ------------------------------------------------------------------- status

    def status(self):
        with self.lock:
            state = {"training": self.training, "progress": dict(self.progress),
                     "error": self.error, "log": list(self.log)[-120:]}
        state["model"] = self._model_info()
        return state

    def _model_info(self):
        ckpt = self.model_dir / "model.npz"
        try:
            stamp = ckpt.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if stamp != self._info_stamp:  # only re-read after training saves a new brain
            try:
                info = read_checkpoint_info(ckpt)
            except Exception:
                return self._info  # mid-save; the next poll will catch up
            c = info["config"]
            size = next((k for k, v in PRESETS.items()
                         if all(c.get(f) == x for f, x in v.items())), "custom")
            self._info = {"size": size, "steps_trained": info["meta"].get("steps_trained", 0)}
            self._info_stamp = stamp
        return self._info

    # ------------------------------------------------------------------ writing

    def writer(self):
        ckpt = self.model_dir / "model.npz"
        stamp = ckpt.stat().st_mtime_ns if ckpt.exists() else None
        if stamp is None:
            raise NoModelError("No trained model yet. Feed it stories, then train it.")
        with self.lock:  # reload whenever training has saved a newer brain
            if self._writer is None or stamp != self._writer_stamp:
                self._writer, self._writer_stamp = StoryWriter(self.model_dir), stamp
            return self._writer


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StoryAI"

        def log_message(self, *args):
            pass  # keep the terminal quiet

        def _send(self, code, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send(200, app.status())
            elif path == "/api/stories":
                self._send(200, app.stories())
            else:
                self._send(404, {"error": "Not found"})

        def do_POST(self):
            # Only our own page sends this header; browsers won't let other websites add it
            # without permission, so they can't use this server behind your back.
            if self.headers.get("X-StoryAI") != "1":
                return self._send(403, {"error": "Forbidden"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "That's too big to send at once."})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/stories":
                    return self._send(200, app.add_story(body.get("text", ""), body.get("name")))
                if path == "/api/stories/delete":
                    return self._send(200, app.delete_story(body.get("name", "")))
                if path == "/api/train":
                    return self._send(200, app.start_training(
                        body.get("steps", 2000), body.get("size") or None, bool(body.get("fresh"))))
                if path == "/api/stop":
                    return self._send(200, app.stop_training())
                if path == "/api/write":
                    return self._write(body)
                self._send(404, {"error": "Not found"})
            except (ValueError, TypeError, AttributeError, Busy, NoModelError) as e:
                self._send(400, {"error": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page was closed mid-story

        def _write(self, body):
            writer = app.writer()
            temperature = min(2.0, max(0.05, float(body.get("temperature", 0.8))))
            max_tokens = LENGTHS.get(body.get("length"), 600)
            # Stream the story as it's written. The last line says how it ended.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            story = writer.write(str(body.get("prompt", "")), max_tokens,
                                 temperature=temperature, rng=np.random.default_rng())
            for piece in story:
                self.wfile.write(piece.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"\x00" + (b"end" if story.finished else b"cut"))

    return Handler


def serve(stories_dir, model_dir, host="127.0.0.1", port=8000):
    app = StoryApp(stories_dir, model_dir)
    try:
        server = ThreadingHTTPServer((host, port), make_handler(app))
    except OSError as e:
        raise SystemExit(f"Couldn't start on port {port} ({e.strerror}). "
                         f"Try another one:  python story.py app --port {port + 1}")
    server.daemon_threads = True
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"\n  Story AI is running. Open this in your browser:  http://{shown}:{port}\n"
          "  (Press Ctrl+C here to quit.)\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        app.stop_requested = True
        server.server_close()
