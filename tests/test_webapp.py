import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from storyai.webapp import StoryApp, make_handler

STORY = ("Once upon a time a small fox lived at the edge of a dark wood. "
         "Every morning the fox ran to the river to watch the fish jump.\n") * 5


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.stories, self.models = root / "stories", root / "model"
        self.app = StoryApp(self.stories, self.models)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.app.stop_requested = True
        while self.app.training:
            time.sleep(0.05)
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, path, body=None, header=True):
        headers = {"Content-Type": "application/json"}
        if header:
            headers["X-StoryAI"] = "1"
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, res.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_feed_train_write(self):
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Story AI", page)

        # Other websites can't post to the app.
        self.assertEqual(self.request("/api/stories", {"text": STORY}, header=False)[0], 403)

        self.assertEqual(self.request("/api/stories", {"text": STORY, "name": "../fox"})[0], 200)
        self.assertEqual([p.name for p in self.stories.iterdir()], ["fox.txt"])
        listed = json.loads(self.request("/api/stories")[1])
        self.assertEqual(listed["stories"][0]["name"], "fox.txt")
        self.assertEqual(self.request("/api/stories/delete", {"name": "../model/x.txt"})[0], 400)

        self.assertEqual(self.request("/api/write", {"prompt": "x"})[0], 400)  # nothing trained yet

        self.assertEqual(self.request("/api/train", {"steps": 3})[0], 200)
        self.assertEqual(self.request("/api/train", {"steps": 3})[0], 400)  # already training
        deadline = time.time() + 120
        while self.app.training and time.time() < deadline:
            time.sleep(0.1)
        state = json.loads(self.request("/api/status")[1])
        self.assertIsNone(state["error"], state["log"])
        self.assertEqual(state["model"], {"size": "tiny", "steps_trained": 3})
        self.assertEqual(state["progress"]["step"], 3)

        status, body = self.request("/api/write", {"prompt": "Once upon", "length": "short"})
        self.assertEqual(status, 200)
        story, info = body.decode().rsplit("\x00", 1)
        info = json.loads(info)
        self.assertNotIn("\x00", story)
        self.assertTrue(story.startswith("Once upon"))
        self.assertIn(info["ending"], ("end", "cut"))
        self.assertIn(info["originality"]["verdict"], ("original", "partly copied", "copied"))

        self.assertEqual(self.request("/api/stories/delete", {"name": "fox.txt"})[0], 200)
        self.assertEqual(list(self.stories.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
