import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiboy.gui import player


class IntroAssetTests(unittest.TestCase):
    def test_orig_asset_preferred_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            assets = Path(d)
            (assets / "gb_intro.npz").write_bytes(b"x")
            self.assertEqual(player.intro_asset("gb_intro.npz", assets), assets / "gb_intro.npz")
            (assets / "orig_gb_intro.npz").write_bytes(b"x")
            self.assertEqual(player.intro_asset("gb_intro.npz", assets), assets / "orig_gb_intro.npz")
            # each file is chosen on its own: no orig wav -> the shipped wav
            self.assertEqual(player.intro_asset("gb_intro.wav", assets), assets / "gb_intro.wav")

    def test_shipped_assets_are_the_aiboy_clip(self):
        """The files in the repo are the generated ones, whichever copy the
        app itself picks at start-up."""
        data = np.load(Path("assets") / "gb_intro.npz")
        frames = data["frames"]
        self.assertEqual(frames.shape[1:], (144, 160, 3))
        idle = frames[int(data["idle_index"])]
        # a flat two-colour frame: background and ink only
        self.assertLessEqual(len(np.unique(idle.reshape(-1, 3), axis=0)), 2)
        self.assertTrue((Path("assets") / "gb_intro.wav").exists())

    @unittest.skipUnless(player.INTRO_FRAMES.exists(), "needs assets/gb_intro.npz")
    def test_intro_asset_shape_and_idle_frame(self):
        intro = player.IntroVideo()
        self.assertEqual(intro.frames.shape[1:], (144, 160, 3))
        self.assertGreater(len(intro.frames), 30)
        self.assertGreater(intro.fps, 10)
        idle = intro.idle_frame()
        self.assertTrue((idle.astype(int).sum(axis=2) < 200).any(), "idle frame must show the logo")
        # the literal last frame is blank; the idle frame is the last one with the logo
        self.assertLess(intro.idle_index, len(intro.frames) - 1)
        self.assertTrue(player.INTRO_SOUND.exists())

    def test_intro_play_ends_with_callback(self):
        if not player.INTRO_FRAMES.exists():
            self.skipTest("needs assets/gb_intro.npz")
        import threading
        intro = player.IntroVideo(sound_path=Path("does-not-exist.wav"))   # silent for the test
        intro.frames = intro.frames[:6]
        intro.fps = 120.0
        frames = player.LatestFrame(); done = threading.Event()
        t = intro.play(frames, threading.Event(), on_done=done.set)
        t.join(5)
        self.assertTrue(done.is_set())
        self.assertIsNotNone(frames.take())


if __name__ == "__main__":
    unittest.main()
