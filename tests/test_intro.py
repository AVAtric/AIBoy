import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from aiboy import paths
from aiboy.gui import gameboy, player


class IntroAssetTests(unittest.TestCase):
    def test_orig_asset_preferred_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            assets = Path(d)
            (assets / "gb_intro.npz").write_bytes(b"x")
            self.assertEqual(player.intro_asset("gb_intro.npz", assets, assets), assets / "gb_intro.npz")
            (assets / "orig_gb_intro.npz").write_bytes(b"x")
            self.assertEqual(player.intro_asset("gb_intro.npz", assets, assets), assets / "orig_gb_intro.npz")
            # each file is chosen on its own: no orig wav -> the shipped wav
            self.assertEqual(player.intro_asset("gb_intro.wav", assets, assets), assets / "gb_intro.wav")

    def test_photo_prefers_a_local_original(self):
        with tempfile.TemporaryDirectory() as d:
            assets = Path(d)
            self.assertEqual(paths.local_or_shipped("gb_interface.png", assets, assets), assets / "gb_interface.png")
            (assets / "orig_gb_interface.png").write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped("gb_interface.png", assets, assets),
                             assets / "orig_gb_interface.png")
        self.assertIn(gameboy.PHOTO.name, ("gb_interface.png", "orig_gb_interface.png"))

    def test_original_next_to_the_app_wins_over_the_bundle(self):
        """A built app ships its artwork inside the bundle; the user's original
        lives in the assets folder next to the app and must be found there.
        AIBOY_SHIPPED_ASSETS=1 ignores originals wherever they are."""
        with tempfile.TemporaryDirectory() as d:
            bundle, local = Path(d) / "bundle", Path(d) / "local"
            bundle.mkdir(); local.mkdir()
            (bundle / "gb_interface.png").write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped("gb_interface.png", bundle, local), bundle / "gb_interface.png")
            (bundle / "orig_gb_interface.png").write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped("gb_interface.png", bundle, local),
                             bundle / "orig_gb_interface.png")
            (local / "orig_gb_interface.png").write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped("gb_interface.png", bundle, local),
                             local / "orig_gb_interface.png")
            self.assertEqual(player.intro_asset("gb_interface.png", bundle, local),
                             local / "orig_gb_interface.png")
            with mock.patch.dict(os.environ, {"AIBOY_SHIPPED_ASSETS": "1"}):
                self.assertEqual(paths.local_or_shipped("gb_interface.png", bundle, local),
                                 bundle / "gb_interface.png")
        # the source tree keeps both folders in one place
        self.assertEqual(paths.LOCAL_ASSET_DIR, paths.ASSET_DIR)

    def test_shipped_photo_carries_no_maker_marks(self):
        """The repository's photo has AIboy lettering and an empty LCD: the
        label under the screen is the only print left in that band and the
        screen holds no dark print at all."""
        from PIL import Image
        img = Image.open(Path("assets") / "gb_interface.png").convert("RGB")
        self.assertEqual(img.size, (gameboy.PHOTO_W, gameboy.PHOTO_H))
        a = np.asarray(img).astype(int)
        x0, y0, x1, y1 = gameboy.LCD
        self.assertFalse((a[y0 + 4:y1 - 4, x0 + 4:x1 - 4].sum(axis=2) < 250).any(), "print on the LCD")
        band = a[354:392, 26:300]
        ink = (band[..., 2] > band[..., 0] + 30) & (band[..., 2] > band[..., 1] + 20)
        cols = np.flatnonzero(ink.any(axis=0))
        self.assertLess(cols.max() - cols.min(), 160, "lettering wider than one short word")

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


class LedPositionTests(unittest.TestCase):
    def test_the_led_hole_is_moved_to_the_middle_of_its_strip(self):
        """The photo's LED hole sits left of the middle; load_photo moves it
        (both the shipped photo and an original have it at the same spot)."""
        import numpy as np
        for name in ("gb_interface.png", "orig_gb_interface.png"):
            path = paths.ASSET_DIR / name
            if not path.exists():
                continue
            img = np.asarray(gameboy.load_photo(path)).astype(int).mean(axis=2)
            for (x, y), expect_hole in ((gameboy.LED, True), (gameboy.LED_SOURCE, False)):
                window = img[y - 4:y + 5, x - 4:x + 5]
                self.assertEqual(bool((window < 70).sum() >= 20), expect_hole, (name, x, y))
            ys, xs = np.nonzero(img[160:190, 45:95] < 70)
            self.assertAlmostEqual(xs.mean() + 45, gameboy.LED[0], delta=1.0, msg=name)
            self.assertAlmostEqual(ys.mean() + 160, gameboy.LED[1], delta=1.0, msg=name)
