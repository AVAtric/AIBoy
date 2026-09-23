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

    def test_pictures_prefer_a_local_original(self):
        """The device pictures are shipped as interface/aiboy_*.png; the
        developer's interface/gameboy_*.png are used instead when present.
        Any other asset falls back to the orig_ prefix."""
        with tempfile.TemporaryDirectory() as d:
            assets = Path(d)
            (assets / "interface").mkdir()
            name, orig = "interface/aiboy_off.png", "interface/gameboy_off.png"
            self.assertEqual(paths.local_or_shipped(name, assets, assets), assets / name)
            (assets / orig).write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped(name, assets, assets), assets / orig)
            # each file is chosen on its own: no original "on" picture -> the shipped one
            self.assertEqual(paths.local_or_shipped("interface/aiboy_on.png", assets, assets),
                             assets / "interface/aiboy_on.png")
        self.assertEqual(paths.original_name("interface/aiboy_on.png"), "interface/gameboy_on.png")
        self.assertEqual(paths.original_name("gb_intro.npz"), "orig_gb_intro.npz")
        self.assertEqual(paths.original_name("other/thing.png"), "other/orig_thing.png")
        for name in paths.ARTWORK.values():
            self.assertTrue(paths.is_original(name), name)
        for name in paths.ARTWORK:
            self.assertFalse(paths.is_original(name), name)
        self.assertTrue(paths.is_original("orig_thing.wav"))
        self.assertIn(gameboy.PHOTO_OFF.name, ("aiboy_off.png", "gameboy_off.png"))
        self.assertIn(gameboy.PHOTO_ON.name, ("aiboy_on.png", "gameboy_on.png"))

    def test_original_next_to_the_app_wins_over_the_bundle(self):
        """A built app ships its artwork inside the bundle; the user's original
        lives in the assets folder next to the app and must be found there.
        AIBOY_SHIPPED_ASSETS=1 ignores originals wherever they are."""
        name, orig = "interface/aiboy_off.png", "interface/gameboy_off.png"
        with tempfile.TemporaryDirectory() as d:
            bundle, local = Path(d) / "bundle", Path(d) / "local"
            (bundle / "interface").mkdir(parents=True); (local / "interface").mkdir(parents=True)
            (bundle / name).write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped(name, bundle, local), bundle / name)
            (bundle / orig).write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped(name, bundle, local), bundle / orig)
            (local / orig).write_bytes(b"x")
            self.assertEqual(paths.local_or_shipped(name, bundle, local), local / orig)
            self.assertEqual(player.intro_asset(name, bundle, local), local / orig)
            with mock.patch.dict(os.environ, {"AIBOY_SHIPPED_ASSETS": "1"}):
                self.assertEqual(paths.local_or_shipped(name, bundle, local), bundle / name)
        # the source tree keeps both folders in one place
        self.assertEqual(paths.LOCAL_ASSET_DIR, paths.ASSET_DIR)

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


class ArtworkTests(unittest.TestCase):
    """The shipped device pictures against the geometry gameboy.py measures
    them in (tools/make_interface.py brings new pictures to that size)."""

    def test_shipped_pictures_match_the_measured_geometry(self):
        from PIL import Image
        off = Image.open(paths.ASSET_DIR / "interface/aiboy_off.png").convert("RGBA")
        on = Image.open(paths.ASSET_DIR / "interface/aiboy_on.png").convert("RGBA")
        self.assertEqual(off.size, (gameboy.PHOTO_W, gameboy.PHOTO_H))
        self.assertEqual(on.size, off.size)
        a, b = np.asarray(off).astype(int), np.asarray(on).astype(int)
        self.assertEqual(a[0, 0, 3], 0, "transparent around the case")
        self.assertEqual(a[gameboy.PHOTO_H // 2, gameboy.PHOTO_W // 2, 3], 255)
        # the LCD window is a plain dark rectangle: nothing printed on it
        x0, y0, x1, y1 = gameboy.LCD
        lcd = a[y0 + 6:y1 - 6, x0 + 6:x1 - 6, :3].mean(axis=2)
        self.assertLess(lcd.max(), 60, "print on the LCD")
        self.assertGreater(a[y0 + 200, x0 - 12, :3].mean(), 45, "the glass beside the window is lighter")
        self.assertGreater(a[y0 - 12, x0 + 200, :3].mean(), 45)
        # the two pictures differ only at the ON light, which is green when on
        diff = np.abs(a - b).sum(axis=2) > 8
        ys, xs = np.nonzero(diff)
        lx0, ly0, lx1, ly1 = gameboy.LED_BOX
        self.assertTrue(xs.min() >= lx0 and ys.min() >= ly0 and xs.max() < lx1 and ys.max() < ly1,
                        (xs.min(), ys.min(), xs.max(), ys.max()))
        cx, cy = (lx0 + lx1) // 2, (ly0 + ly1) // 2
        r, g, bl = b[cy, cx, :3]
        self.assertTrue(g > r + 40 and g > bl + 20, (r, g, bl))
        self.assertLess(a[cy, cx, :3].mean(), 60, "the light is dark when off")
        # every button sits on the case, inside the pad box
        px0, py0, px1, py1 = gameboy.PAD_BOX
        for key, (_kind, (bx0, by0, bx1, by1)) in gameboy.REGIONS.items():
            self.assertTrue(px0 < bx0 < bx1 < px1 and py0 < by0 < by1 < py1, key)

    def test_load_photo_lays_the_device_on_the_window_colour(self):
        bg = (50, 50, 50)
        img = gameboy.load_photo(paths.ASSET_DIR / "interface/aiboy_off.png", background=bg,
                                 size=(400, 664))
        self.assertEqual(img.size, (400, 664))
        self.assertEqual(img.getpixel((0, 0)), bg)
        self.assertEqual(img.getpixel((399, 663)), bg)
        self.assertLess(sum(img.getpixel((200, 100))), 3 * 45, "the case is dark")
        # the LCD window is flattened to the rim colour (no reflection left)
        geo = gameboy.Geometry(1.5)
        img = gameboy.load_photo(paths.ASSET_DIR / "interface/aiboy_off.png", background=bg,
                                 size=(geo.width, geo.height))
        x0, y0, x1, y1 = geo.box(gameboy.LCD)
        window = np.asarray(img)[y0 + 4:y1 - 4, x0 + 4:x1 - 4].astype(int)
        self.assertLess(np.abs(window - np.array(gameboy.LCD_OFF)).max(), 3)

    def test_without_pictures_a_drawn_device_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            off = gameboy.load_photo(Path(d) / "missing.png", size=(300, 498))
            on = gameboy.load_photo(Path(d) / "missing.png", size=(300, 498), lit=True)
        self.assertEqual(off.size, (300, 498))
        geo = gameboy.Geometry(1.5)
        f = 300 / gameboy.PHOTO_W
        cx, cy = round((gameboy.LED_BOX[0] + gameboy.LED_BOX[2]) / 2 * f), round((gameboy.LED_BOX[1] + gameboy.LED_BOX[3]) / 2 * f)
        self.assertGreater(on.getpixel((cx, cy))[1], off.getpixel((cx, cy))[1] + 60)
        self.assertEqual(geo.screen_size, (240, 216))

    def test_the_screen_and_its_rim_fill_the_lcd_window(self):
        """At every scale the frame plus the rim covers the picture's LCD
        window exactly (within a pixel of rounding)."""
        for scale in gameboy.LCD_SCALES:
            geo = gameboy.Geometry(scale)
            lcd = geo.box(gameboy.LCD)
            for got, want in zip(geo.rim_box, lcd):
                self.assertLessEqual(abs(got - want), 1, (scale, geo.rim_box, lcd))
            self.assertEqual(geo.screen_size, (round(160 * scale), round(144 * scale)))
            # the ON light and the pad lie outside the screen
            self.assertLess(geo.led_box[2], geo.rim_box[0])
            self.assertGreater(geo.pad_box[1], geo.rim_box[3])


if __name__ == "__main__":
    unittest.main()
