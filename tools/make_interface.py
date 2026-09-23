"""Prepare the Game Boy artwork the Preview shows (assets/interface/): each
PNG in the folder is brought to the size the GUI measures its geometry in
(aiboy/gui/gameboy.py: PHOTO_W x PHOTO_H) and re-saved with its transparency,
optimized. The pictures are drawn at 1317 x 2185; the window never shows
the device wider than about 600 px, so the shipped copies are smaller and
compress to about half.

    python tools/make_interface.py            # assets/interface/*.png, in place
    python tools/make_interface.py <folder>   # another folder of pictures

The shipped pair is aiboy_off.png / aiboy_on.png (the AIboy design, LED off
and on); gameboy_off.png / gameboy_on.png are the developer's originals,
never distributed (.gitignore, build_release.py) but shown instead when
present (aiboy.paths.local_or_shipped). For each picture the LCD window it
finds (the big dark rectangle behind the glass) is printed next to
gameboy.LCD, as a check that the geometry still matches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aiboy.gui.gameboy import LCD, PHOTO_H, PHOTO_W  # noqa: E402


def scaled(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """`img` at `size`, resampled with premultiplied alpha so the soft edge
    of the case picks up no fringe from the transparent pixels' colour."""
    if img.size == size:
        return img
    return img.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")


def edge_near(profile: np.ndarray, expected: int, reach: int = 12) -> int:
    """Where the sharpest change of `profile` within `reach` of `expected` is."""
    g = np.abs(np.diff(profile))
    lo, hi = max(0, expected - reach), min(len(g), expected + reach)
    return lo + int(np.argmax(g[lo:hi])) + 1


def measure_lcd(img: Image.Image) -> tuple[int, int, int, int]:
    """The LCD window's edges: the sharpest changes near gameboy.LCD's,
    along a row profile through a strip just inside the window's top and a
    column profile just inside its left (clear of anything printed on the
    middle of an original's screen; works for a dark and a light window)."""
    lum = np.asarray(img.convert("RGB")).astype(np.float32).mean(axis=2)
    x0, y0, x1, y1 = LCD
    rows = lum[y0 + 15:y0 + 45].mean(axis=0)
    cols = lum[:, x0 + 15:x0 + 45].mean(axis=1)
    return edge_near(rows, x0), edge_near(cols, y0), edge_near(rows, x1), edge_near(cols, y1)


def prepare(folder: Path) -> None:
    for path in sorted(folder.glob("*.png")):
        img = scaled(Image.open(path).convert("RGBA"), (PHOTO_W, PHOTO_H))
        img.save(path, optimize=True)
        found = measure_lcd(img)
        # the constants are measured on the shipped pair; an original may
        # frame its window differently (a printed border, say)
        shipped = path.name.startswith("aiboy_")
        note = "" if not shipped or all(abs(a - b) <= 6 for a, b in zip(found, LCD)) else "   <-- differs from gameboy.LCD"
        print(f"{path.name}: {img.size[0]}x{img.size[1]}, {path.stat().st_size // 1024} KB, "
              f"LCD window {found} (gameboy.LCD {LCD}){note}")


if __name__ == "__main__":
    prepare(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "interface")
