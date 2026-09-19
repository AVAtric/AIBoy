"""Screenshots for the README (macOS): the window on the wizard tab once
the boot video has ended, written to assets/screenshot.png at half the
Retina size. Shows the repository's artwork (AIBOY_SHIPPED_ASSETS), never a
local original, so the README carries no maker's marks. The boot video
plays once, with sound. With a window screenshot as argument (one written by
`AIBOY_SHIPPED_ASSETS=1 python tests/gui/check_play_visual.py <dir>`) the
Game Boy is cut out of it instead, into assets/screenshot_play.png.

    python tools/readme_shots.py                          # assets/screenshot.png
    python tools/readme_shots.py <dir>/play-2-a+b+right.png   # assets/screenshot_play.png
"""
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
os.environ["AIBOY_SHIPPED_ASSETS"] = "1"

from aiboy.gui import app as gui  # noqa: E402

OUT = ROOT / "assets" / "screenshot.png"
OUT_PLAY = ROOT / "assets" / "screenshot_play.png"


def cut_gameboy(shot: Path, margin: int = 24) -> None:
    """The device out of a whole-window screenshot: the bounding box of the
    case's beige, at window pixels (a Retina capture is halved)."""
    import numpy as np
    from PIL import Image
    img = Image.open(shot).convert("RGB")
    a = np.asarray(img).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    case = (r > 150) & (g > 140) & (b > 110) & (r - b > 12) & (r - b < 60)
    cols = np.flatnonzero(case.sum(axis=0) > img.height // 6)
    rows = np.flatnonzero(case.sum(axis=1) > img.width // 12)
    crop = img.crop((cols.min() - margin, rows.min() - margin, cols.max() + margin, rows.max() + margin))
    if img.width >= 3000:
        crop = crop.resize((crop.width // 2, crop.height // 2), Image.LANCZOS)
    crop.save(OUT_PLAY, optimize=True)
    print(f"wrote {OUT_PLAY} ({crop.width}x{crop.height})")


def main() -> None:
    if sys.platform != "darwin":
        sys.exit("screencapture is macOS only")
    root = tk.Tk(); app = gui.AIboyGUI(root)
    # Raised, not "-topmost": on macOS a topmost main window hides the
    # Game Boy's child windows (the screen and the pad), see gameboy.py.
    root.lift(); root.focus_force()

    def shoot():
        if app.gameboy.power:                       # the boot video is still on
            root.after(200, shoot); return
        root.update()
        x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(OUT)], check=True)
        from PIL import Image
        img = Image.open(OUT)
        if img.width >= 2 * w:                      # Retina capture: back to window pixels
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        img.save(OUT, optimize=True)
        print(f"wrote {OUT} ({img.width}x{img.height})")
        app._on_close()

    root.after(1500, shoot)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cut_gameboy(Path(sys.argv[1]))
    else:
        main()
