"""Make the shipped Game Boy photo, assets/gb_interface.png, from a local
assets/orig_gb_interface.png: the maker's name and logo on the LCD are
replaced by plain screen colour and the "GAME BOY" label under the screen
by "AIBOY", so the repository ships no trademark artwork. Everything the
GUI measures (LCD window, LED, buttons, see aiboy/gui/gameboy.py) is
untouched: the two files have the same size and geometry.

    python tools/make_interface.py

Like the boot video (tools/make_intro.py), the original is not distributed
(.gitignore: assets/orig_gb*) but is used instead of the shipped file by
anyone who puts one there (aiboy.paths.local_or_shipped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ORIG = ASSETS / "orig_gb_interface.png"
OUT = ASSETS / "gb_interface.png"

LCD = (109, 99, 351, 317)          # the screen window, as in gameboy.py
LABEL = (26, 354, 300, 392)        # "Nintendo GAME BOY(tm)" under the screen
LABEL_INK = (45, 35, 86)           # the label's blue-purple, measured
FONTS = ["/System/Library/Fonts/Supplemental/Arial Black.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "C:/Windows/Fonts/ariblk.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fill_from_neighbours(img: Image.Image, box: tuple[int, int, int, int],
                         donor_x: tuple[int, int] = (305, 425)) -> None:
    """Paint `box` with case plastic taken from the same rows just to the
    right of it (`donor_x`), so the moulding's top-to-bottom shading is kept:
    each row gets the donor's tone, and the donor's grain is mirror-tiled
    across the width so the patch has the same texture as its surroundings."""
    x0, y0, x1, y1 = box
    a = np.asarray(img).astype(np.float32)
    donor = a[y0:y1, donor_x[0]:donor_x[1]]
    tone = donor.mean(axis=1, keepdims=True)                  # per-row colour
    grain = donor - tone
    tiles = [grain, grain[:, ::-1]]
    strip = np.concatenate(tiles * ((x1 - x0) // (2 * grain.shape[1]) + 1), axis=1)[:, :x1 - x0]
    a[y0:y1, x0:x1] = np.clip(tone + strip, 0, 255)
    img.paste(Image.fromarray(a.astype(np.uint8)))


def clear_lcd(img: Image.Image, inset: int = 2) -> None:
    """Replace the LCD's printed logo (and the dither the photo picked up
    from it) with a clean, evenly shaded screen: each row takes the median
    colour of its unprinted pixels, smoothed down the screen, with a little
    grain so it still reads as glass and not as a flat fill."""
    x0, y0, x1, y1 = (LCD[0] + inset, LCD[1] + inset, LCD[2] - inset, LCD[3] - inset)
    a = np.asarray(img).astype(np.float32)
    lcd = a[y0:y1, x0:x1]
    ink = lcd.sum(axis=2) < 300
    ink = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))) > 0
    rows = np.array([np.median(lcd[y][~ink[y]], axis=0) if (~ink[y]).sum() > 20 else np.nan * np.ones(3)
                     for y in range(lcd.shape[0])])
    for c in range(3):                                        # fill rows that were all print
        col = rows[:, c]
        bad = np.isnan(col)
        col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    k = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2); k /= k.sum()
    pad = np.pad(rows, ((12, 12), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(3)], axis=1)
    grain = np.random.default_rng(3).normal(0.0, 2.0, lcd.shape)
    a[y0:y1, x0:x1] = np.clip(smooth[:, None, :] + grain, 0, 255)
    img.paste(Image.fromarray(a.astype(np.uint8)))


def write_label(img: Image.Image, text: str = "AIBOY") -> None:
    """The product name where "GAME BOY" was: same ink, same height,
    slightly widened like the original lettering, with the print's soft edge."""
    x0, y0, x1, y1 = LABEL
    f = font(30)
    layer = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(layer)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x0 + 8 - bbox[0], (y0 + y1) // 2 - th // 2 - bbox[1] + 1), text, font=f, fill=255)
    # stretch the lettering horizontally (the original logo is wide) then soften
    stretched = layer.crop((x0, y0, x0 + tw + 16, y1)).resize((int((tw + 16) * 1.2), y1 - y0), Image.LANCZOS)
    layer.paste(0, (0, 0, *layer.size))
    layer.paste(stretched, (x0, y0))
    layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    ink = Image.new("RGB", img.size, LABEL_INK)
    img.paste(ink, (0, 0), layer)


def make(src: Path = ORIG, dst: Path = OUT) -> Image.Image:
    img = Image.open(src).convert("RGB")
    fill_from_neighbours(img, LABEL)
    write_label(img)
    clear_lcd(img)
    img.save(dst, optimize=True)
    return img


if __name__ == "__main__":
    if not ORIG.exists():
        sys.exit(f"missing {ORIG}")
    out = make()
    print(f"wrote {OUT} ({out.size[0]}x{out.size[1]})")
