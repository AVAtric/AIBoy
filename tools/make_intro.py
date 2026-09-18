"""Generate the shipped start-up assets: an "AIboy" boot video in the style of
the Game Boy start-up (logo scrolls down, rests, screen clears) plus a
slightly altered version of the original chime.

    python tools/make_intro.py

Reads (optional, not part of the repo)
  assets/orig_gb_intro.wav   the original chime; pitched up one semitone and
                             softened a little for the shipped file. Without
                             it a simple two-note chime is synthesised.
Writes
  assets/gb_intro.npz   frames (N, 144, 160, 3) uint8, fps, idle_index
  assets/gb_intro.wav   mono 22.05 kHz 16-bit
  assets/gb_intro.mp4   the same clip as a video (needs ffmpeg; skipped if absent)

The timeline mirrors the classic boot: ~0.9 s blank, the logo slides down
over ~3.4 s, holds for ~1.2 s, then the screen goes blank for a moment.
"""
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ORIG_WAV = ASSETS / "orig_gb_intro.wav"
NPZ = ASSETS / "gb_intro.npz"
WAV = ASSETS / "gb_intro.wav"
MP4 = ASSETS / "gb_intro.mp4"

W, H = 160, 144
FPS = 29.864
BG = (219, 243, 204)            # lit LCD green
INK = (2, 23, 2)                # darkest LCD shade
BLANK_LEAD = 28                 # frames before the logo appears
SCROLL = 100                    # frames the logo takes to slide into place
HOLD = 39                       # frames it rests (the last of these is the idle frame)
BLANK_TAIL = 7                  # blank frames at the end
LOGO_TOP = 64                   # resting position, same band as the classic logo
LOGO_H = 16
TEXT = "AIboy"
RATE = 22050

FONTS = [
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/Library/Fonts/Verdana Bold.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def logo_mask() -> np.ndarray:
    """Bool (LOGO_H, w) bitmap of the word, rendered chunky and stretched a
    little so it fills the screen like a boot logo."""
    for path in FONTS:
        if Path(path).exists():
            font = ImageFont.truetype(path, 17)
            break
    else:
        font = ImageFont.load_default()
    img = Image.new("L", (W, 32), 255)
    ImageDraw.Draw(img).text((4, 2), TEXT, font=font, fill=0)
    a = np.array(img) < 128
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    wide = Image.fromarray((a * 255).astype(np.uint8)).resize((int(w * 1.5), LOGO_H), Image.BILINEAR)
    return np.array(wide) > 100


def frames() -> tuple[np.ndarray, int]:
    mask = logo_mask()
    h, w = mask.shape
    x0 = (W - w) // 2
    total = BLANK_LEAD + SCROLL + HOLD + BLANK_TAIL
    out = np.empty((total, H, W, 3), dtype=np.uint8)
    out[:] = BG
    ink = np.array(INK, dtype=np.uint8)
    for i in range(SCROLL + HOLD):
        # ease-free linear slide from just above the screen to LOGO_TOP
        t = min(i / SCROLL, 1.0)
        top = int(round(-h + (LOGO_TOP + h) * t))
        frame = out[BLANK_LEAD + i]
        for r in range(h):
            y = top + r
            if 0 <= y < H:
                row = frame[y, x0:x0 + w]
                row[mask[r]] = ink
    idle_index = BLANK_LEAD + SCROLL + HOLD - 1
    return out, idle_index


def sound() -> np.ndarray:
    if ORIG_WAV.exists():
        with wave.open(str(ORIG_WAV)) as wv:
            assert wv.getnchannels() == 1 and wv.getsampwidth() == 2 and wv.getframerate() == RATE
            src = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16).astype(np.float32)
        # one semitone up (resample), then a light 2-tap smoothing to soften it
        ratio = 2 ** (1 / 12)
        n = int(len(src) / ratio)
        pos = np.arange(n) * ratio
        shifted = np.interp(pos, np.arange(len(src)), src)
        shifted = 0.6 * shifted + 0.4 * np.concatenate(([shifted[0]], shifted[:-1]))
        out = np.zeros(len(src), dtype=np.float32)
        out[:n] = shifted
    else:
        # fallback: two short decaying notes at the moment the logo lands
        t = np.arange(int(1.4 * RATE)) / RATE
        note = lambda f, start: (np.sin(2 * np.pi * f * t) * np.exp(-4 * t) * (t >= 0)) * 6000
        chime = np.zeros(int(6.0 * RATE), dtype=np.float32)
        at = int((BLANK_LEAD + SCROLL) / FPS * RATE)
        chime[at:at + len(t)] += note(1046.5, 0)
        chime[at + RATE // 8:at + RATE // 8 + len(t)] += note(2093.0, 0)
        out = chime
    return np.clip(out, -32768, 32767).astype(np.int16)


def write_mp4(fr: np.ndarray, wav: Path) -> None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; mp4 not written")
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{FPS}", "-i", "-",
         "-i", str(wav),
         "-vf", "scale=640:576:flags=neighbor", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(MP4)],
        input=fr.tobytes(), check=True)


def main() -> None:
    fr, idle_index = frames()
    np.savez_compressed(NPZ, frames=fr, fps=np.float32(FPS), idle_index=np.int32(idle_index))
    pcm = sound()
    with wave.open(str(WAV), "wb") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(RATE)
        wv.writeframes(pcm.tobytes())
    write_mp4(fr, WAV)
    print(f"{NPZ.name}: {len(fr)} frames @ {FPS:.2f} fps, idle frame {idle_index} "
          f"({idle_index / FPS:.2f} s), {NPZ.stat().st_size // 1024} KB")
    print(f"{WAV.name}: {len(pcm) / RATE:.2f} s, {WAV.stat().st_size // 1024} KB")
    if MP4.exists():
        print(f"{MP4.name}: {MP4.stat().st_size // 1024} KB")


if __name__ == "__main__":
    sys.exit(main())
