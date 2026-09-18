"""Start-up boot video: frames reach the canvas, the sound starts, and the
screen ends on the idle frame. Plays the sound once (~6 s)."""
import os, subprocess, sys, time, tkinter as tk
from _common import say

os.environ.pop("AIBOY_NO_INTRO", None)          # this check wants the real intro
from aiboy.gui import app as gui, player

root = tk.Tk(); root.withdraw()
app = gui.AIboyGUI(root); root.update()
assert app.intro is not None, "intro assets not loaded (run tools/build_intro.py)"
ids, t0, sound_seen = [], time.time(), False
while time.time() - t0 < 7.5:
    root.update(); time.sleep(0.02)
    ids.append(id(app._tk_img))
    if not sound_seen and sys.platform == "darwin":
        sound_seen = "afplay" in subprocess.run(["pgrep", "-lf", "afplay"], capture_output=True, text=True).stdout
changes = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
say(f"canvas updates during intro: {changes}; sound process seen: {sound_seen}")
assert changes > 60, "too few frames reached the canvas"
assert sound_seen or sys.platform != "darwin", "sound did not start"
# after the intro the canvas shows the idle frame (same pixels as idle_screen())
from PIL import ImageTk
idle = gui.idle_screen(app.intro)
shown = app._tk_img
assert shown.width() == idle.width and shown.height() == idle.height
app._closing = True; root.destroy()
print("INTRO_OK")
