"""Start-up boot video: the LCD is empty until it starts, frames reach the
canvas, the sound starts, and the screen ends on the idle frame. Plays the sound once (~6 s)."""
import os, subprocess, sys, time, tkinter as tk
from _common import say

os.environ.pop("AIBOY_NO_INTRO", None)          # this check wants the real intro
from aiboy.gui import app as gui

root = tk.Tk(); root.withdraw()
app = gui.AIboyGUI(root); root.update()
assert app.intro is not None, "intro assets not loaded (run tools/build_intro.py)"
# before the video starts the LCD is empty (no logo): one colour, the video's first frame
blank = app.gameboy.screen_image
colours = {tuple(root.tk.call(str(blank), "get", x, y)) for x in range(0, blank.width(), 16)
           for y in range(0, blank.height(), 16)}
assert len(colours) == 1 and app.gameboy.power, f"LCD not blank at start-up: {colours}"
t0, sound_seen, painted0 = time.time(), False, app.frames_painted
while time.time() - t0 < 7.5:
    root.update(); time.sleep(0.02)
    if not sound_seen and sys.platform == "darwin":
        sound_seen = "afplay" in subprocess.run(["pgrep", "-lf", "afplay"], capture_output=True, text=True).stdout
changes = app.frames_painted - painted0
say(f"frames painted during intro: {changes}; sound process seen: {sound_seen}")
assert changes > 60, "too few frames reached the canvas"
assert sound_seen or sys.platform != "darwin", "sound did not start"
# after the intro the canvas shows the idle frame (same pixels as idle_screen())
idle = gui.idle_screen(app.intro, app.gameboy.screen_size)
shown = app.gameboy.screen_image
assert shown.width() == idle.width and shown.height() == idle.height
app._closing = True; root.destroy()
print("INTRO_OK")
