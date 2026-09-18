"""Every tab, the Preview and the Tracking panel must fit the default window."""
import tkinter as tk
from _common import say

from aiboy.gui import app as gui

root = tk.Tk(); root.withdraw()
app = gui.AIboyGUI(root); root.update()
W, H = gui.WINDOW_W, gui.WINDOW_H
pw, ph = app.preview_frame.winfo_reqwidth(), app.preview_frame.winfo_reqheight()
tw, th = app.tracking_frame.winfo_reqwidth(), app.tracking_frame.winfo_reqheight()
avail_h, left_w = H - 140, W - 40 - pw - tw
ok = max(ph, th) <= H - 85          # panels sit beside the tabs: only top bar + status bar above/below
say(f"preview {pw}x{ph}, tracking {tw}x{th} (limit {H - 85} high) {'ok' if ok else 'OVERFLOW'}")
# Worst case for the wizard: two-line knowledge and plan notes.
app.wizard.known_note.config(text="Best known settings so far: curiosity 0.03 · learning speed "
                                  "0.0003 (score 1234, from 12 short tests of 100k steps).")
app.wizard.template_note.config(text="AIboy knows 14 earlier tests of this goal. Best so far: "
                                     "curiosity 0.03 · learning speed 0.0003 (score 1234). It now "
                                     "tries 6 untested changes around those settings.")
for name, tab in [("wizard", app.wizard_tab), ("train", app.train_tab), ("tune", app.tune_tab),
                  ("presets", app.presets_frame), ("experience", app.experience_frame)]:
    app.nb.select(tab); root.update()
    rw, rh = tab.winfo_reqwidth(), tab.winfo_reqheight()
    fits = rw <= left_w and rh <= avail_h; ok &= fits
    say(f"{name:10s} {rw}x{rh} / {left_w}x{avail_h} {'ok' if fits else 'OVERFLOW'}")
# The buttons light up for the agent's actions and clear for none; the LED follows playback.
app.gameboy.show("RIGHT+RUN+JUMP"); assert app.gameboy.pressed == {"right", "b", "a"}
app.gameboy.show("NOOP"); assert not app.gameboy.pressed
assert not app.gameboy.power; app.gameboy.set_power(True); assert app.gameboy.power
app._closing = True; root.destroy()
print("LAYOUT_OK" if ok else "LAYOUT_PROBLEM")
raise SystemExit(0 if ok else 1)
