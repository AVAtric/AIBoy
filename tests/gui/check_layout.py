"""Every tab and the screen panel must fit the 1300x900 window."""
import tkinter as tk
from _common import say

from aiboy.gui import app as gui

root = tk.Tk(); root.withdraw()
app = gui.AIboyGUI(root); root.update()
W, H = 1300, 900
sw, sh = app.canvas.master.winfo_reqwidth(), app.canvas.master.winfo_reqheight()
avail_h, left_w = H - 140, W - 30 - sw
ok = sh <= H - 85          # panel sits beside the tabs: only top bar + status bar above/below
say(f"screen panel {sw}x{sh}")
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
    say(f"{name:8s} {rw}x{rh} / {left_w}x{avail_h} {'ok' if fits else 'OVERFLOW'}")
app._closing = True; root.destroy()
print("LAYOUT_OK" if ok else "LAYOUT_PROBLEM")
raise SystemExit(0 if ok else 1)
