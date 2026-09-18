"""Game selector: ROM discovery, background probe, gating of Start buttons."""
import time
import tkinter as tk
from _common import say, silence_dialogs

from aiboy.gui import app as gui

rec = silence_dialogs(gui)
root = tk.Tk(); app = gui.AIboyGUI(root); root.update()
names = list(app.game_combo["values"]); say(f"roms: {names}")
assert names and names[0] == "mario", names
assert app.game == "mario" and app.current_rom().status()[0] == "ok"
t0 = time.time()
while not all(r.probed for r in app.roms.values()):
    root.update(); time.sleep(0.05)
    assert time.time() - t0 < 120, "probe timeout"
for n, r in app.roms.items():
    say(f"  {n:6s} {r.status()[0]:12s} {r.status()[1][:70]}")
assert app.roms["mario"].status()[0] == "ok"
other = next((n for n, r in app.roms.items() if r.status()[0] != "ok"), None)
if other:
    app.game_var.set(other); app._on_game_changed(); root.update()
    assert str(app.btn_train_start.cget("state")) == "disabled"
    assert str(app.wizard.btn_search.cget("state")) == "disabled"
    assert app.start_training() is False and rec["errors"]
    app.game_var.set("mario"); app._on_game_changed(); root.update()
assert str(app.btn_train_start.cget("state")) == "normal"
assert len(app.preset_combo["values"]) >= 12
app.rescan_roms(); root.update(); assert app.game == "mario"
app._on_close(); print("GAMES_OK")
