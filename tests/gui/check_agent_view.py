"""The agent's view under Tracking: play a model with the checkbox on, and
the table must show Mario (-1) and the ground (.5) where the tile
observation has them, refresh while the agent plays, and go away with the
checkbox. Needs a trained Mario model (any run's best_model.zip). ~20 s."""
import sys, time, tkinter as tk
from _common import say, silence_dialogs

from aiboy.gui import app as gui
rec = silence_dialogs(gui)
root = tk.Tk(); app = gui.AIboyGUI(root)
state = {"s": "start", "changes": 0, "last": None}
SHOT = sys.argv[1] if len(sys.argv) > 1 else None

def fail(msg):
    say("FAIL: " + msg); app._on_close(); sys.exit(1)

def cells():
    v = app.agent_view
    return [[v.itemcget(v._texts[r][c], "text") for c in range(20)] for r in range(16)]

def tick():
    if time.time() - __import__("_common").T0 > 120:
        fail(f"timeout in {state['s']} (table changes {state['changes']}, status "
             f"'{app.play_status_var.get()}', wanted {app.player.view_wanted.is_set()}, "
             f"errors {rec['errors']}, log tail {app.log_text.get('end-8l', 'end')!r})")
    s = state["s"]
    if s == "start":
        app.refresh_models()
        label = next((l for l, pth in app._model_paths.items() if pth.name == "best_model.zip"), None)
        if label is None: fail("no best_model.zip to play")
        app.model_var.set(label); app._on_model_selected()
        app.view_var.set(True); app._on_view_toggled()
        app.play_episodes_var.set(1); app.play_speed_label_var.set("1× (real time)")
        if not app.start_playing(): fail("play did not start: " + str(rec["errors"]))
        state["s"], state["t"] = "playing", time.time()
    elif s == "playing":
        if not app._track_shown.get("view"): fail("the table is not shown while the agent plays")
        grid = cells()
        if grid != state["last"]:
            state["changes"] += 1; state["last"] = grid
        if time.time() - state["t"] > 6 and state["changes"] >= 5:
            flat = [x for row in grid[1:] for x in row]
            if "-1" not in flat: fail("no Mario (-1) in the table")
            if ".5" not in flat: fail("no ground (.5) in the table")
            say(f"table refreshed {state['changes']} times; row 0 (HUD): {grid[0][:7]}")
            for row in grid:
                say(" ".join(f"{x:>3}" for x in row))
            if SHOT:
                root.update()
                x, y = app.tracking_frame.winfo_rootx(), app.gameboy.winfo_rooty()
                import subprocess
                w = app.tracking_frame.winfo_width() + (app.tracking_frame.winfo_rootx() - app.gameboy.winfo_rootx())
                subprocess.run(["screencapture", "-x", "-R",
                                f"{app.gameboy.winfo_rootx()},{app.gameboy.winfo_rooty()},{w},{app.tracking_frame.winfo_height()}",
                                SHOT])
            app.view_var.set(False); app._on_view_toggled(); root.update()
            if app._track_shown.get("view"): fail("the table stayed after unchecking")
            if app.player.view_wanted.is_set(): fail("the player still builds grids")
            app.stop_playing(); state["s"] = "done"
    elif s == "done" and not app.playing_active():
        say("AGENT_VIEW_OK"); app._on_close(); return
    root.after(200, tick)

root.after(300, tick); root.mainloop()
