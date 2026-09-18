"""Wizard, Auto-complete: one click -> preset saved -> training -> playing. ~30 s."""
import json, sys, tkinter as tk
from _common import say, silence_dialogs, cleanup

from aiboy import presets, runs, tuning
from aiboy.gui import app as gui
rec = silence_dialogs(gui); cleanup()
root = tk.Tk(); app = gui.AIboyGUI(root); w = app.wizard
state = {"s": "start"}

def fail(msg):
    say("FAIL: " + msg); app._on_close(); cleanup([w.run_name]); sys.exit(1)

def tick():
    import time
    if time.time() - __import__("_common").T0 > 420: fail(f"timeout in {state['s']}: {w.snapshot()}")
    s = state["s"]
    if s == "start":
        w.goal_var.set(presets.SMOKE_TEST_PRESET); w._on_goal_changed()
        # a fixed template: "Let AIboy choose" would replace a hand-written sweep at Start
        w.template_var.set(tuning.TEMPLATE_PLAIN[tuning.DEFAULT_TEMPLATE]); w._on_template_changed()
        w.trial_steps_var.set(4000); w.seeds_var.set(1); w._sync_tune_tab()
        app.set_sweep({"ent_coef": [0.01, 0.03]}, "check sweep")
        w.auto_var.set(True); app.play_episodes_var.set(1); app.play_speed_label_var.set("Unlimited")
        w.start_search()
        if w.phase != "tuning": fail("search did not start")
        state["s"] = "running"
    elif s == "running":
        if w.phase == "training" and not state.get("seen"):
            state["seen"] = True
            say(f"auto: preset '{w.preset_name}' -> run '{w.run_name}'")
            if w.preset_name and (not presets.is_user(w.preset_name) or not w.preset_name.startswith("Quick smoke test · ent ")): fail("preset " + w.preset_name)
            if not w.run_name.startswith("quick-smoke-test"): fail("run name " + w.run_name)
            say("auto decision: " + w.status_var.get()[:110])
            if runs.tune_trial_runs("mario", "e2etest"): fail("search runs not deleted in auto mode")
        if w.phase == "playing": say("auto: playing"); state["s"] = "playing"
        elif w.phase == "idle" and state.get("seen") and w.step != 3: fail("ended idle: " + w.status_var.get())
    elif s == "playing" and w.phase == "idle":
        say("auto: play done -> " + w.watch_result_var.get()); say("AUTO_OK")
        preset, run = w.preset_name, w.run_name
        app._on_close(); cleanup([run])
        if presets.is_user(preset): presets.delete(preset)
        return
    root.after(400, tick)

root.after(200, tick); root.mainloop()
