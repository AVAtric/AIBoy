"""Marathon through the GUI: train a short marathon run from the Train tab,
then play it on the screen panel (one life from 1-1) and check the
episode-end report. ~40 s."""
import sys, time, tkinter as tk
from _common import say, silence_dialogs, cleanup

from aiboy import presets, runs
from aiboy.gui import app as gui
RUN = "e2etest-marathon"
rec = silence_dialogs(gui); cleanup([RUN])
root = tk.Tk(); app = gui.AIboyGUI(root)
state = {"s": "start"}

def fail(msg):
    say("FAIL: " + msg); app._on_close(); cleanup([RUN]); sys.exit(1)

def tick():
    if time.time() - __import__("_common").T0 > 420: fail(f"timeout in {state['s']}")
    s = state["s"]
    if s == "start":
        marathon = next(n for n in app.presets_by_name() if "Marathon, all levels" in n and presets.is_builtin(n))
        app.preset_var.set(marathon); app.apply_preset()
        for k, v in (("timesteps", 4000), ("n_envs", 2), ("n_steps", 128), ("batch_size", 128),
                     ("checkpoint_freq", 4000), ("eval_freq", 4000)):
            app.form.vars[k].set(v)
        app.run_name_var.set(RUN); app.resume_var.set(False); app.preview_var.set(False)
        if app.form.vars["start_level"].get() != "marathon": fail("preset is not marathon")
        if not app.start_training(): fail("training did not start: " + str(rec["errors"]))
        state["s"] = "training"
    elif s == "training" and app.stat_vars["status"].get() not in ("running", "stopping…", "idle"):
        log = app.log_text.get("1.0", "end")
        if "marathon: every episode is one life from 1-1" not in log: fail("trainer did not report marathon mode")
        cfg = runs.read_run_config("mario", RUN)
        if not cfg or cfg["start_level"] != "marathon": fail("run.json missing marathon")
        say("trained; " + app.stat_vars["status"].get())
        best = runs.best_model_for_run("mario", RUN)
        if not best or not app.select_model(best): fail("model not selectable")
        if app.play_level_var.get() != "marathon": fail("play options did not follow run.json")
        app.play_episodes_var.set(1); app.play_speed_label_var.set("Unlimited"); app.play_max_steps_var.set(600)
        if not app.start_playing(): fail("play did not start: " + str(rec["errors"]))
        state["s"] = "playing"
    elif s == "playing" and app.play_status_var.get().startswith("done"):
        status = app.play_status_var.get(); say("play status: " + status)
        if not any(k in status for k in ("died", "no progress", "step cap", "cleared", "completed")): fail("no end reason in status")
        if app.play_stat_vars["world"].get() == "—": fail("world stat not updated")
        say("MARATHON_OK"); app._on_close(); cleanup([RUN]); return
    root.after(400, tick)

root.after(200, tick); root.mainloop()
