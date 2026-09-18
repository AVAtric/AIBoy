"""Wizard, manual flow: 2-candidate search (keep best 1) -> continue with best
(search data deleted) -> save preset -> train -> autoplay. ~30 s."""
import sys, tkinter as tk
from _common import say, silence_dialogs, cleanup

from aiboy import presets, runs, tuning
from aiboy.gui import app as gui, wizard
wizard.KEEP_BEST_TRIALS = 1
RUN = "e2etest-run"
rec = silence_dialogs(gui); cleanup([RUN])
root = tk.Tk(); app = gui.AIboyGUI(root); w = app.wizard
state = {"s": "search"}

def fail(msg):
    say("FAIL: " + msg); app._on_close(); cleanup([RUN]); sys.exit(1)

def tick():
    import time
    if time.time() - __import__("_common").T0 > 420: fail(f"timeout in {state['s']}: {w.snapshot()}")
    s = state["s"]
    if s == "search":
        w.goal_var.set(presets.SMOKE_TEST_PRESET); w._on_goal_changed()
        # a fixed template: "Let AIboy choose" would replace a hand-written sweep at Start
        w.template_var.set(tuning.TEMPLATE_PLAIN[tuning.DEFAULT_TEMPLATE]); w._on_template_changed()
        w.trial_steps_var.set(4000); w.seeds_var.set(1); w._sync_tune_tab()
        app.set_sweep({"ent_coef": [0.01, 0.03]}, "check sweep")
        w.auto_var.set(False)                                  # the manual flow: continue by hand
        w.start_search()
        if w.phase != "tuning": fail("search did not start: " + w.status_var.get())
        state["s"] = "searching"
    elif s == "searching" and w.phase == "idle":
        kept = runs.tune_trial_runs("mario", "e2etest"); say(f"after search (keep best 1): {kept}")
        if len(kept) != 1: fail(f"expected 1 kept trial run, found {kept}")
        kp = runs.run_paths("mario", kept[0])
        if kp["checkpoints"].exists() or kp["tensorboard"].exists(): fail("trial not slimmed")
        if len(w.tree.get_children()) != 3 or str(w.btn_use_best.cget("state")) != "normal": fail("results/use-best state")
        say("winner: " + w.status_var.get()[:100])
        w.use_best()
        if runs.tune_trial_runs("mario", "e2etest") or (runs.MODELS_ROOT / "mario" / "_tune" / "e2etest.json").exists(): fail("search data not deleted")
        say("search data deleted: " + app.status_bar_var.get()[:60])
        w.preset_name_var.set("__check_e2e_preset"); w.save_and_continue()
        if w.step != 2 or not presets.is_user("__check_e2e_preset"): fail("preset step")
        w.run_name_var.set(RUN); w.timesteps_var.set(4000)
        app.play_episodes_var.set(1); app.play_speed_label_var.set("Unlimited")
        w.start_training()
        if w.phase != "training": fail("training did not start: " + w.status_var.get())
        state["s"] = "training"
    elif s == "training":
        if w.phase == "playing":
            say("training done -> autoplay; " + w.train_result_var.get()[:60]); state["s"] = "playing"
            ck = runs.run_paths("mario", RUN)["checkpoints"]
            if list(ck.glob("ppo_*")): fail("finished run still has step snapshots")
        elif w.phase == "idle": fail("training ended without autoplay: " + w.status_var.get())
    elif s == "playing" and w.phase == "idle":
        say("play done: " + w.watch_result_var.get()); say("disk readout: " + app.disk_var.get())
        if not w.watch_result_var.get(): fail("no watch result")
        say("E2E_OK"); app._on_close(); cleanup([RUN]); return
    root.after(400, tick)

root.after(200, tick); root.mainloop()
