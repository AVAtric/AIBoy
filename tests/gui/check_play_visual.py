"""Play through the GUI and watch the Game Boy react: train a tiny model,
play it at real speed, and check mid-play that frames reach the LCD, the
pressed buttons light up, the battery LED is on, and finished rounds are
kept (app.play_rounds). With a directory as first argument, screenshots of
the whole window are written there (macOS). ~60 s."""
import subprocess, sys, time, tkinter as tk
from _common import say, silence_dialogs, cleanup

from aiboy import runs
from aiboy.gui import app as gui
RUN = "e2etest-play"
OUT = sys.argv[1] if len(sys.argv) > 1 else None
rec = silence_dialogs(gui); cleanup([RUN])
root = tk.Tk(); app = gui.AIboyGUI(root)
state = {"s": "start", "lit": 0, "shots": 0, "frames0": 0, "t_play": 0.0}

def fail(msg):
    say(f"FAIL: {msg} (status: {app.play_status_var.get()!r}, errors: {rec['errors']})")
    app._on_close(); cleanup([RUN]); sys.exit(1)

def shot(name):
    if OUT and sys.platform == "darwin":
        x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", f"{OUT}/{name}.png"])

def tick():
    if time.time() - __import__("_common").T0 > 420: fail(f"timeout in {state['s']}")
    s = state["s"]
    if s == "start":
        for k, v in (("timesteps", 4000), ("n_envs", 2), ("n_steps", 128), ("batch_size", 128),
                     ("checkpoint_freq", 4000), ("eval_freq", 4000)):
            app.form.vars[k].set(v)
        app.run_name_var.set(RUN); app.resume_var.set(False); app.preview_var.set(False)
        if not app.start_training(): fail("training did not start: " + str(rec["errors"]))
        state["s"] = "training"
    elif s == "training" and app.stat_vars["status"].get() not in ("running", "stopping…", "idle"):
        best = runs.best_model_for_run("mario", RUN)
        if not best or not app.select_model(best): fail("model not selectable")
        app.play_episodes_var.set(2); app.play_speed_label_var.set("1× (real time)")
        app.play_max_steps_var.set(150)
        if not app.start_playing(): fail("play did not start: " + str(rec["errors"]))
        root.lift(); root.attributes("-topmost", True)          # so screenshots show the app
        state["s"] = "playing"; state["t_play"] = time.time(); state["frames0"] = app.frames_painted
    elif s == "playing":
        if app.gameboy.pressed:
            state["lit"] += 1
            if state["shots"] < 3 and state["lit"] % 4 == 1:
                state["shots"] += 1; shot(f"play-{state['shots']}-{'+'.join(sorted(app.gameboy.pressed))}")
        if app.gameboy.power: state["led"] = state.get("led", 0) + 1
        if app.play_status_var.get().startswith("done"):
            rows = app.play_rounds
            played = [(r["episode"], round(r["reward"]), r["steps"], r["end"]) for r in rows]
            frames = app.frames_painted - state["frames0"]
            say(f"lit ticks: {state['lit']}, LED-on ticks: {state.get('led', 0)}, frames painted: "
                f"{frames}, rounds: {played}, {time.time() - state['t_play']:.1f} s")
            if not state.get("led"): fail("LED never on during play")
            if state["lit"] < 5: fail("buttons never lit up during play")
            if frames < 30: fail("too few frames reached the LCD")
            if len(rows) != 2: fail("rounds incomplete")
            if app.gameboy.pressed or app.gameboy.power: fail("pad or LED still on after play")
            shot("done")
            say("PLAY_VISUAL_OK"); app._on_close(); cleanup([RUN]); return
    root.after(100, tick)

root.after(200, tick); root.mainloop()
