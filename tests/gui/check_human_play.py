"""Play the game yourself through the GUI, without touching a key: start
"Play yourself" from the title screen, feed the Game Boy synthetic key
events (Enter to start, then hold → and X), and check that frames reach the
LCD, the LED is on, the held buttons light up, Mario's numbers appear in
the live panel, and that stopping puts everything to rest. ~15 s. The key
events are handed to the key handlers directly: Tk delivers generated key
events to the focused window only, and the app is not necessarily the
active application while the check runs.

Needs ROMs/mario.gb. No controller is needed; if one is connected its name
is printed."""
import sys, time, tkinter as tk
from _common import say, silence_dialogs

from aiboy.gui import app as gui

rec = silence_dialogs(gui)
root = tk.Tk(); app = gui.AIboyGUI(root)
state = {"s": "start", "frames0": 0, "lit": 0, "led": 0, "t0": 0.0, "worlds": set(), "x": 0}


def fail(msg):
    say(f"FAIL: {msg} (status: {app.play_status_var.get()!r}, errors: {rec['errors']})")
    app._on_close(); sys.exit(1)


class _Key:                       # what KeyboardInput / the dialog read from a Tk key event
    def __init__(self, keysym): self.keysym = keysym


def key(sym, press=True):
    (app.keyboard._on_press if press else app.keyboard._on_release)(_Key(sym))


def tick():
    if time.time() - __import__("_common").T0 > 120: fail(f"timeout in {state['s']}")
    s = state["s"]
    if s == "start":
        if app.gamepad.pad_name: say(f"controller: {app.gamepad.pad_name}")
        assert app.btn_human.cget("text").startswith("🎮"), app.btn_human.cget("text")
        app.human_start_var.set(gui.HUMAN_FROM_START)
        if not app.start_human_play(): fail("play yourself did not start")
        assert app.play_mode == "human" and app.playing_active()
        assert app.btn_human.cget("text").startswith("■"), "button did not turn into Stop"
        assert app._live_labels["reward"].cget("text") == "score:", "live panel not in human wording"
        root.lift()
        state["s"] = "booting"; state["t0"] = time.time(); state["frames0"] = app.frames_painted
    elif s == "booting":
        if time.time() - state["t0"] > 3.0:               # past the Nintendo logo, at the title
            key("Return"); root.after(120, lambda: key("Return", False))
            state["s"] = "started"; state["t0"] = time.time()
    elif s == "started":
        if time.time() - state["t0"] > 1.5:
            key("Right"); key("x")                       # walk right and jump, held
            state["s"] = "running"; state["t0"] = time.time()
    elif s == "running":
        if app.gameboy.pressed: state["lit"] += 1
        if app.gameboy.power: state["led"] += 1
        w = app.play_stat_vars["world"].get()
        if w != "—": state["worlds"].add(w)
        if app.play_stat_vars["x"].get().isdigit(): state["x"] = max(state["x"], int(app.play_stat_vars["x"].get()))
        if time.time() - state["t0"] > 4.0:
            held = app.held.held()
            if held != {"right", "a"}: fail(f"held buttons {set(held)} != {{right, a}}")
            if app.play_stat_vars["action"].get() != "RIGHT+A": fail("action stat not RIGHT+A")
            key("Right", False); key("x", False)
            app.held.set("mouse", frozenset({"b"}))       # a click on the picture's B
            state["s"] = "mouse"; state["t0"] = time.time()
    elif s == "mouse":
        if time.time() - state["t0"] > 0.5:
            if app.held.held() != {"b"}: fail(f"mouse press not held: {set(app.held.held())}")
            app.held.set("mouse", frozenset())
            # Re-bind A to the M key through the Controls window: click the
            # cell, press M; the change is saved and drives the running game.
            app.open_controls(); dlg = app._controls_dialog
            dlg.begin_capture("a", "keys")
            if dlg.cells[("a", "keys")].cget("text") != "press…": fail("cell not waiting for a key")
            dlg._on_key(_Key("m"))
            if app.controls.keys["a"] != ["m"]: fail(f"A not re-bound: {app.controls.keys['a']}")
            if dlg.cells[("a", "keys")].cget("text") != "M": fail("cell does not show the new key")
            import json; from aiboy import settings
            if json.loads(settings.SETTINGS_FILE.read_text())["controls"]["keys"]["a"] != ["m"]: fail("not saved")
            if app.controls.key_hints()["a"] != "M": fail("hover hints not updated")
            dlg.close()
            state["s"] = "rebound"; state["t0"] = time.time()
    elif s == "rebound":
        if time.time() - state["t0"] > 0.5:              # the focus change has settled
            key("m"); key("x")                           # M is A now, X no longer
            state["s"] = "rebound2"; state["t0"] = time.time()
    elif s == "rebound2":
        if time.time() - state["t0"] > 0.5:
            if app.held.held() != {"a"}: fail(f"re-bound key not held as A: {set(app.held.held())}")
            key("m", False); key("x", False)
            app.controls.reset(); app.controls.save(); app.on_controls_changed()
            app.toggle_human_play()                      # the same button stops
            state["s"] = "stopping"; state["t0"] = time.time()
    elif s == "stopping":
        if not app.playing_active() and app.play_status_var.get().startswith("you played"):
            frames = app.frames_painted - state["frames0"]
            say(f"frames painted: {frames}, lit ticks: {state['lit']}, LED ticks: "
                f"{state['led']}, worlds: {sorted(state['worlds'])}, furthest x: {state['x']}, "
                f"end: {app.play_status_var.get()!r}")
            if frames < 300: fail("too few frames reached the LCD (display slower than 30 fps)")
            if state["lit"] < 10: fail("held buttons never lit up")
            if not state["led"]: fail("LED never on")
            if not state["worlds"]: fail("no world reached the live panel (game did not start?)")
            if state["x"] <= 0: fail("Mario never moved: Enter did not start the game or → was not held")
            if app.gameboy.pressed or app.gameboy.power: fail("pad or LED still on after play")
            if app.held.held(): fail("buttons still held after play")
            if app.keyboard.enabled: fail("keyboard still enabled after play")
            if not app.btn_human.cget("text").startswith("🎮"): fail("button not reset")
            if rec["errors"]: fail("errors were shown")
            say("HUMAN_PLAY_OK"); app._on_close(); return
    root.after(50, tick)


root.after(200, tick); root.mainloop()
