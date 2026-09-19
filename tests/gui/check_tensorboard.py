"""The TensorBoard button: the program's own `tensorboard` sub-command comes up
and the browser opens on it; with the port taken, a dialog says why. ~20 s."""
import socket
import sys
import time
import tkinter as tk
from _common import say, silence_dialogs, cleanup

from aiboy.gui import app as gui

gui.TENSORBOARD_PORT = 6098                     # never the user's real TensorBoard
rec = silence_dialogs(gui); cleanup()
opened = []
gui.webbrowser.open = lambda url, **k: opened.append(url)
root = tk.Tk(); app = gui.AIboyGUI(root); root.update()
state = {"s": "start", "t": 0.0}


def fail(msg):
    say("FAIL: " + msg); app._on_close(); cleanup(); sys.exit(1)


def stop_tb():
    if app.tb_proc is not None and app.tb_proc.poll() is None:
        app.tb_proc.terminate(); app.tb_proc.wait(timeout=10)


def tick():
    s = state["s"]
    if s == "start":
        app.open_tensorboard()
        if app.tb_proc is None or app.tb_proc.poll() is not None: fail("no TensorBoard process")
        state.update(s="coming-up", t=time.time())
    elif s == "coming-up":
        if opened:
            say(f"browser opened on {opened[-1]}; pid {app.tb_proc.pid}")
            if rec["errors"]: fail(f"errors while coming up: {rec['errors']}")
            stop_tb()
            # The port taken by someone else: refused at once, nothing started.
            state["sock"] = socket.socket(); state["sock"].bind(("localhost", gui.TENSORBOARD_PORT))
            state["sock"].listen(1)
            app.open_tensorboard()
            if not rec["errors"] or "already in use" not in rec["errors"][-1]: fail(f"busy port: {rec['errors']}")
            if app.tb_proc.poll() is None or len(opened) != 1: fail("something started on a busy port")
            say("busy port: " + rec["errors"][-1].splitlines()[0][:70])
            state["sock"].close(); rec["errors"].clear()
            # The process dies right after starting: the dialog carries its output.
            gui.app_command = lambda *a: [sys.executable, "-c", "print('boom'); raise SystemExit(3)"]
            app.open_tensorboard()
            state.update(s="crashed", t=time.time())
    elif s == "crashed":
        if rec["errors"]:
            msg = rec["errors"][-1]; say("dialog: " + msg.splitlines()[0][:80])
            if "stopped right after starting" not in msg or "boom" not in msg or gui.TENSORBOARD_LOG.name not in msg:
                fail("wrong dialog: " + msg)
            if len(opened) != 1: fail("browser opened although TensorBoard died")
            stop_tb(); gui.TENSORBOARD_LOG.unlink(missing_ok=True)
            say("TENSORBOARD_OK"); app._on_close(); cleanup(); return
        elif time.time() - state["t"] > 30: fail("no dialog for the dead process")
    root.after(300, tick)


root.after(200, tick); root.mainloop()
