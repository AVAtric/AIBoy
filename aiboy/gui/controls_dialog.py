"""The "Controls…" window of Play yourself: one row per Game Boy button
with its keyboard keys and its controller inputs. Click a cell, press the
key or the controller button you want, done; the change is saved at once
(settings.json) and applies to a game in progress. The controller column
uses the connected pad's own names (a PlayStation ✕, a Switch −).

Binding "the next thing pressed" works the same way for both columns:
the keyboard cell listens for the next KeyPress on the window, the
controller cell watches `Gamepad.raw_held` for an input that was not down
when the click happened.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from aiboy.gui.controls import BUTTON_TITLES, BUTTONS, ControlMap, Gamepad, key_label, pad_label
from aiboy.gui.widgets import MONO, THEME, tooltip

POLL_MS = 30
NO_PAD = "no controller"


class ControlsDialog:
    """`on_change()` is called after every saved change; `on_close()` when
    the window goes away."""

    def __init__(self, parent: tk.Misc, controls: ControlMap, gamepad: Gamepad,
                 on_change=None, on_close=None):
        self.controls = controls
        self.gamepad = gamepad
        self.on_change = on_change
        self.on_close = on_close
        self.capturing: tuple[str, str, bool] | None = None   # (button, "keys"|"pad", add)
        self._baseline: frozenset[str] = frozenset()
        self._poll: str | None = None
        self.cells: dict[tuple[str, str], ttk.Button] = {}

        win = self.win = tk.Toplevel(parent)
        win.title("Controls — Play yourself")
        win.resizable(False, False)
        win.transient(parent.winfo_toplevel())
        win.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Game Boy", foreground=THEME.muted).grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Keyboard", foreground=THEME.muted).grid(row=0, column=1, sticky="w",
                                                                       padx=(12, 0))
        self.pad_head_var = tk.StringVar()
        ttk.Label(body, textvariable=self.pad_head_var, foreground=THEME.muted).grid(
            row=0, column=2, sticky="w", padx=(12, 0))
        for i, button in enumerate(BUTTONS, start=1):
            ttk.Label(body, text=BUTTON_TITLES[button]).grid(row=i, column=0, sticky="w", pady=2)
            for col, column in ((1, "keys"), (2, "pad")):
                cell = ttk.Button(body, width=26, style="Cell.TButton",
                                  command=lambda b=button, c=column: self.begin_capture(b, c))
                cell.grid(row=i, column=col, sticky="ew", padx=(12, 0), pady=2)
                cell.bind("<Button-2>", lambda e, b=button, c=column: self._menu(e, b, c))
                cell.bind("<Button-3>", lambda e, b=button, c=column: self._menu(e, b, c))
                self.cells[(button, column)] = cell
                tooltip(cell, lambda b=button, c=column: self._cell_help(b, c))

        self.status_var = tk.StringVar()
        status = ttk.Label(body, textvariable=self.status_var, font=MONO, foreground=THEME.muted,
                           wraplength=560)
        status.grid(row=len(BUTTONS) + 1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        foot = ttk.Frame(body)
        foot.grid(row=len(BUTTONS) + 2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        reset = ttk.Button(foot, text="Reset to defaults", command=self.reset)
        reset.pack(side="left")
        tooltip(reset, "Back to the standard keys (arrows / W A S D, X = A, Z = B, Enter, Shift) "
                       "and the standard controller buttons.")
        ttk.Button(foot, text="Close", command=self.close).pack(side="right")

        try:
            ttk.Style(win).configure("Cell.TButton", anchor="w")
        except tk.TclError:
            pass
        win.bind("<KeyPress>", self._on_key)
        win.bind("<Escape>", lambda e: self.cancel_capture())
        self._was_active = gamepad.active
        gamepad.active = True                   # fast polling while binding
        self.refresh()
        self._poll = win.after(POLL_MS, self._tick)
        win.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        win.geometry(f"+{max(0, px - win.winfo_reqwidth() - 10)}+{py + 40}")
        win.focus_set()

    # ----- display -----

    def refresh(self) -> None:
        kind = self.gamepad.pad_kind
        self.pad_head_var.set(f"Controller ({self.gamepad.pad_name})" if self.gamepad.pad_name
                              else "Controller (none connected)")
        for (button, column), cell in self.cells.items():
            if self.capturing == (button, column, True) or self.capturing == (button, column, False):
                cell.config(text="press…", state="normal")
            elif column == "keys":
                cell.config(text=self.controls.key_text(button), state="normal")
            else:
                cell.config(text=self.controls.pad_text(button, kind)
                            if self.gamepad.pad_name or self.controls.pad[button] else NO_PAD,
                            state="normal" if self.gamepad.pad_name else "disabled")
        if self.capturing is None:
            self.status_var.set("Click a cell, then press the key or controller button you "
                                "want. Right-click a cell to add a second one or to clear it. "
                                "Changes are saved at once.")

    def _cell_help(self, button: str, column: str) -> str:
        what = "key" if column == "keys" else "controller button"
        return (f"Click, then press the {what} that should be {BUTTON_TITLES[button]} (it "
                f"replaces the current one; Esc cancels). Right-click to add another "
                f"{what} instead, or to clear.")

    # ----- capturing -----

    def begin_capture(self, button: str, column: str, add: bool = False) -> None:
        if column == "pad" and not self.gamepad.pad_name:
            self.status_var.set("No controller connected. Plug one in by USB or pair it by "
                                "Bluetooth; it is picked up while the app runs.")
            return
        self.capturing = (button, column, add)
        self._baseline = self.gamepad.raw_held
        self.win.focus_set()
        self.refresh()
        what = "a key" if column == "keys" else f"a button on the {self.gamepad.pad_name}"
        self.status_var.set(f"Press {what} for {BUTTON_TITLES[button]}"
                            f"{' (added to the current ones)' if add else ''}… Esc cancels.")

    def cancel_capture(self) -> None:
        if self.capturing is None:
            return
        self.capturing = None
        self.refresh()

    def _on_key(self, event) -> str | None:
        if self.capturing is None or self.capturing[1] != "keys":
            return None
        if event.keysym == "Escape":
            return None                          # the Escape binding cancels
        button, _column, add = self.capturing
        self.capturing = None
        self.controls.bind_key(button, event.keysym, add=add)
        self._changed(f"{key_label(self.controls.keys[button][-1])} is now {BUTTON_TITLES[button]}.")
        return "break"

    def _tick(self) -> None:
        self._poll = None
        try:
            if self.capturing is not None and self.capturing[1] == "pad":
                new = self.gamepad.raw_held - self._baseline
                if new:
                    button, _column, add = self.capturing
                    raw = sorted(new)[0]
                    self.capturing = None
                    self.controls.bind_pad(button, raw, add=add)
                    self._changed(f"{pad_label(raw, self.gamepad.pad_kind)} is now "
                                  f"{BUTTON_TITLES[button]}.")
                else:
                    self._baseline &= self.gamepad.raw_held     # released inputs may be bound
            elif self.pad_head_var.get() != (
                    f"Controller ({self.gamepad.pad_name})" if self.gamepad.pad_name
                    else "Controller (none connected)"):
                self.refresh()                   # a pad came or went
            self._poll = self.win.after(POLL_MS, self._tick)
        except tk.TclError:
            pass

    # ----- changes -----

    def _changed(self, note: str) -> None:
        self.controls.save()
        self.refresh()
        self.status_var.set(note)
        if self.on_change is not None:
            self.on_change()

    def _menu(self, event, button: str, column: str) -> None:
        menu = tk.Menu(self.win, tearoff=0)
        what = "key" if column == "keys" else "controller button"
        menu.add_command(label=f"Add another {what}…",
                         command=lambda: self.begin_capture(button, column, add=True))
        menu.add_command(label="Clear", command=lambda: self.clear(button, column))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def clear(self, button: str, column: str) -> None:
        if column == "keys":
            self.controls.clear_key(button)
        else:
            self.controls.clear_pad(button)
        self._changed(f"{BUTTON_TITLES[button]} has no {'key' if column == 'keys' else 'controller button'} now.")

    def reset(self) -> None:
        self.controls.reset()
        self._changed("Standard controls restored.")

    def close(self) -> None:
        self.capturing = None
        if self._poll is not None:
            try:
                self.win.after_cancel(self._poll)
            except tk.TclError:
                pass
        self.gamepad.active = self._was_active
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        if self.on_close is not None:
            self.on_close()
