"""Human input for playing a game yourself: the keyboard, the mouse on the
Game Boy picture, and a USB / Bluetooth game controller.

Every source keeps the set of Game Boy buttons it is holding right now in
one shared `HeldButtons`; the emulator thread reads the union once per
frame and presses / releases what changed (see player.EmbeddedPlayer.
play_human). Nothing here touches the emulator.

  ControlMap      which keys and which controller inputs drive each Game
                  Boy button; editable (controls_dialog.py), remembered in
                  settings.json under "controls"
  KeyboardInput   Tk KeyPress / KeyRelease on the Game Boy canvas
  Gamepad         a thread polling SDL2's game-controller API (PySDL2 is
                  already a dependency through PyBoy), so any pad SDL
                  knows works: Switch Pro Controller, Xbox, PlayStation,
                  8BitDo… Every connected pad plays; hot-plugging is
                  handled; the names and the kind of the connected pads
                  are reported to the GUI, and the raw inputs they hold
                  are exposed so the dialog can bind "the next button you
                  press" and show what the app hears (`watch` prints the
                  same on the command line: main.py controller-test).

Game Boy button names: up, down, left, right, a, b, select, start.
Controller inputs are SDL's generic names ("a", "dpad_up", "leftx-",
"triggerright"…); what the person sees are the labels of their console
(`pad_label`): a PlayStation "a" is ✕, a Switch "back" is −.
"""
from __future__ import annotations

import threading
import warnings
from collections.abc import Callable

from aiboy import settings

BUTTONS = ("up", "down", "left", "right", "a", "b", "select", "start")
BUTTON_TITLES = {"up": "D-pad up", "down": "D-pad down", "left": "D-pad left",
                 "right": "D-pad right", "a": "A", "b": "B", "select": "SELECT",
                 "start": "START"}

# ---------- keyboard ----------

# Tk keysyms, normalised (see normalize_keysym): letters lower-case, the
# modifier pairs collapsed ("Shift_L" / "Shift_R" -> "Shift").
DEFAULT_KEYS: dict[str, list[str]] = {
    "up": ["Up", "w"], "down": ["Down", "s"], "left": ["Left", "a"], "right": ["Right", "d"],
    "a": ["x", "k", "space"], "b": ["z", "j"],
    "start": ["Return", "KP_Enter"], "select": ["Shift", "BackSpace"],
}
MODIFIERS = ("Shift", "Control", "Alt", "Meta", "Super")
KEY_NAMES = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→", "space": "Space",
             "Return": "Enter", "KP_Enter": "Enter (keypad)", "BackSpace": "Backspace",
             "Tab": "Tab", "Escape": "Esc", "Meta": "Cmd", "Alt": "Option / Alt",
             "Control": "Ctrl", "Shift": "Shift", "Super": "Super", "Caps_Lock": "Caps Lock",
             "comma": ",", "period": ".", "slash": "/", "semicolon": ";", "apostrophe": "'",
             "minus": "-", "plus": "+", "equal": "=", "bracketleft": "[", "bracketright": "]",
             "backslash": "\\", "grave": "`"}


def normalize_keysym(keysym: str) -> str:
    """One name per physical key: 'A' and 'a' are the same key, so are
    Shift_L and Shift_R."""
    if len(keysym) == 1:
        return keysym.lower()
    for mod in MODIFIERS:
        if keysym.startswith(mod + "_"):
            return mod
    return keysym


def key_label(keysym: str) -> str:
    """How a key is shown: '↑', 'W', 'Space', 'Enter'."""
    if keysym in KEY_NAMES:
        return KEY_NAMES[keysym]
    if len(keysym) == 1:
        return keysym.upper()
    if keysym.startswith("KP_"):
        return f"{keysym[3:]} (keypad)"
    return keysym.replace("_", " ")


# ---------- game controller ----------

# Every controller input the mapping can use, in the order the dialog
# lists them: SDL's buttons, then the sticks' four directions and the
# triggers (an axis past AXIS_THRESHOLD counts as pressed).
PAD_BUTTONS = ("a", "b", "x", "y", "back", "guide", "start", "leftstick", "rightstick",
               "leftshoulder", "rightshoulder", "dpad_up", "dpad_down", "dpad_left",
               "dpad_right", "misc1", "paddle1", "paddle2", "paddle3", "paddle4", "touchpad")
PAD_AXES = ("leftx-", "leftx+", "lefty-", "lefty+", "rightx-", "rightx+", "righty-", "righty+",
            "triggerleft", "triggerright")
PAD_INPUTS = PAD_BUTTONS + PAD_AXES
AXIS_THRESHOLD = 0.5            # fraction of full deflection before a stick / trigger counts

DEFAULT_PAD: dict[str, list[str]] = {
    "up": ["dpad_up", "lefty-"], "down": ["dpad_down", "lefty+"],
    "left": ["dpad_left", "leftx-"], "right": ["dpad_right", "leftx+"],
    # Face buttons follow their labels (a Switch Pro's A is A) and the two
    # on the same side double up (X = A, Y = B), so both the "label" and
    # the "position" habit work; the shoulders too.
    "a": ["a", "x", "rightshoulder"], "b": ["b", "y", "leftshoulder"],
    "start": ["start"], "select": ["back"],
}

# What the inputs are called on each console. SDL names the face buttons
# by label on Nintendo pads and by position elsewhere ("a" = south), so
# the tables mirror that.
_COMMON = {"leftstick": "left stick click", "rightstick": "right stick click",
           "dpad_up": "D-pad ↑", "dpad_down": "D-pad ↓", "dpad_left": "D-pad ←",
           "dpad_right": "D-pad →", "leftx-": "left stick ←", "leftx+": "left stick →",
           "lefty-": "left stick ↑", "lefty+": "left stick ↓", "rightx-": "right stick ←",
           "rightx+": "right stick →", "righty-": "right stick ↑", "righty+": "right stick ↓",
           "misc1": "extra", "paddle1": "paddle 1", "paddle2": "paddle 2", "paddle3": "paddle 3",
           "paddle4": "paddle 4", "touchpad": "touchpad"}
PAD_LABELS: dict[str, dict[str, str]] = {
    "switch": {**_COMMON, "a": "A", "b": "B", "x": "X", "y": "Y", "back": "−", "start": "+",
               "guide": "Home", "leftshoulder": "L", "rightshoulder": "R",
               "triggerleft": "ZL", "triggerright": "ZR", "misc1": "Capture"},
    "xbox": {**_COMMON, "a": "A", "b": "B", "x": "X", "y": "Y", "back": "View", "start": "Menu",
             "guide": "Xbox", "leftshoulder": "LB", "rightshoulder": "RB",
             "triggerleft": "LT", "triggerright": "RT", "misc1": "Share"},
    "playstation": {**_COMMON, "a": "✕", "b": "○", "x": "□", "y": "△", "back": "Share",
                    "start": "Options", "guide": "PS", "leftshoulder": "L1",
                    "rightshoulder": "R1", "triggerleft": "L2", "triggerright": "R2",
                    "misc1": "Mute"},
    "generic": {**_COMMON, "a": "A (south)", "b": "B (east)", "x": "X (west)", "y": "Y (north)",
                "back": "Back", "start": "Start", "guide": "Guide", "leftshoulder": "LB",
                "rightshoulder": "RB", "triggerleft": "LT", "triggerright": "RT"},
}


def pad_label(raw: str, kind: str = "generic") -> str:
    """The console's own name for an SDL input ('a' on a PlayStation pad is ✕)."""
    return PAD_LABELS.get(kind, PAD_LABELS["generic"]).get(raw, raw)


def pad_kind_of(sdl_type_name: str) -> str:
    """'switch' / 'xbox' / 'playstation' / 'generic' from an
    SDL_CONTROLLER_TYPE_* constant name."""
    t = sdl_type_name.upper()
    if "SWITCH" in t:
        return "switch"
    if "XBOX" in t:
        return "xbox"
    if "PS3" in t or "PS4" in t or "PS5" in t:
        return "playstation"
    return "generic"


def axis_inputs(name: str, value: float, threshold: float = AXIS_THRESHOLD) -> frozenset[str]:
    """Raw inputs an SDL axis at `value` (-1..1) holds: a trigger past the
    threshold is 'triggerleft'; a stick is 'leftx-' / 'leftx+' and so on
    (y grows downwards, so 'lefty-' is up)."""
    if name.startswith("trigger"):
        return frozenset({name}) if value >= threshold else frozenset()
    if value <= -threshold:
        return frozenset({name + "-"})
    if value >= threshold:
        return frozenset({name + "+"})
    return frozenset()


# ---------- the mapping ----------

def _clean(table, allowed: tuple[str, ...] | None, defaults: dict) -> dict[str, list[str]]:
    """A {button: [inputs]} table from anything (a settings file), keeping
    only known buttons and, for the pad, known inputs. A keyboard has more
    keys than we can list, so key names are kept as they are."""
    if not isinstance(table, dict):
        return {b: list(v) for b, v in defaults.items()}
    out: dict[str, list[str]] = {}
    for button in BUTTONS:
        raw = table.get(button, [])
        if isinstance(raw, str):
            raw = [raw]
        names = [str(v) for v in raw if isinstance(v, str) and v] if isinstance(raw, list) else []
        if allowed is not None:
            names = [v for v in names if v in allowed]
        out[button] = list(dict.fromkeys(names))
    return out


class ControlMap:
    """Which keys and which controller inputs drive each Game Boy button.

    Every input drives at most one button (binding it elsewhere moves it).
    `key_to_button` / `pad_to_button` are rebuilt on every change and
    swapped in whole, so the input threads can read them without a lock.
    """

    SETTING = "controls"

    def __init__(self, keys: dict | None = None, pad: dict | None = None):
        self.keys = _clean(keys if keys is not None else DEFAULT_KEYS, None, DEFAULT_KEYS)
        self.pad = _clean(pad if pad is not None else DEFAULT_PAD, PAD_INPUTS, DEFAULT_PAD)
        self._rebuild()

    # ----- persistence -----

    @classmethod
    def load(cls) -> ControlMap:
        saved = settings.get(cls.SETTING)
        if not isinstance(saved, dict):
            saved = {}
        return cls(saved.get("keys"), saved.get("pad"))

    def save(self) -> None:
        settings.put(self.SETTING, self.to_dict())

    def to_dict(self) -> dict:
        return {"keys": {b: list(v) for b, v in self.keys.items()},
                "pad": {b: list(v) for b, v in self.pad.items()}}

    def is_default(self) -> bool:
        return self.keys == DEFAULT_KEYS and self.pad == DEFAULT_PAD

    # ----- editing -----

    def _rebuild(self) -> None:
        self.key_to_button = {k: b for b, ks in self.keys.items() for k in ks}
        self.pad_to_button = {r: b for b, rs in self.pad.items() for r in rs}

    def _bind(self, table: dict, button: str, name: str, add: bool) -> None:
        if button not in BUTTONS:
            raise ValueError(f"unknown Game Boy button {button!r}")
        for other in table.values():          # an input drives one button only
            if name in other:
                other.remove(name)
        current = table[button] if add else []
        table[button] = current + [name]
        self._rebuild()

    def bind_key(self, button: str, keysym: str, add: bool = False) -> None:
        """Make `keysym` drive `button`: replacing its keys, or added to
        them with `add`."""
        self._bind(self.keys, button, normalize_keysym(keysym), add)

    def bind_pad(self, button: str, raw: str, add: bool = False) -> None:
        if raw not in PAD_INPUTS:
            raise ValueError(f"unknown controller input {raw!r}")
        self._bind(self.pad, button, raw, add)

    def clear_key(self, button: str) -> None:
        self.keys[button] = []
        self._rebuild()

    def clear_pad(self, button: str) -> None:
        self.pad[button] = []
        self._rebuild()

    def reset(self) -> None:
        self.keys = {b: list(v) for b, v in DEFAULT_KEYS.items()}
        self.pad = {b: list(v) for b, v in DEFAULT_PAD.items()}
        self._rebuild()

    # ----- wording -----

    def key_text(self, button: str) -> str:
        """'↑ or W'; '—' for none."""
        return " or ".join(key_label(k) for k in self.keys[button]) or "—"

    def pad_text(self, button: str, kind: str = "generic") -> str:
        return " or ".join(pad_label(r, kind) for r in self.pad[button]) or "—"

    def key_hints(self) -> dict[str, str]:
        """Button -> the keys that drive it, for the picture's hover labels."""
        return {b: self.key_text(b) for b in BUTTONS}

    def help_text(self, kind: str = "generic", pad_name: str | None = None) -> str:
        """The whole scheme in a few lines, for the ⓘ next to "Play yourself"."""
        keys = " · ".join(f"{BUTTON_TITLES[b]} = {self.key_text(b)}" for b in BUTTONS)
        pad = " · ".join(f"{BUTTON_TITLES[b]} = {self.pad_text(b, kind)}" for b in BUTTONS)
        who = f"Controller ({pad_name})" if pad_name else "Controller"
        return (f"Keyboard: {keys}.\n{who}: {pad}.\n"
                "The buttons on the picture can be clicked too. Controls… changes the keys "
                "and the controller buttons; a Switch Pro, Xbox or PlayStation pad plugged "
                "in by USB or paired by Bluetooth is picked up while the app runs.")


# What each held combination is called in the live panel and for the glow
# on the picture (gameboy.buttons_for_action understands these words).
ACTION_WORDS = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
                "a": "A", "b": "B", "select": "SELECT", "start": "START"}


def action_name(held: frozenset[str]) -> str:
    """'RIGHT+A' for {'right', 'a'}; 'NOOP' for nothing."""
    if not held:
        return "NOOP"
    return "+".join(ACTION_WORDS[b] for b in BUTTONS if b in held)


def diff_presses(before: frozenset[str], after: frozenset[str]) -> tuple[list[str], list[str]]:
    """(newly pressed, newly released) between two held sets, in BUTTONS order."""
    pressed = [b for b in BUTTONS if b in after and b not in before]
    released = [b for b in BUTTONS if b in before and b not in after]
    return pressed, released


class HeldButtons:
    """Thread-safe union of what every input source holds right now."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sources: dict[str, frozenset[str]] = {}

    def set(self, source: str, held: frozenset[str] | set[str]) -> None:
        with self._lock:
            if held:
                self._sources[source] = frozenset(held)
            else:
                self._sources.pop(source, None)

    def held(self) -> frozenset[str]:
        with self._lock:
            if not self._sources:
                return frozenset()
            return frozenset().union(*self._sources.values())

    def clear(self) -> None:
        with self._lock:
            self._sources.clear()


class KeyboardInput:
    """Keys held on `widget` (it must have the keyboard focus), mapped by
    `controls`. Key auto-repeat on X11 arrives as release/press pairs, so
    a release only counts if no press of the same key follows within a
    few ms."""

    REPEAT_GRACE_MS = 40

    def __init__(self, widget, held: HeldButtons, controls: ControlMap,
                 source: str = "keyboard"):
        self.widget = widget
        self.held = held
        self.controls = controls
        self.source = source
        self._down: set[str] = set()            # normalised keysyms
        self._pending: dict[str, str] = {}      # keysym -> after id of a deferred release
        self.enabled = False
        widget.bind("<KeyPress>", self._on_press, add="+")
        widget.bind("<KeyRelease>", self._on_release, add="+")
        widget.bind("<FocusOut>", lambda e: self.release_all(), add="+")

    def _publish(self) -> None:
        table = self.controls.key_to_button
        self.held.set(self.source, frozenset(table[k] for k in self._down if k in table))

    def _on_press(self, event) -> str | None:
        if not self.enabled:
            return None
        key = normalize_keysym(event.keysym)
        if key not in self.controls.key_to_button:
            return None
        after_id = self._pending.pop(key, None)
        if after_id is not None:
            self.widget.after_cancel(after_id)
        if key not in self._down:
            self._down.add(key)
            self._publish()
        return "break"

    def _on_release(self, event) -> str | None:
        key = normalize_keysym(event.keysym)
        if key not in self._down:
            return None
        if key not in self._pending:
            self._pending[key] = self.widget.after(self.REPEAT_GRACE_MS,
                                                   lambda k=key: self._really_release(k))
        return "break"

    def _really_release(self, key: str) -> None:
        self._pending.pop(key, None)
        if key in self._down:
            self._down.discard(key)
            self._publish()

    def release_all(self) -> None:
        for after_id in self._pending.values():
            try:
                self.widget.after_cancel(after_id)
            except Exception:
                pass
        self._pending.clear()
        self._down.clear()
        self.held.set(self.source, frozenset())

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.release_all()


def _load_sdl():
    """PySDL2, or None when it is not installed / its library is missing.
    Imported lazily and quietly (pysdl2-dll announces itself with a warning)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import sdl2
        return sdl2
    except Exception:
        return None


class _Pads:
    """The game controllers SDL has open right now, keyed by SDL's instance
    id. Lives on the polling thread only (see Gamepad.run)."""

    AXES = ("leftx", "lefty", "rightx", "righty", "triggerleft", "triggerright")

    def __init__(self, sdl2):
        self.sdl2 = sdl2
        self.open: dict[int, tuple[object, str, str]] = {}      # id -> (handle, name, kind)
        self.buttons = [(raw, getattr(sdl2, f"SDL_CONTROLLER_BUTTON_{raw.upper()}"))
                        for raw in PAD_BUTTONS
                        if hasattr(sdl2, f"SDL_CONTROLLER_BUTTON_{raw.upper()}")]
        self.axes = [(name, getattr(sdl2, f"SDL_CONTROLLER_AXIS_{name.upper()}"))
                     for name in self.AXES if hasattr(sdl2, f"SDL_CONTROLLER_AXIS_{name.upper()}")]
        # SDL_CONTROLLER_TYPE_* code -> "switch" / "xbox" / "playstation" / "generic"
        self.kinds = {getattr(sdl2, const): pad_kind_of(const)
                      for const in dir(sdl2) if const.startswith("SDL_CONTROLLER_TYPE_")}

    def __bool__(self) -> bool:
        return bool(self.open)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for _pad, name, _kind in self.open.values())

    @property
    def kind(self) -> str:
        """The first pad's console, which names the inputs in the GUI."""
        for _pad, _name, kind in self.open.values():
            return kind
        return "generic"

    def sync(self) -> bool:
        """Close the pads that went away and open the ones that appeared.
        True if the set of pads changed."""
        sdl2 = self.sdl2
        changed = False
        for iid, (pad, _name, _kind) in list(self.open.items()):
            if not sdl2.SDL_GameControllerGetAttached(pad):
                sdl2.SDL_GameControllerClose(pad)
                del self.open[iid]
                changed = True
        for i in range(sdl2.SDL_NumJoysticks()):
            if not sdl2.SDL_IsGameController(i):
                continue
            iid = int(sdl2.SDL_JoystickGetDeviceInstanceID(i))
            if iid in self.open:
                continue
            pad = sdl2.SDL_GameControllerOpen(i)
            if not pad:
                continue
            raw = sdl2.SDL_GameControllerName(pad)
            name = raw.decode(errors="replace") if raw else "game controller"
            kind = "generic"
            if hasattr(sdl2, "SDL_GameControllerGetType"):
                try:
                    kind = self.kinds.get(int(sdl2.SDL_GameControllerGetType(pad)), "generic")
                except Exception:
                    pass
            self.open[iid] = (pad, name, kind)
            changed = True
        return changed

    def raw_inputs(self) -> frozenset[str]:
        """Every input held on any open pad."""
        sdl2 = self.sdl2
        raw: set[str] = set()
        for pad, _name, _kind in self.open.values():
            raw.update(name for name, btn in self.buttons
                       if sdl2.SDL_GameControllerGetButton(pad, btn))
            for name, axis in self.axes:
                raw |= axis_inputs(name, sdl2.SDL_GameControllerGetAxis(pad, axis) / 32767.0)
        return frozenset(raw)

    def close_all(self) -> None:
        for pad, _name, _kind in self.open.values():
            try:
                self.sdl2.SDL_GameControllerClose(pad)
            except Exception:
                pass
        self.open.clear()


class Gamepad(threading.Thread):
    """Polls every connected game controller on its own thread, maps what
    they hold through `controls` and keeps `held` up to date. All pads play
    at once, so it never matters which one SDL lists first (a pad plugged
    in by USB while it is still paired by Bluetooth can show up twice; a
    second pad in the room may come first).

    `pad_names` / `pad_name` / `pad_kind` describe the connected pads (empty
    / None / "generic" without one); `on_change(name_or_None)` is called from
    the thread when a pad connects or disconnects, or when controller
    support fails (`error` is set then). `raw_held` is the set of raw inputs
    down right now, for the dialog that binds "the next button pressed" and
    for telling the person what the app sees (`raw_text`). `active`
    switches between a fast poll (playing, binding) and a slow one (only
    watching for a pad to appear); setting it wakes the thread at once.

    SDL's joystick layer must be initialised, polled and shut down on the
    same thread, so everything SDL happens inside run()."""

    FAST_PERIOD = 1 / 120
    SLOW_PERIOD = 0.5

    def __init__(self, held: HeldButtons, controls: ControlMap,
                 on_change: Callable[[str | None], None] | None = None,
                 source: str = "gamepad"):
        super().__init__(daemon=True, name="gamepad")
        self.held = held
        self.controls = controls
        self.on_change = on_change
        self.source = source
        self.pad_names: tuple[str, ...] = ()    # every connected controller
        self.pad_kind: str = "generic"          # the first one's console, for labels
        self.raw_held: frozenset[str] = frozenset()
        self.error: str | None = None
        self._active = False
        self._quit = threading.Event()
        self._wake = threading.Event()          # ends a wait early: quit, active change

    # ----- what the GUI reads -----

    @property
    def pad_name(self) -> str | None:
        """The connected controller(s) as one name; None without one."""
        return " + ".join(self.pad_names) if self.pad_names else None

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = bool(value)
        self._wake.set()

    def raw_text(self) -> str:
        """What is pressed on the controller right now, in its own words:
        'A, D-pad ↑'; '' for nothing."""
        return ", ".join(pad_label(r, self.pad_kind)
                         for r in sorted(self.raw_held, key=PAD_INPUTS.index))

    def stop(self) -> None:
        self._quit.set()
        self._wake.set()

    def remap(self) -> None:
        """Publish what is held right now through the current mapping: after
        the mapping changed, and after `held` was cleared (the loop only
        publishes when the raw inputs change)."""
        table = self.controls.pad_to_button
        self.held.set(self.source, frozenset(table[r] for r in self.raw_held if r in table))

    # ----- the thread -----

    def run(self) -> None:
        sdl2 = _load_sdl()
        if sdl2 is None:
            self._fail("PySDL2 is not installed")
            return
        try:
            # Without this SDL drops pad input while no SDL window has the
            # focus (there is none: the window is Tk's). Face buttons are
            # named by their labels on Nintendo pads (SDL's default; pinned).
            # A pair of Joy-Cons counts as one pad.
            sdl2.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")
            sdl2.SDL_SetHint(b"SDL_GAMECONTROLLER_USE_BUTTON_LABELS", b"1")
            sdl2.SDL_SetHint(b"SDL_JOYSTICK_HIDAPI_JOY_CONS", b"1")
            if sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER) != 0:
                self._fail(sdl2.SDL_GetError().decode(errors="replace"))
                return
        except Exception as e:      # a broken SDL build must not take the app down
            self._fail(repr(e))
            return
        pads = _Pads(sdl2)
        try:
            while not self._quit.is_set():
                self._wake.clear()
                sdl2.SDL_GameControllerUpdate()
                if pads.sync():
                    self._set_pads(pads.names, pads.kind)
                raw = pads.raw_inputs()
                if raw != self.raw_held:
                    self.raw_held = raw
                    self.remap()
                self._wake.wait(self.FAST_PERIOD if self._active and pads else self.SLOW_PERIOD)
        except Exception as e:      # an SDL call failed: say so instead of dying quietly
            self._fail(repr(e))
        finally:
            pads.close_all()
            self.raw_held = frozenset()
            self.held.set(self.source, frozenset())
            try:
                sdl2.SDL_Quit()
            except Exception:
                pass

    def _set_pads(self, names: tuple[str, ...], kind: str) -> None:
        if names == self.pad_names and kind == self.pad_kind:
            return
        self.pad_names, self.pad_kind = names, kind
        self._notify()

    def _fail(self, message: str) -> None:
        self.error = message
        self.pad_names, self.pad_kind = (), "generic"
        self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change(self.pad_name)
            except Exception:
                pass


def watch(seconds: float = 20.0, controls: ControlMap | None = None, out=print) -> bool:
    """Show what the app sees from the game controllers for `seconds`: which
    pads connect, every change of what is pressed and what it means on the
    Game Boy (`python main.py controller-test`). For finding out why a pad
    does nothing. True if a controller was seen."""
    import time
    held = HeldButtons()
    pad = Gamepad(held, controls or ControlMap.load(),
                  on_change=lambda n: out(f"[controller] {'connected: ' + n if n else 'no controller connected'}"))
    pad.start()
    pad.active = True
    out(f"[controller] watching for {seconds:.0f} s: press buttons on the pad (Ctrl-C ends)")
    t0 = time.time()
    last = None
    seen = False
    try:
        while time.time() - t0 < seconds and pad.is_alive():
            seen = seen or bool(pad.pad_names)
            now = (pad.raw_held, held.held())
            if now != last:
                out(f"[controller] pressed: {pad.raw_text() or '—':<24} Game Boy: "
                    f"{action_name(held.held())}")
                last = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    if pad.error:
        out(f"[controller] controllers are off: {pad.error}")
    elif not seen:
        out("[controller] no game controller was seen; plug one in by USB or pair it by "
            "Bluetooth and try again")
    pad.stop()
    pad.join(timeout=2)
    return seen
