"""Reusable Tk pieces shared by the tabs.

- `FIELDS` / `ConfigForm`: the declarative description of every training
  parameter and the Basic / Advanced form built from it. The Train tab and
  the Presets tab both use the same form, so a preset always shows the same
  fields with the same widgets and validation.
- `make_table`: a Treeview with one monospaced font and a fixed row height
  for every row, so highlighted rows are the same size as the others.
- `Tooltip` / `tooltip()` / `info_icon()`: explanations that appear on
  hover instead of taking up space on the tab. The window keeps only the
  controls and one-line status texts; everything else is a hover away.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont, ttk

from aiboy.games import OBS_TYPES, level_choices
from aiboy.presets import DEFAULT_CONFIG
from aiboy.runs import cpu_count

# Monospaced fonts for tables, logs and status lines. Lists, not tuples:
# `setup_styles()` swaps in the family this desktop actually has (Menlo is
# macOS-only) and every module that imported the name sees the change, so
# widgets created afterwards line up on Windows and Linux too.
MONO = ["Menlo", 10]
MONO_BOLD = ["Menlo", 11, "bold"]
MONO_FAMILIES = ("Menlo", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Courier New")
TABLE_STYLE = "Mono.Treeview"


class Theme:
    """Colours for the current appearance. `setup_styles()` picks light or
    dark from the real window background, so the same code reads well on a
    Mac in dark mode and on a light desktop. Always read attributes at
    widget-creation time (`THEME.muted`), never copy them at import."""

    def __init__(self):
        self.apply(dark=False)

    def apply(self, dark: bool) -> None:
        self.dark = dark
        if dark:
            self.text_soft = "#c8c8c8"      # explanatory paragraphs
            self.muted = "#9a9a9a"          # hints, notes
            self.accent = "#6cb6ff"
            self.done = "#e0e0e0"           # completed wizard steps
            self.row_best = ("#2e4a72", "#ffffff")
            self.row_user = ("#28472c", "#ffffff")
            self.row_modified = ("#5a4520", "#ffffff")
            self.panel = ("#2a2a2a", "#e6e6e6")   # read-only text panels
            self.ok, self.warn, self.err = "#5fd07a", "#f0b34a", "#ff6b6b"
        else:
            self.text_soft = "#444444"
            self.muted = "#6f6f6f"
            self.accent = "#1f6feb"
            self.done = "#222222"
            self.row_best = ("#dce9fb", "#000000")
            self.row_user = ("#e3f3e5", "#000000")
            self.row_modified = ("#fff1d6", "#000000")
            self.panel = ("#f4f4f4", "#111111")
            self.ok, self.warn, self.err = "#1e7e34", "#b26a00", "#c0392b"


THEME = Theme()


def detect_dark(root: tk.Misc) -> bool:
    """True if the window background is dark (macOS dark mode, dark themes)."""
    try:
        bg = ttk.Style(root).lookup("TLabel", "background") or root.cget("background")
        r, g, b = root.winfo_rgb(bg)
    except tk.TclError:
        return False
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 65535.0
    return luminance < 0.5


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str                   # "int" | "float" | "choice"
    group: str                  # "basic" | "ppo" | "run"
    lo: float = 0
    hi: float = 0
    inc: float = 1
    choices: tuple[str, ...] = ()
    entry: bool = False         # plain Entry instead of a Spinbox (scientific notation)
    help: str = ""              # shown on hover over the label and the input


FIELDS: tuple[FieldSpec, ...] = (
    # Basic: what and how long
    FieldSpec("timesteps", "Timesteps", "int", "basic", 1000, 200_000_000, 10_000,
              help="How many game steps to train in total. More steps take longer and usually "
                   "give a better player."),
    FieldSpec("n_envs", f"Envs ({cpu_count()} cores)", "int", "basic", 1, 32,
              help="How many games run in parallel. One per core is a good default; beyond "
                   "about 12 the learning update, not the games, sets the pace."),
    FieldSpec("obs_type", "Obs type", "choice", "basic", choices=tuple(OBS_TYPES),
              help="What the agent sees. 'tiles': the level's tile map (fast, recommended). "
                   "'pixels': the raw screen through a small CNN (much slower)."),
    FieldSpec("start_level", "Start level", "choice", "basic", choices=tuple(level_choices()),
              help="Where every attempt starts: the campaign from 1-1 ('default'), a random "
                   "level, level by level, a marathon through all levels, or one fixed level."),
    FieldSpec("device", "Device", "choice", "basic", choices=("cpu", "auto", "mps", "cuda"),
              help="Where the network is computed. 'cpu' is fastest for this small network; "
                   "'mps' / 'cuda' use the GPU, 'auto' lets PyTorch decide."),
    FieldSpec("seed", "Seed", "int", "basic", 0, 2_147_483_647,
              help="Random seed. The same seed with the same settings gives a repeatable run."),
    # PPO: the learning algorithm
    FieldSpec("learning_rate", "Learning rate", "float", "ppo", entry=True,
              help="Learning speed. Too high diverges after the first level, too low never "
                   "leaves 1-1. Typical values: 0.0001 to 0.001."),
    FieldSpec("ent_coef", "Entropy coef", "float", "ppo", 0.0, 0.5, 0.01,
              help="Curiosity: how much random exploration is rewarded. Raise it if the agent "
                   "keeps getting stuck at one obstacle."),
    FieldSpec("n_steps", "n_steps", "int", "ppo", 32, 8192, 32,
              help="Steps each game plays before the network reviews them (the rollout). "
                   "Longer rollouts see whole levels."),
    FieldSpec("batch_size", "Batch size", "int", "ppo", 8, 4096, 8,
              help="How many steps are learned from at once. Bigger batches make fewer, "
                   "smoother updates."),
    FieldSpec("n_epochs", "Epochs", "int", "ppo", 1, 30,
              help="Passes over each rollout. More squeezes more out of it but risks "
                   "over-fitting to that rollout."),
    FieldSpec("gamma", "Gamma", "float", "ppo", 0.9, 0.9999, 0.005,
              help="Foresight: how much future reward counts. Long levels favour values "
                   "close to 1."),
    FieldSpec("gae_lambda", "GAE lambda", "float", "ppo", 0.8, 1.0, 0.01,
              help="Smoothing of the reward estimate, from 0.8 (steadier, biased) to 1.0 "
                   "(noisier, unbiased)."),
    FieldSpec("clip_range", "Clip range", "float", "ppo", 0.05, 0.5, 0.05,
              help="Step size: how far one update may move the policy. Smaller is safer "
                   "but slower."),
    # Run: input shape and cadence
    FieldSpec("action_repeat", "Action repeat", "int", "run", 1, 16,
              help="Emulator frames each chosen action is held. 4 means the agent decides "
                   "15 times per second of game time."),
    FieldSpec("frame_stack", "Frame stack", "int", "run", 1, 16,
              help="How many recent observations the agent sees at once, so it can tell "
                   "which way things move."),
    FieldSpec("checkpoint_freq", "Checkpoint every", "int", "run", 100, 10_000_000, 1000,
              help="Save a resumable snapshot every this many steps."),
    FieldSpec("eval_freq", "Eval every", "int", "run", 100, 10_000_000, 1000,
              help="Play test rounds every this many steps; the model with the best test "
                   "score is kept as best_model.zip."),
    FieldSpec("n_eval_episodes", "Eval episodes", "int", "run", 1, 100,
              help="Test rounds per evaluation. More give a steadier score but take longer."),
    FieldSpec("time_budget", "Time budget /400", "int", "run", 0, 400, 10,
              help="Game-clock units (of the 400 a level starts with) an attempt may use "
                   "before it ends, which counts as a death. 0 = the game's own timer."),
    FieldSpec("stall_steps", "Stall limit (0=off)", "int", "run", 0, 5000, 50,
              help="End the attempt, counted as a death, after this many steps without "
                   "getting further. 300 is at least 20 s of game time: standing at a "
                   "hard spot must not be the safe choice. 0 = never."),
)
GROUP_TITLES = {"basic": "Basic", "ppo": "PPO", "run": "Input & cadence"}
FIELD_BY_KEY = {f.key: f for f in FIELDS}


class WidgetLock:
    """Disable a set of inputs while a run is active and restore each one to
    the state it had before: an editable Combobox comes back editable, a
    read-only one read-only. (Guessing "readonly" for every Combobox used to
    make the Train tab's run-name box untypeable after the first run.)"""

    def __init__(self):
        self._saved: dict[tk.Widget, str] = {}

    def apply(self, widgets, disabled: bool) -> None:
        for w in widgets:
            try:
                if disabled:
                    if w not in self._saved:
                        self._saved[w] = str(w.cget("state"))
                    w.config(state="disabled")
                elif w in self._saved:          # never locked: already enabled
                    w.config(state=self._saved.pop(w))
            except tk.TclError:
                pass


def mono_family(root: tk.Misc) -> str:
    """The first of MONO_FAMILIES installed here, else Tk's own fixed font."""
    try:
        installed = set(tkfont.families(root))
        for family in MONO_FAMILIES:
            if family in installed:
                return family
        return tkfont.nametofont("TkFixedFont").actual("family")
    except tk.TclError:
        return MONO_FAMILIES[0]


def setup_styles(root: tk.Misc) -> None:
    THEME.apply(detect_dark(root))
    MONO[0] = MONO_BOLD[0] = mono_family(root)
    style = ttk.Style(root)
    style.configure(TABLE_STYLE, font=MONO, rowheight=22)
    style.configure(f"{TABLE_STYLE}.Heading", font=("Helvetica", 11))
    # The aqua theme wraps a notebook's pane in 18/8/18/17 px of empty
    # margin, so the tab box ended 17 px above the Game Boy panel next to it
    # and was narrower than its column. Without the margin both boxes share
    # the same top and bottom edge (other themes have no such margin).
    style.configure("TNotebook", padding=0)


class Tooltip:
    """A small popup with explanatory text, shown after the pointer rests on
    `widget` for a moment. `text` may be a string or a callable returning
    the current text (so a tooltip can follow a setting); an empty text
    shows nothing. `show_at()` / `hide()` let a canvas drive it by hand."""

    DELAY_MS = 450
    WRAP = 360

    def __init__(self, widget: tk.Widget, text, wrap: int = WRAP):
        self.widget = widget
        self.text = text
        self.wrap = wrap
        self._after: str | None = None
        self._win: tk.Toplevel | None = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", lambda e: self.hide(), add="+")
        widget.bind("<ButtonPress>", lambda e: self.hide(), add="+")
        widget.bind("<Destroy>", lambda e: self.hide(), add="+")

    def current_text(self) -> str:
        return str(self.text() if callable(self.text) else self.text or "")

    def schedule(self, _event=None) -> None:
        """Show the popup after the usual delay (as hovering the widget does)."""
        self._cancel()
        self._after = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self) -> None:
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self) -> None:
        self._after = None
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 18
        self.show_at(x, y)

    def show_at(self, x: int, y: int, text: str | None = None) -> None:
        """Show the popup with its top-left corner at screen position (x, y)."""
        text = self.current_text() if text is None else text
        if not text:
            self.hide()
            return
        if self._win is None or not self._win.winfo_exists():
            try:
                win = tk.Toplevel(self.widget)
            except tk.TclError:
                return
            win.wm_overrideredirect(True)
            try:
                win.tk.call("::tk::unsupported::MacWindowStyle", "style", win._w, "help", "none")
            except tk.TclError:
                pass
            self._label = tk.Label(win, text=text, justify="left", wraplength=self.wrap,
                                   background="#fffbe6", foreground="#222222",
                                   relief="solid", borderwidth=1, padx=8, pady=5)
            self._label.pack()
            self._win = win
        else:
            self._label.config(text=text)
        win = self._win
        win.update_idletasks()
        # Keep the popup on the screen the pointer is on.
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        x = max(0, min(x, sw - w - 4))
        y = max(0, min(y, sh - h - 4))
        win.geometry(f"+{x}+{y}")
        win.deiconify()
        win.lift()

    def hide(self, _event=None) -> None:
        self._cancel()
        if self._win is not None:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None


def tooltip(widget: tk.Widget, text, *widgets: tk.Widget) -> Tooltip:
    """Attach the same hover text to one or more widgets; returns the first
    Tooltip (its `text` can be replaced later)."""
    first = Tooltip(widget, text)
    for w in widgets:
        Tooltip(w, text)
    return first


def info_icon(parent: tk.Misc, text, **grid_or_pack) -> ttk.Label:
    """A small ⓘ that explains something on hover. The caller places it
    (`.grid()` / `.pack()`), exactly like a Label."""
    lbl = ttk.Label(parent, text="ⓘ", foreground=THEME.accent, cursor="question_arrow",
                    font=("Helvetica", 13))
    Tooltip(lbl, text)
    return lbl


class ConfigForm(ttk.Frame):
    """Basic, PPO and Input & cadence parameter groups side by side.

    `vars` maps field key -> Tk variable; `widgets` lists every input so the
    owner can lock the form while a run is active.
    """

    def __init__(self, parent: tk.Misc, on_change=None):
        super().__init__(parent)
        self.vars: dict[str, tk.Variable] = {}
        self.widgets: list[tk.Widget] = []
        self._on_change = on_change
        self._lock = WidgetLock()
        groups: dict[str, ttk.LabelFrame] = {}
        for col, (key, title) in enumerate(GROUP_TITLES.items()):
            self.columnconfigure(col, weight=1)
            frame = ttk.LabelFrame(self, text=title, padding=6)
            frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 3, 0))
            groups[key] = frame
        rows = {key: 0 for key in GROUP_TITLES}
        for spec in FIELDS:
            frame = groups[spec.group]
            frame.columnconfigure(1, weight=1)
            var = self._make_var(spec)
            self.vars[spec.key] = var
            widget = self._make_widget(frame, spec, var)
            label = ttk.Label(frame, text=f"{spec.label}:")
            label.grid(row=rows[spec.group], column=0, sticky="w", pady=1)
            widget.grid(row=rows[spec.group], column=1, sticky="ew", pady=1, padx=(6, 0))
            if spec.help:
                tooltip(label, spec.help, widget)
            rows[spec.group] += 1
            self.widgets.append(widget)
            if on_change is not None:
                var.trace_add("write", lambda *_a: on_change())

    @staticmethod
    def _make_var(spec: FieldSpec) -> tk.Variable:
        default = DEFAULT_CONFIG[spec.key]
        if spec.kind == "int":
            return tk.IntVar(value=int(default))
        if spec.kind == "float":
            return tk.DoubleVar(value=float(default))
        return tk.StringVar(value=str(default))

    @staticmethod
    def _make_widget(frame: tk.Misc, spec: FieldSpec, var: tk.Variable) -> tk.Widget:
        if spec.kind == "choice":
            return ttk.Combobox(frame, textvariable=var, values=list(spec.choices),
                                state="readonly", width=9)
        if spec.entry:
            return ttk.Entry(frame, textvariable=var, width=11)
        return ttk.Spinbox(frame, from_=spec.lo, to=spec.hi, increment=spec.inc,
                           textvariable=var, width=10)

    def set_config(self, cfg: dict) -> None:
        """Fill the form; fields missing from `cfg` fall back to DEFAULT_CONFIG."""
        for key, var in self.vars.items():
            var.set(cfg.get(key, DEFAULT_CONFIG[key]))

    def get_config(self) -> dict:
        """Form values as a preset dict (without `game`). Raises ValueError
        naming the field if a value is not a number of the right kind."""
        cfg: dict = {}
        for spec in FIELDS:
            raw = self.vars[spec.key]
            try:
                val = raw.get()
                if spec.kind == "int":
                    cfg[spec.key] = int(val)
                elif spec.kind == "float":
                    cfg[spec.key] = float(val)
                else:
                    cfg[spec.key] = str(val)
            except (tk.TclError, ValueError, TypeError):
                raise ValueError(f"'{spec.label}' must be a "
                                 f"{'whole number' if spec.kind == 'int' else 'number'}") from None
            if spec.kind == "choice" and spec.choices and cfg[spec.key] not in spec.choices:
                raise ValueError(f"'{spec.label}' must be one of {', '.join(spec.choices)}")
        return cfg

    def set_enabled(self, enabled: bool) -> None:
        self._lock.apply(self.widgets, disabled=not enabled)


def make_table(parent: tk.Misc, columns: list[tuple[str, str, int, str, bool]],
               height: int = 8) -> ttk.Treeview:
    """Treeview + scrollbar gridded into `parent` (row 0). `columns` entries are
    (id, heading, width, anchor, stretch). Rows tagged "best" get a highlight
    background, rows tagged "muted" a grey foreground; both keep the shared
    font so every row has the same height."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)
    tree = ttk.Treeview(parent, columns=[c[0] for c in columns], show="headings",
                        height=height, style=TABLE_STYLE)
    for cid, heading, width, anchor, stretch in columns:
        tree.heading(cid, text=heading)
        tree.column(cid, width=width, minwidth=min(width, 40), anchor=anchor, stretch=stretch)
    tree.tag_configure("best", background=THEME.row_best[0], foreground=THEME.row_best[1])
    tree.tag_configure("user", background=THEME.row_user[0], foreground=THEME.row_user[1])
    tree.tag_configure("modified", background=THEME.row_modified[0],
                       foreground=THEME.row_modified[1])
    tree.tag_configure("muted", foreground=THEME.muted)
    tree.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(parent, command=tree.yview)
    sb.grid(row=0, column=1, sticky="ns")
    tree.config(yscrollcommand=sb.set)
    return tree
