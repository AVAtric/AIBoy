"""Reusable Tk pieces shared by the tabs.

- `FIELDS` / `ConfigForm`: the declarative description of every training
  parameter and the Basic / Advanced form built from it. The Train tab and
  the Presets tab both use the same form, so a preset always shows the same
  fields with the same widgets and validation.
- `make_table`: a Treeview with one monospaced font and a fixed row height
  for every row, so highlighted rows are the same size as the others.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from env import level_choices
from presets import PRESET_DEFAULTS

MONO = ("Menlo", 10)
MONO_BOLD = ("Menlo", 11, "bold")
MUTED = "#777"
ACCENT = "#1f6feb"
BEST_ROW_BG = "#e3eefc"
TABLE_STYLE = "Mono.Treeview"


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str                   # "int" | "float" | "choice"
    group: str                  # "basic" | "advanced"
    lo: float = 0
    hi: float = 0
    inc: float = 1
    choices: tuple[str, ...] = ()
    entry: bool = False         # plain Entry instead of a Spinbox (scientific notation)


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("timesteps", "Total timesteps", "int", "basic", 1000, 200_000_000, 10_000),
    FieldSpec("n_envs", "Parallel envs", "int", "basic", 1, 16),
    FieldSpec("learning_rate", "Learning rate", "float", "basic", entry=True),
    FieldSpec("ent_coef", "Entropy coef", "float", "basic", 0.0, 0.5, 0.01),
    FieldSpec("n_steps", "PPO n_steps", "int", "basic", 32, 8192, 32),
    FieldSpec("batch_size", "PPO batch size", "int", "basic", 8, 4096, 8),
    FieldSpec("obs_type", "Obs type", "choice", "basic", choices=("tiles", "pixels")),
    FieldSpec("start_level", "Start level", "choice", "basic", choices=tuple(level_choices())),
    FieldSpec("device", "Device", "choice", "basic", choices=("cpu", "auto", "mps", "cuda")),
    FieldSpec("action_repeat", "Action repeat", "int", "advanced", 1, 16),
    FieldSpec("frame_stack", "Frame stack", "int", "advanced", 1, 16),
    FieldSpec("n_epochs", "PPO n_epochs", "int", "advanced", 1, 30),
    FieldSpec("gamma", "Gamma (discount)", "float", "advanced", 0.9, 0.9999, 0.005),
    FieldSpec("gae_lambda", "GAE lambda", "float", "advanced", 0.8, 1.0, 0.01),
    FieldSpec("clip_range", "PPO clip range", "float", "advanced", 0.05, 0.5, 0.05),
    FieldSpec("seed", "Seed", "int", "advanced", 0, 2_147_483_647),
    FieldSpec("checkpoint_freq", "Checkpoint / N steps", "int", "advanced", 100, 10_000_000, 1000),
    FieldSpec("eval_freq", "Eval / N steps", "int", "advanced", 100, 10_000_000, 1000),
    FieldSpec("n_eval_episodes", "Eval episodes", "int", "advanced", 1, 100),
)
FIELD_BY_KEY = {f.key: f for f in FIELDS}


def setup_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    style.configure(TABLE_STYLE, font=MONO, rowheight=22)
    style.configure(f"{TABLE_STYLE}.Heading", font=("Helvetica", 11))


class ConfigForm(ttk.Frame):
    """Basic and Advanced parameter groups side by side.

    `vars` maps field key -> Tk variable; `widgets` lists every input so the
    owner can lock the form while a run is active.
    """

    def __init__(self, parent: tk.Misc, on_change=None):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.vars: dict[str, tk.Variable] = {}
        self.widgets: list[tk.Widget] = []
        self._on_change = on_change
        groups = {
            "basic": ttk.LabelFrame(self, text="Basic", padding=8),
            "advanced": ttk.LabelFrame(self, text="Advanced", padding=8),
        }
        groups["basic"].grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        groups["advanced"].grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        rows = {"basic": 0, "advanced": 0}
        for spec in FIELDS:
            frame = groups[spec.group]
            frame.columnconfigure(1, weight=1)
            var = self._make_var(spec)
            self.vars[spec.key] = var
            widget = self._make_widget(frame, spec, var)
            ttk.Label(frame, text=f"{spec.label}:").grid(row=rows[spec.group], column=0,
                                                         sticky="w", pady=1)
            widget.grid(row=rows[spec.group], column=1, sticky="ew", pady=1, padx=(6, 0))
            rows[spec.group] += 1
            self.widgets.append(widget)
            if on_change is not None:
                var.trace_add("write", lambda *_a: on_change())

    @staticmethod
    def _make_var(spec: FieldSpec) -> tk.Variable:
        default = PRESET_DEFAULTS.get(spec.key)
        if spec.kind == "int":
            return tk.IntVar(value=int(default) if default is not None else int(spec.lo))
        if spec.kind == "float":
            return tk.DoubleVar(value=float(default) if default is not None else float(spec.lo))
        return tk.StringVar(value=spec.choices[0] if spec.choices else "")

    @staticmethod
    def _make_widget(frame: tk.Misc, spec: FieldSpec, var: tk.Variable) -> tk.Widget:
        if spec.kind == "choice":
            return ttk.Combobox(frame, textvariable=var, values=list(spec.choices),
                                state="readonly", width=13)
        if spec.entry:
            return ttk.Entry(frame, textvariable=var, width=16)
        return ttk.Spinbox(frame, from_=spec.lo, to=spec.hi, increment=spec.inc,
                           textvariable=var, width=14)

    def set_config(self, cfg: dict) -> None:
        """Fill the form; fields missing from `cfg` fall back to PRESET_DEFAULTS."""
        for key, var in self.vars.items():
            if key in cfg:
                var.set(cfg[key])
            elif key in PRESET_DEFAULTS:
                var.set(PRESET_DEFAULTS[key])

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
        for w in self.widgets:
            if not enabled:
                state = "disabled"
            else:
                state = "readonly" if isinstance(w, ttk.Combobox) else "normal"
            try:
                w.config(state=state)
            except tk.TclError:
                pass


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
    tree.tag_configure("best", background=BEST_ROW_BG)
    tree.tag_configure("muted", foreground=MUTED)
    tree.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(parent, command=tree.yview)
    sb.grid(row=0, column=1, sticky="ns")
    tree.config(yscrollcommand=sb.set)
    return tree
