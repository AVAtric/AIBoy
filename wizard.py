"""Wizard tab: a guided path from "what should it learn?" to "watch it play".

    1 Set up  choose a training goal; optionally search for better settings
    2 Save    keep the chosen settings under a name (optional)
    3 Train   the real training run
    4 Watch   the trained agent plays on the Game Boy screen

Written for people who have never heard of a learning rate: every choice is
a plain question, the expert vocabulary lives on the Tune / Train tabs, and
one Start button runs the whole pipeline (search -> save -> train -> watch)
when "Run everything by itself" is ticked.

The wizard owns no training logic. It fills in the Tune / Train / Play tab
variables and calls the same `start_*` methods the tabs use, so the expert
tabs always show the full picture of what the wizard is doing, and the
GUI's `_pump` reports completion back through the `on_*` hooks below.
"""
from __future__ import annotations

import json
import os
import re
import tkinter as tk
from tkinter import messagebox, ttk

import presets
import runs
import tuning
from widgets import FIELD_BY_KEY, MONO, MONO_BOLD, THEME, WidgetLock, make_table

STEPS = ("Set up", "Save", "Train", "Watch")
TITLE = ("Helvetica", 15, "bold")

# Run-name prefix of the wizard's tuning trials. Overridable so automated
# tests never share trial directories with a real wizard session.
WIZARD_PREFIX = os.environ.get("GAMEBOY_WIZARD_PREFIX", "wizard")
KEEP_BEST_TRIALS = 9            # run folders kept during a search; the rest are deleted
INTRO = {
    0: "Train your own Super Mario Land player. Choose what it should learn, decide whether "
       "the app should look for better settings first, and press Start. Every default is "
       "sensible; nothing here needs AI knowledge.",
    1: "These are the settings that will be trained. Give them a name so you can find them "
       "again on the Train tab, or continue without saving.",
    2: "The agent now learns by playing, using every core of this computer. It keeps its best "
       "version as it goes, so you can stop early and still watch what it has learned.",
    3: "The trained agent plays on the Game Boy screen to the right, exactly the way it "
       "learned to.",
}
MODE_PLAIN = {
    "default": "plays through the game from 1-1, lives and all, like a person would",
    "random": "practises a different level every time",
    "sequential": "works through the levels in order, repeating a level until it beats it",
    "marathon": "tries to beat every level in one go",
}
STAT_LABELS = (("status", "status"), ("total_timesteps", "steps trained"),
               ("ep_rew_mean", "average score"), ("ep_len_mean", "episode length"),
               ("fps", "speed (steps/s)"), ("time_elapsed", "elapsed (s)"))
ROM_HINT = ("Put your Super Mario Land ROM file, named mario.gb, into the ROMs folder next to "
            "the app, then click 'Rescan ROMs' at the top.")


def short_goal(preset_name: str) -> str:
    """'Mario — Campaign, recommended (~15 min)' -> 'Campaign, recommended'."""
    name = re.sub(r"^\s*Mario\s*[—-]\s*", "", preset_name)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    return name.strip() or preset_name


def describe_preset(cfg: dict) -> str:
    """One plain sentence about a goal: what the agent does and how long it takes."""
    level = str(cfg.get("start_level", "default"))
    mode = MODE_PLAIN.get(level, f"practises level {level} only")
    eta = tuning.estimate_seconds(1, int(cfg.get("timesteps", 0)), cfg)
    return (f"It {mode}. About {tuning.format_duration(eta)} of training "
            f"({tuning.format_steps(int(cfg.get('timesteps', 0)))} steps, "
            f"{cfg.get('n_envs', '?')} games in parallel).")


SHORT_KEYS = {"learning_rate": "lr", "ent_coef": "ent", "n_steps": "steps", "batch_size": "batch",
              "n_epochs": "epochs", "gamma": "gamma", "gae_lambda": "gae", "clip_range": "clip",
              "n_envs": "envs", "action_repeat": "repeat", "frame_stack": "stack",
              "obs_type": "obs", "start_level": "level", "device": "device"}


def compact_overrides(overrides: dict) -> str:
    """'ent 0.03 · lr 0.0003' — readable summary of tuned values."""
    parts = []
    for key, value in overrides.items():
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{SHORT_KEYS.get(key, key)} {text}")
    return " · ".join(parts)


def default_preset_name(goal: str, overrides: dict) -> str:
    """Preset name for a wizard result: the goal plus the tuned values, e.g.
    'Campaign, recommended · ent 0.03 · lr 0.0003'."""
    base = short_goal(goal)
    return (f"{base} · {compact_overrides(overrides)}" if overrides else f"{base} (wizard)")[:90]


def run_slug(name: str) -> str:
    """Directory-safe run name from a preset name:
    'Campaign, recommended · ent 0.03 · lr 0.0003' -> 'campaign-recommended-ent0.03-lr0.0003'."""
    text = name.lower().replace("(wizard)", "")
    for key, short in SHORT_KEYS.items():
        text = text.replace(f"{short} ", short)          # 'ent 0.03' -> 'ent0.03'
    text = re.sub(r"[^a-z0-9.\-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-.")
    return text or "wizard"


def unique_run_name(game: str, base: str) -> str:
    """`base`, or `base-2`, `base-3`, … if runs with that name already exist."""
    existing = set(runs.list_runs(game))
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def plain_plan(n_variations: int, n_trials: int, search_seconds: float,
               train_seconds: float, search: bool) -> str:
    """The wizard's one-line plan, e.g. 'Plan: try 6 variations (12 short
    training runs, about 12 min), then train the winner for about 15 min.'"""
    train = f"train for about {tuning.format_duration(train_seconds)}"
    if not search:
        return f"Plan: {train} with the goal's own settings."
    return (f"Plan: try {n_variations} variations plus the goal's own settings ({n_trials} short "
            f"training runs, about {tuning.format_duration(search_seconds)}), then {train} with "
            f"the winner. Total about {tuning.format_duration(search_seconds + train_seconds)}.")


class WizardTab:
    def __init__(self, app, parent: ttk.Frame):
        self.app = app
        self.step = 0
        self.phase = "idle"             # idle | tuning | training | playing
        self.goal_name = ""
        self.config: dict = {}
        self.overrides: dict = {}
        self.preset_name = ""
        self.run_name = ""
        self._auto_run_name = ""       # last run name the wizard filled in itself
        self._inputs: list[tk.Widget] = []
        self._input_lock = WidgetLock()
        self._presets: dict[str, dict] = {}
        self._build(parent)
        self.goto(0)

    # ---------- layout ----------

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew")
        self._step_labels: list[ttk.Label] = []
        for i, name in enumerate(STEPS):
            lbl = ttk.Label(header, text=f"  {i + 1}  {name}  ", font=MONO_BOLD, padding=(6, 4),
                            cursor="hand2")
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, step=i: self._click_step(step))
            self._step_labels.append(lbl)
            if i < len(STEPS) - 1:
                ttk.Label(header, text="→", foreground=THEME.muted).pack(side="left")
        self.title_var = tk.StringVar()
        ttk.Label(parent, textvariable=self.title_var, font=TITLE).grid(
            row=1, column=0, sticky="w", pady=(10, 2))

        self.body = ttk.Frame(parent)
        self.body.grid(row=2, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)
        self.panes = [ttk.Frame(self.body) for _ in STEPS]
        for pane in self.panes:
            pane.grid(row=0, column=0, sticky="nsew")
            pane.columnconfigure(0, weight=1)
        self._build_tune(self.panes[0])
        self._build_preset(self.panes[1])
        self._build_train(self.panes[2])
        self._build_watch(self.panes[3])

        self.status_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.status_var, font=MONO, foreground=THEME.muted,
                  wraplength=600).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _intro(self, pane: ttk.Frame, step: int, row: int) -> None:
        ttk.Label(pane, text=INTRO[step], wraplength=600, foreground=THEME.text_soft).grid(
            row=row, column=0, sticky="w", pady=(0, 10))

    def _register(self, *widgets: tk.Widget) -> None:
        self._inputs.extend(widgets)

    # ----- pane 1: set up -----

    def _build_tune(self, pane: ttk.Frame) -> None:
        self._intro(pane, 0, 0)
        self.rom_hint_var = tk.StringVar(value="")
        self.rom_hint = ttk.Label(pane, textvariable=self.rom_hint_var, wraplength=600,
                                  foreground=THEME.err)
        self.rom_hint.grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.rom_hint.grid_remove()                       # shown only while no ROM can run

        goal = ttk.LabelFrame(pane, text="What should it learn?", padding=8)
        goal.grid(row=2, column=0, sticky="ew")
        goal.columnconfigure(1, weight=1)
        ttk.Label(goal, text="Goal:").grid(row=0, column=0, sticky="w", pady=2)
        self.goal_var = tk.StringVar()
        self.goal_combo = ttk.Combobox(goal, textvariable=self.goal_var, state="readonly")
        self.goal_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        self.goal_combo.bind("<<ComboboxSelected>>", lambda e: self._on_goal_changed())
        self.goal_note = ttk.Label(goal, text="", foreground=THEME.muted, wraplength=540)
        self.goal_note.grid(row=1, column=1, sticky="w", padx=6)

        search = ttk.LabelFrame(pane, text="Look for better settings first?", padding=8)
        search.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        search.columnconfigure(1, weight=1)
        self.search_var = tk.StringVar(value="yes")
        rb_yes = ttk.Radiobutton(search, variable=self.search_var, value="yes",
                                 command=self._on_search_choice,
                                 text="Yes — try a few variations and keep the best one (recommended)")
        rb_yes.grid(row=0, column=0, columnspan=2, sticky="w")
        rb_no = ttk.Radiobutton(search, variable=self.search_var, value="no",
                                command=self._on_search_choice,
                                text="No — use the goal's settings as they are")
        rb_no.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(search, text="Vary:").grid(row=2, column=0, sticky="w", pady=(8, 2))
        self.template_var = tk.StringVar(value=tuning.TEMPLATE_PLAIN[tuning.DEFAULT_TEMPLATE])
        self.template_combo = ttk.Combobox(search, textvariable=self.template_var, state="readonly",
                                           values=list(tuning.TEMPLATE_FROM_PLAIN))
        self.template_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=(8, 2))
        self.template_combo.bind("<<ComboboxSelected>>", lambda e: self._on_template_changed())
        self.template_note = ttk.Label(search, text="", foreground=THEME.muted, wraplength=540)
        self.template_note.grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(search, text="Effort:").grid(row=4, column=0, sticky="w", pady=(6, 2))
        effort_row = ttk.Frame(search)
        effort_row.grid(row=4, column=1, sticky="ew", padx=6, pady=(6, 2))
        self.effort_var = tk.StringVar(value=tuning.DEFAULT_EFFORT)
        self.effort_combo = ttk.Combobox(effort_row, textvariable=self.effort_var, state="readonly",
                                         values=list(tuning.EFFORT_LEVELS), width=10)
        self.effort_combo.pack(side="left")
        self.effort_combo.bind("<<ComboboxSelected>>", lambda e: self._on_effort_changed())
        self.effort_note = ttk.Label(effort_row, text="", foreground=THEME.muted)
        self.effort_note.pack(side="left", padx=(8, 0))
        self.advanced_var = tk.BooleanVar(value=False)
        adv_cb = ttk.Checkbutton(search, text="Advanced…", variable=self.advanced_var,
                                 command=self._toggle_advanced)
        adv_cb.grid(row=5, column=1, sticky="w", padx=6, pady=(4, 0))
        self.advanced = ttk.Frame(search)
        ttk.Label(self.advanced, text="Steps per variation:").pack(side="left")
        self.trial_steps_var = tk.IntVar(value=100_000)
        steps_spin = ttk.Spinbox(self.advanced, from_=2000, to=5_000_000, increment=10_000, width=10,
                                 textvariable=self.trial_steps_var, command=self._sync_tune_tab)
        steps_spin.pack(side="left", padx=(6, 16))
        steps_spin.bind("<KeyRelease>", lambda e: self._sync_tune_tab())
        ttk.Label(self.advanced, text="Seeds per variation:").pack(side="left")
        self.seeds_var = tk.IntVar(value=tuning.EFFORT_LEVELS[tuning.DEFAULT_EFFORT][0])
        seeds_spin = ttk.Spinbox(self.advanced, from_=1, to=5, width=4, textvariable=self.seeds_var,
                                 command=self._sync_tune_tab)
        seeds_spin.pack(side="left", padx=(6, 0))
        ttk.Label(self.advanced, text="(the Tune tab shows the full search)",
                  foreground=THEME.muted).pack(side="left", padx=(12, 0))
        self.search_widgets: list[tk.Widget] = [self.template_combo, self.effort_combo, adv_cb,
                                                steps_spin, seeds_spin]

        then = ttk.Frame(search)
        then.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(then, text="Then:").pack(side="left")
        self.auto_var = tk.BooleanVar(value=True)
        auto_cb = ttk.Checkbutton(then, variable=self.auto_var, command=self._update_summary,
                                  text="run everything by itself (search → save → train → watch)")
        auto_cb.pack(side="left", padx=(6, 0))
        self.preview_var = tk.BooleanVar(value=False)
        preview_cb = ttk.Checkbutton(then, variable=self.preview_var,
                                     text="show it playing while it trains (slower)")
        preview_cb.pack(side="left", padx=(12, 0))
        self.summary_var = tk.StringVar(value="")
        ttk.Label(pane, textvariable=self.summary_var, wraplength=600).grid(
            row=4, column=0, sticky="w", pady=(8, 0))
        self._register(self.goal_combo, rb_yes, rb_no, *self.search_widgets, auto_cb, preview_cb)

        btns = ttk.Frame(pane)
        btns.grid(row=5, column=0, sticky="ew", pady=(8, 4))
        self.btn_search = ttk.Button(btns, text="▶ Start", command=self.start)
        self.btn_search.pack(side="left")
        self.btn_search_stop = ttk.Button(btns, text="■ Stop", command=self.app.stop_tuning,
                                          state="disabled")
        self.btn_search_stop.pack(side="left", padx=6)
        self.btn_use_best = ttk.Button(btns, text="Continue with the best →", command=self.use_best,
                                       state="disabled")
        self.btn_use_best.pack(side="right")
        self.btn_clear_search = ttk.Button(btns, text="Delete old search results…",
                                           command=self.clear_search_data)
        self.btn_clear_search.pack(side="right", padx=(0, 6))

        prog = ttk.Frame(pane)
        prog.grid(row=6, column=0, sticky="ew")
        prog.columnconfigure(0, weight=1)
        ttk.Progressbar(prog, mode="determinate", maximum=100,
                        variable=self.app.tune_progress_var).grid(row=0, column=0, sticky="ew")
        ttk.Label(prog, textvariable=self.app.tune_progress_text, width=20, anchor="e").grid(
            row=0, column=1, padx=(6, 0))
        ttk.Label(pane, textvariable=self.app.tune_live_var, font=MONO).grid(
            row=7, column=0, sticky="w", pady=(2, 4))

        res = ttk.LabelFrame(pane, text="Variations tried (best first)", padding=4)
        res.grid(row=8, column=0, sticky="nsew")
        pane.rowconfigure(8, weight=1)
        self.tree = make_table(res, [
            ("rank", "#", 32, "e", False), ("config", "settings", 300, "w", True),
            ("score", "score", 110, "e", False), ("time", "time", 64, "e", False),
        ], height=3)
        self._on_template_changed()
        self._on_effort_changed()
        self._on_search_choice()

    # ----- pane 2: save -----

    def _build_preset(self, pane: ttk.Frame) -> None:
        self._intro(pane, 1, 0)
        box = ttk.LabelFrame(pane, text="Settings", padding=8)
        box.grid(row=1, column=0, sticky="nsew")
        pane.rowconfigure(1, weight=1)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.config_text = tk.Text(box, height=14, font=MONO, wrap="none", relief="flat",
                                   background=THEME.panel[0], foreground=THEME.panel[1],
                                   highlightthickness=0)
        self.config_text.grid(row=0, column=0, sticky="nsew")
        self.config_text.tag_configure("tuned", foreground=THEME.accent, font=("Menlo", 10, "bold"))
        self.config_text.config(state="disabled")

        name_row = ttk.Frame(pane)
        name_row.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        name_row.columnconfigure(1, weight=1)
        ttk.Label(name_row, text="Name:").grid(row=0, column=0, sticky="w")
        self.preset_name_var = tk.StringVar()
        name_entry = ttk.Entry(name_row, textvariable=self.preset_name_var)
        name_entry.grid(row=0, column=1, sticky="ew", padx=6)
        self._register(name_entry)

        btns = ttk.Frame(pane)
        btns.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="← Back", command=lambda: self.goto(0)).pack(side="left")
        self.btn_save_preset = ttk.Button(btns, text="Save & continue →",
                                          command=self.save_and_continue)
        self.btn_save_preset.pack(side="right")
        ttk.Button(btns, text="Continue without saving →",
                   command=lambda: self.goto(2)).pack(side="right", padx=6)

    # ----- pane 3: train -----

    def _build_train(self, pane: ttk.Frame) -> None:
        self._intro(pane, 2, 0)
        form = ttk.LabelFrame(pane, text="Training run", padding=8)
        form.grid(row=1, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Name of this run:").grid(row=0, column=0, sticky="w", pady=2)
        self.run_name_var = tk.StringVar()
        run_entry = ttk.Entry(form, textvariable=self.run_name_var, width=32)
        run_entry.grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(form, text="Training length (steps):").grid(row=1, column=0, sticky="w", pady=2)
        self.timesteps_var = tk.IntVar(value=2_000_000)
        steps_spin = ttk.Spinbox(form, from_=10_000, to=200_000_000, increment=100_000, width=14,
                                 textvariable=self.timesteps_var, command=self._update_train_eta)
        steps_spin.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        steps_spin.bind("<KeyRelease>", lambda e: self._update_train_eta())
        self.train_eta_var = tk.StringVar()
        ttk.Label(form, textvariable=self.train_eta_var, foreground=THEME.muted).grid(
            row=2, column=1, sticky="w", padx=6)
        preview_cb = ttk.Checkbutton(form, text="Show it playing while it trains (a bit slower)",
                                     variable=self.preview_var)
        preview_cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._register(run_entry, steps_spin, preview_cb)

        btns = ttk.Frame(pane)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.btn_train_back = ttk.Button(btns, text="← Back", command=lambda: self.goto(1))
        self.btn_train_back.pack(side="left")
        self.btn_train = ttk.Button(btns, text="▶ Start training", command=self.start_training)
        self.btn_train.pack(side="left", padx=(18, 6))
        self.btn_train_stop = ttk.Button(btns, text="■ Stop", command=self.app.stop_training,
                                         state="disabled")
        self.btn_train_stop.pack(side="left")
        ttk.Button(btns, text="Show log", command=lambda: self.app.show_tab(self.app.train_tab)).pack(
            side="left", padx=(18, 0))
        self.btn_watch = ttk.Button(btns, text="Watch it play →", command=lambda: self.goto(3),
                                    state="disabled")
        self.btn_watch.pack(side="right")

        prog = ttk.Frame(pane)
        prog.grid(row=3, column=0, sticky="ew")
        prog.columnconfigure(0, weight=1)
        ttk.Progressbar(prog, mode="determinate", maximum=100,
                        variable=self.app.train_progress_var).grid(row=0, column=0, sticky="ew")
        ttk.Label(prog, textvariable=self.app.train_progress_text, width=32, anchor="e").grid(
            row=0, column=1, padx=(6, 0))

        stats = ttk.LabelFrame(pane, text="Live", padding=8)
        stats.grid(row=4, column=0, sticky="w", pady=(8, 0))
        for i, (key, label) in enumerate(STAT_LABELS):
            ttk.Label(stats, text=f"{label}:").grid(row=i % 3, column=(i // 3) * 2, sticky="w",
                                                    padx=(0 if i < 3 else 24, 10))
            ttk.Label(stats, textvariable=self.app.stat_vars[key], font=MONO_BOLD, width=22,
                      anchor="w").grid(row=i % 3, column=(i // 3) * 2 + 1, sticky="w")
        self.train_result_var = tk.StringVar()
        ttk.Label(pane, textvariable=self.train_result_var, wraplength=640).grid(
            row=5, column=0, sticky="w", pady=(10, 0))

    # ----- pane 4: watch -----

    def _build_watch(self, pane: ttk.Frame) -> None:
        self._intro(pane, 3, 0)
        box = ttk.LabelFrame(pane, text="Trained agent", padding=8)
        box.grid(row=1, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.model_var = tk.StringVar(value="—")
        ttk.Label(box, textvariable=self.model_var, font=MONO, wraplength=600).grid(
            row=0, column=0, sticky="w")
        ttk.Label(box, foreground=THEME.muted, wraplength=600,
                  text="How many rounds it plays and how fast are set on the screen panel to "
                       "the right; the live numbers update there while it plays.").grid(
            row=1, column=0, sticky="w", pady=(4, 0))

        btns = ttk.Frame(pane)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.btn_play = ttk.Button(btns, text="▶ Play again", command=self.start_watch)
        self.btn_play.pack(side="left")
        self.btn_play_stop = ttk.Button(btns, text="■ Stop", command=self.app.stop_playing,
                                        state="disabled")
        self.btn_play_stop.pack(side="left", padx=4)
        ttk.Button(btns, text="Train another agent", command=self.restart).pack(side="right")
        self.watch_result_var = tk.StringVar()
        ttk.Label(pane, textvariable=self.watch_result_var, wraplength=600).grid(
            row=3, column=0, sticky="w", pady=(8, 0))

    # ---------- navigation ----------

    def _click_step(self, step: int) -> None:
        """Completed steps in the indicator are links back; later ones are not."""
        if self.phase != "idle" or step >= self.step:
            return
        if step >= 1 and not self.config:
            return
        self.goto(step)

    def goto(self, step: int) -> None:
        self.step = step
        for i, lbl in enumerate(self._step_labels):
            lbl.config(foreground=THEME.accent if i == step
                       else (THEME.done if i < step else THEME.muted))
        self.title_var.set(f"Step {step + 1} of {len(STEPS)} — {STEPS[step]}")
        self.panes[step].tkraise()
        if step == 1:
            self._render_config()
            if not self.preset_name_var.get().strip():
                self.preset_name_var.set(default_preset_name(self.goal_name, self.overrides))
        elif step == 2:
            current = self.run_name_var.get().strip()
            if not current or current == self._auto_run_name:
                # Reuse the preset's name (saved or just chosen) for the run.
                base = self.preset_name or default_preset_name(self.goal_name, self.overrides)
                self._auto_run_name = unique_run_name(self.app.game, run_slug(base))
                self.run_name_var.set(self._auto_run_name)
            self.timesteps_var.set(int(self.config.get("timesteps", 2_000_000)))
            self._update_train_eta()
            self.train_result_var.set("")
            self.btn_watch.config(state="normal" if self._best_model() else "disabled")
        elif step == 3:
            best = self._best_model()
            self.model_var.set(f"Model file: {best}" if best is not None else "—")
            self.watch_result_var.set("")
            if best is not None and self.phase == "idle" and not self.app.playing_active():
                self.start_watch()

    def restart(self) -> None:
        if self.app.playing_active():
            self.app.stop_playing()
        if not self.app.busy():
            self.app.clear_screen()
        self.config, self.overrides, self.preset_name, self.run_name = {}, {}, "", ""
        self._auto_run_name = ""
        self.preset_name_var.set("")
        self.run_name_var.set("")
        self.train_result_var.set("")
        self.watch_result_var.set("")
        self.btn_use_best.config(state="disabled")
        self.btn_watch.config(state="disabled")
        self.status_var.set("")
        self.goto(0)

    def set_inputs_disabled(self, disabled: bool) -> None:
        self._input_lock.apply(self._inputs, disabled)
        if not disabled:
            self._on_search_choice()            # the search widgets follow the yes / no choice
        busy = disabled or self.app.busy()
        self.btn_clear_search.config(state="disabled" if busy else "normal")
        self.btn_save_preset.config(state="disabled" if busy else "normal")
        self.btn_train_back.config(state="disabled" if busy else "normal")
        self.btn_play.config(state="disabled" if busy else "normal")

    # ---------- step 1: set up ----------

    def on_presets_changed(self, names: list[str]) -> None:
        self._presets = self.app.presets_by_name()
        self.goal_combo["values"] = names
        if self.goal_var.get() not in names:
            self.goal_var.set(presets.RECOMMENDED_PRESET if presets.RECOMMENDED_PRESET in names
                              else (names[0] if names else ""))
        self._on_goal_changed()

    def set_goal(self, preset_name: str) -> None:
        """Select `preset_name` as the wizard's training goal (Presets tab)."""
        if preset_name in self._presets:
            self.goal_var.set(preset_name)
            self._on_goal_changed()

    def _on_goal_changed(self) -> None:
        cfg = self._presets.get(self.goal_var.get())
        self.goal_note.config(text=describe_preset(cfg) if cfg else "")
        self._on_effort_changed()

    def _on_effort_changed(self) -> None:
        """Effort level -> seeds and trial length for the chosen goal."""
        level = self.effort_var.get()
        self.effort_note.config(text=tuning.EFFORT_NOTES.get(level, ""))
        cfg = self._presets.get(self.goal_var.get())
        if cfg and level in tuning.EFFORT_LEVELS:
            seeds, steps = tuning.effort_plan(int(cfg.get("timesteps", 0)), level)
            self.seeds_var.set(seeds)
            self.trial_steps_var.set(steps)
        self._sync_tune_tab()

    def _template_name(self) -> str:
        """The Tune-tab template behind the plain name shown in the wizard."""
        return tuning.TEMPLATE_FROM_PLAIN.get(self.template_var.get(), tuning.DEFAULT_TEMPLATE)

    def _on_template_changed(self) -> None:
        self.template_note.config(text=tuning.TEMPLATE_PLAIN_NOTES.get(self._template_name(), ""))
        self._sync_tune_tab()

    def _on_search_choice(self) -> None:
        searching = self.search_var.get() == "yes"
        for w in self.search_widgets:
            try:
                if not searching:
                    w.config(state="disabled")
                elif isinstance(w, ttk.Combobox):
                    w.config(state="readonly")
                else:
                    w.config(state="normal")
            except tk.TclError:
                pass
        self._update_summary()

    def _toggle_advanced(self) -> None:
        if self.advanced_var.get():
            self.advanced.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        else:
            self.advanced.grid_forget()

    def _sync_tune_tab(self) -> None:
        """Mirror the wizard's choices into the Tune tab, which owns the plan."""
        app = self.app
        if app.busy():
            return
        app.tune_preset_var.set(self.goal_var.get())
        template = self._template_name()
        if app.tune_template_var.get() != template:
            app.tune_template_var.set(template)
            app.apply_tune_template()
        try:
            app.tune_trial_steps_var.set(int(self.trial_steps_var.get()))
            app.tune_seeds_var.set(int(self.seeds_var.get()))
        except (tk.TclError, ValueError):
            pass
        search, n_random = tuning.template_search(template)
        app.tune_search_type_var.set(search)
        if n_random:
            app.tune_n_random_var.set(n_random)
        app.tune_run_prefix_var.set(WIZARD_PREFIX)
        app.tune_keep_best_var.set(KEEP_BEST_TRIALS)
        app.tune_metric_var.set(tuning.METRIC_LATE)
        app.tune_skip_done_var.set(True)
        app.tune_baseline_var.set(True)
        app.tune_update_summary()

    def on_tune_plan_changed(self) -> None:
        self._update_summary()

    def _update_summary(self) -> None:
        """One plain sentence: what will happen and how long it takes."""
        if not hasattr(self, "summary_var"):
            return                                  # callback during construction
        cfg = self._presets.get(self.goal_var.get())
        if not cfg:
            self.summary_var.set("")
            return
        train_eta = tuning.estimate_seconds(1, int(cfg.get("timesteps", 0)), cfg)
        searching = self.search_var.get() == "yes"
        plan, err = self.app.tune_plan()
        if searching and err:
            self.summary_var.set(f"⚠ {err}")
            return
        if searching:
            n_trials = len(plan["combos"]) * plan["n_seeds"]
            search_eta = tuning.estimate_seconds(n_trials, plan["trial_steps"], plan["base"])
            n_variations = sum(1 for c in plan["combos"] if c)      # the baseline is not one
            text = plain_plan(n_variations, n_trials, search_eta, train_eta, True)
        else:
            text = plain_plan(0, 0, 0.0, train_eta, False)
        if not self.auto_var.get():
            text += " (Each step waits for you to continue.)"
        self.summary_var.set(text)

    def on_game_changed(self, runnable: bool) -> None:
        self.rom_hint_var.set("" if runnable else ROM_HINT)
        if runnable:
            self.rom_hint.grid_remove()
        else:
            self.rom_hint.grid()
        state = "normal" if runnable and not self.app.busy() else "disabled"
        if self.phase == "idle":
            self.btn_search.config(state=state)
            self.btn_train.config(state=state)
            self.btn_play.config(state=state)

    def start(self) -> None:
        """The Start button: search first, or go straight on with the goal."""
        if self.search_var.get() == "yes":
            self.start_search()
        else:
            self.skip_search()

    def start_search(self) -> None:
        ok, why = self.app.game_runnable()
        if not ok:
            messagebox.showerror("Wizard", why)
            return
        self._sync_tune_tab()
        self.goal_name = self.goal_var.get()
        self._render_candidates()
        if not self.app.start_tuning():
            return
        self.phase = "tuning"
        self.btn_search.config(state="disabled")
        self.btn_search_stop.config(state="normal")
        self.btn_use_best.config(state="disabled")
        self.status_var.set("Trying the variations: each one is a short training run "
                            "(details on the Tune tab).")

    def skip_search(self) -> None:
        ok, why = self.app.game_runnable()
        if not ok:
            messagebox.showerror("Wizard", why)
            return
        cfg = self._presets.get(self.goal_var.get())
        if cfg is None:
            messagebox.showwarning("Wizard", "Pick a goal first.")
            return
        self.goal_name = self.goal_var.get()
        self.overrides = {}
        self.config = presets.normalize(cfg)
        self.preset_name = ""
        self.preset_name_var.set("")
        if self.auto_var.get():
            # Nothing to save: the preset is used as is. Straight to training.
            self.goto(2)
            self.status_var.set(f"Training '{short_goal(self.goal_name)}' with its own settings.")
            self.start_training()
            return
        self.status_var.set(f"Using '{short_goal(self.goal_name)}' as it is.")
        self.goto(1)

    def clear_search_data(self) -> None:
        """Delete the wizard's trial runs and results from earlier searches."""
        if self.phase != "idle":
            return
        if self.app.delete_tune_data(WIZARD_PREFIX):
            self._render_candidates()
            self.btn_use_best.config(state="disabled")
            self.status_var.set("Old search results deleted.")
        elif self.app.tune_results():
            self._render_candidates()

    def on_tune_result(self) -> None:
        if self.phase == "tuning":
            self._render_candidates()

    def on_tune_done(self, text: str, cancelled: bool) -> None:
        if self.phase != "tuning":
            return
        self.phase = "idle"
        self.btn_search.config(state="normal")
        self.btn_search_stop.config(state="disabled")
        self._render_candidates()
        winner, why = tuning.explain_winner(self.app.tune_results())
        if winner is None:
            self.status_var.set("The search " + ("was stopped" if cancelled else "finished")
                                + " before any variation could be scored. The Train tab log "
                                  "has the details.")
            return
        self.btn_use_best.config(state="normal")
        if self.auto_var.get() and not cancelled:
            self._auto_continue(winner)
            return
        note = " (search stopped early)" if cancelled else ""
        self.status_var.set(f"{why}{note} Click 'Continue with the best'.")

    def _auto_continue(self, winner: tuning.ConfigResult) -> None:
        """Auto-complete: winner -> preset (saved under its default name unless
        the winner is the unchanged base preset) -> training. Training's own
        completion hook then switches to Watch and plays."""
        self.use_best()                                   # step 2 with the default name filled in
        if self.overrides:
            name = self.preset_name_var.get().strip()
            if not self.app.save_preset_named(name, self.config, confirm_overwrite=False):
                self.status_var.set("Stopped: the settings could not be saved.")
                return
            self.preset_name = name
            note = f"saved as '{name}'"
        else:
            self.preset_name = ""
            note = f"no variation beat '{short_goal(self.goal_name)}', training it as it is"
        self.goto(2)                                      # fills the run name
        self.status_var.set(f"{note}; training as '{self.run_name_var.get()}'…")
        self.start_training()

    def _render_candidates(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        for rank, r in enumerate(self.app.tune_results(), start=1):
            self.tree.insert("", "end", iid=str(r.index),
                             values=(rank, tuning.plain_overrides(r.overrides), r.score_text(),
                                     tuning.format_duration(r.duration)),
                             tags=("best",) if rank == 1 and r.values else (() if r.values else ("muted",)))

    def use_best(self) -> None:
        best, _why = tuning.pick_winner(self.app.tune_results())
        if best is None:
            return
        self.overrides = dict(best.overrides)
        self.config = presets.normalize(best.config)
        self.preset_name_var.set("")
        self._discard_search_runs()
        self.goto(1)

    def _discard_search_runs(self) -> None:
        """The chosen config is all that matters from here on; the trial runs
        of the search only take up disk space."""
        try:
            n_runs, freed = runs.delete_tune_data(self.app.game, WIZARD_PREFIX)
        except (OSError, ValueError):
            return
        if n_runs:
            for res in self.app.tune_results():
                res.pruned = True
            self.app.render_tune_results()
            self.app.flash(f"Search data deleted: {n_runs} trial run(s), "
                           f"{runs.format_size(freed)} freed.")

    # ---------- step 2: save ----------

    def _render_config(self) -> None:
        self.config_text.config(state="normal")
        self.config_text.delete("1.0", "end")
        if self.goal_name:
            self.config_text.insert("end", f"# based on: {self.goal_name}\n")
        if self.overrides:
            self.config_text.insert("end", "# values found by the search are highlighted\n")
        self.config_text.insert("end", "\n")
        labels = {key: (FIELD_BY_KEY[key].label if key in FIELD_BY_KEY else key)
                  for key in self.config}
        width = max((len(v) for v in labels.values()), default=10)
        for key in presets.PRESET_FIELDS:
            if key not in self.config:
                continue
            line = f"{labels[key]:<{width}}  {self.config[key]}\n"
            self.config_text.insert("end", line, ("tuned",) if key in self.overrides else ())
        self.config_text.config(state="disabled")

    def save_and_continue(self) -> None:
        name = self.preset_name_var.get().strip()
        if not name:
            messagebox.showwarning("Wizard", "Enter a name (or continue without saving).")
            return
        if not self.app.save_preset_named(name, self.config):
            return
        self.preset_name = name
        self.status_var.set(f"Saved as '{name}'. It is now selected on the Train tab.")
        self.goto(2)

    # ---------- step 3: train ----------

    def _update_train_eta(self) -> None:
        try:
            steps = int(self.timesteps_var.get())
        except (tk.TclError, ValueError):
            self.train_eta_var.set("")
            return
        eta = tuning.estimate_seconds(1, steps, self.config)
        self.train_eta_var.set(f"≈ {tuning.format_duration(eta)} on an Apple-Silicon-class CPU")

    def start_training(self) -> None:
        run_name = self.run_name_var.get().strip()
        if not run_name:
            messagebox.showwarning("Wizard", "Enter a name for this run.")
            return
        if not runs.is_run_name(run_name):
            messagebox.showwarning("Wizard", "Run names may not start with '_'.")
            return
        try:
            timesteps = int(self.timesteps_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Wizard", "The training length must be a whole number.")
            return
        cfg = {**self.config, "timesteps": timesteps}
        app = self.app
        app.load_config_into_train(cfg)
        if self.preset_name and self.preset_name in app.presets_by_name():
            app.preset_var.set(self.preset_name)
        else:
            app.preset_var.set("")
        app.run_name_var.set(run_name)
        app.resume_var.set(False)
        app.preview_var.set(bool(self.preview_var.get()))
        if not app.start_training():
            return
        self.run_name = run_name
        self.config = cfg
        self.phase = "training"
        self.btn_train.config(state="disabled")
        self.btn_train_stop.config(state="normal")
        self.btn_watch.config(state="disabled")
        self.train_result_var.set("")
        self.status_var.set(f"Training '{run_name}'… the full log is on the Train tab.")

    def _best_model(self):
        if not self.run_name:
            return None
        return runs.best_model_for_run(self.app.game, self.run_name)

    def on_train_done(self, rc: int, stopped: bool) -> None:
        if self.phase != "training":
            return
        self.phase = "idle"
        self.btn_train.config(state="normal")
        self.btn_train_stop.config(state="disabled")
        best = self._best_model()
        if best is None:
            self.train_result_var.set(
                "Training ended without saving a model (see the Train tab log)."
                if rc != 0 else "Training finished but no model file was found.")
            self.status_var.set("Nothing to play.")
            return
        self.btn_watch.config(state="normal")
        how = "finished" if rc == 0 else ("was stopped" if stopped else f"failed (rc={rc})")
        self.train_result_var.set(f"Training {how}. Model: {best}")
        self.status_var.set("Training done — switching to Watch.")
        if rc == 0 or stopped:
            self.goto(3)

    # ---------- step 4: watch ----------

    def start_watch(self) -> None:
        best = self._best_model()
        if best is None:
            messagebox.showwarning("Wizard", "No trained agent for this run yet.")
            return
        app = self.app
        app.refresh_models()
        if not app.select_model(best):
            messagebox.showerror("Wizard", f"Model not listed on the screen panel: {best}")
            return
        app.sync_play_options(self.config)
        app.play_max_steps_var.set(0)
        app.play_stochastic_var.set(False)
        if not app.start_playing():
            return
        self.phase = "playing"
        self.btn_play.config(state="disabled")
        self.btn_play_stop.config(state="normal")
        self.watch_result_var.set("")
        self.status_var.set(f"Playing {best.name} from run '{self.run_name}'.")

    def on_play_done(self, summary: list[dict]) -> None:
        if self.phase != "playing":
            return
        self.phase = "idle"
        self.btn_play.config(state="normal")
        self.btn_play_stop.config(state="disabled")
        if summary:
            rewards = [s["reward"] for s in summary]
            self.watch_result_var.set(
                f"{len(summary)} round(s) played: average score {sum(rewards) / len(rewards):.0f}, "
                f"best {max(rewards):.0f}.")
        self.status_var.set("Done. Play again, or train another agent.")

    def on_play_error(self, _traceback: str) -> None:
        if self.phase != "playing":
            return
        self.phase = "idle"
        self.btn_play.config(state="normal")
        self.btn_play_stop.config(state="disabled")
        self.status_var.set("Playback failed — see the error dialog.")

    # ---------- test / scripting support ----------

    def snapshot(self) -> dict:
        """Current wizard state (used by the smoke test)."""
        return {"step": self.step, "phase": self.phase, "goal": self.goal_name,
                "overrides": dict(self.overrides), "config": dict(self.config),
                "preset": self.preset_name, "run": self.run_name,
                "config_json": json.dumps(self.config, sort_keys=True)}
