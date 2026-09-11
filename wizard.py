"""Wizard tab: a guided path from "which settings?" to "watch it play".

    1 Tune    short trials over a sweep of hyperparameters (or skip)
    2 Preset  save the winning config under a name
    3 Train   run the real training with that preset
    4 Watch   play the trained model in the embedded Game Boy

The wizard owns no training logic. It fills in the Tune / Train / Play tab
variables and calls the same `start_*` methods the tabs use, so the expert
tabs always show the full picture of what the wizard is doing, and the
GUI's `_pump` reports completion back through the `on_*` hooks below.
"""
from __future__ import annotations

import json
import re
import time
import tkinter as tk
from tkinter import messagebox, ttk

import presets
import runs
import tuning
from widgets import MONO, MONO_BOLD, THEME, make_table

STEPS = ("Tune", "Preset", "Train", "Watch")
TITLE = ("Helvetica", 15, "bold")

WIZARD_PREFIX = "wizard"        # run-name prefix of tuning trials
INTRO = {
    0: "Pick a training goal and a group of hyperparameters to compare. Every "
       "candidate is trained briefly and scored on its best evaluation reward. "
       "Or skip straight to training with the preset as it is.",
    1: "This is the configuration that will be trained. Give it a name so it "
       "shows up in the preset list of the Train tab, or continue without saving.",
    2: "The real training run. It writes to models/mario/<run name>/ and keeps "
       "the best-scoring model as best_model.zip. You can stop early; the best "
       "model so far is kept.",
    3: "The trained agent plays on the Game Boy screen to the right, with the exact "
       "observation setup it was trained with.",
}


def short_goal(preset_name: str) -> str:
    """'Mario — Campaign, recommended (~15 min)' -> 'Campaign, recommended'."""
    name = re.sub(r"^\s*Mario\s*[—-]\s*", "", preset_name)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    return name.strip() or preset_name


def describe_preset(cfg: dict) -> str:
    level = cfg.get("start_level", "default")
    mode = {"default": "campaign", "random": "random levels",
            "sequential": "sequential levels", "marathon": "marathon"}.get(level, f"level {level}")
    eta = tuning.estimate_seconds(1, int(cfg.get("timesteps", 0)), cfg)
    return (f"{mode} · {cfg.get('obs_type', 'tiles')} obs · {int(cfg.get('timesteps', 0)):,} steps "
            f"· ≈ {tuning.format_duration(eta)} · {cfg.get('n_envs', '?')} envs")


def default_preset_name(goal: str, overrides: dict) -> str:
    name = f"Wizard — {short_goal(goal)}"
    if overrides:
        name += " · " + tuning.describe_overrides(overrides)
    return name[:90]


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
        self._inputs: list[tk.Widget] = []
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

    # ----- pane 1: tune -----

    def _build_tune(self, pane: ttk.Frame) -> None:
        self._intro(pane, 0, 0)
        form = ttk.LabelFrame(pane, text="Search", padding=8)
        form.grid(row=1, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Training goal:").grid(row=0, column=0, sticky="w", pady=2)
        self.goal_var = tk.StringVar()
        self.goal_combo = ttk.Combobox(form, textvariable=self.goal_var, state="readonly")
        self.goal_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        self.goal_combo.bind("<<ComboboxSelected>>", lambda e: self._on_goal_changed())
        self.goal_note = ttk.Label(form, text="", foreground=THEME.muted)
        self.goal_note.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(form, text="Compare:").grid(row=2, column=0, sticky="w", pady=(8, 2))
        self.template_var = tk.StringVar(value=tuning.DEFAULT_TEMPLATE)
        self.template_combo = ttk.Combobox(form, textvariable=self.template_var, state="readonly",
                                           values=list(tuning.SWEEP_TEMPLATES))
        self.template_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=(8, 2))
        self.template_combo.bind("<<ComboboxSelected>>", lambda e: self._on_template_changed())
        self.template_note = ttk.Label(form, text="", foreground=THEME.muted, wraplength=520)
        self.template_note.grid(row=3, column=1, sticky="w", padx=6)

        budget = ttk.Frame(form)
        budget.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(budget, text="Steps per candidate:").pack(side="left")
        self.trial_steps_var = tk.IntVar(value=100_000)
        steps_spin = ttk.Spinbox(budget, from_=2000, to=5_000_000, increment=10_000, width=10,
                                 textvariable=self.trial_steps_var, command=self._sync_tune_tab)
        steps_spin.pack(side="left", padx=(6, 16))
        steps_spin.bind("<KeyRelease>", lambda e: self._sync_tune_tab())
        ttk.Label(budget, text="Seeds per candidate:").pack(side="left")
        self.seeds_var = tk.IntVar(value=1)
        seeds_spin = ttk.Spinbox(budget, from_=1, to=5, width=4, textvariable=self.seeds_var,
                                 command=self._sync_tune_tab)
        seeds_spin.pack(side="left", padx=(6, 16))
        self.summary_var = tk.StringVar(value="")
        ttk.Label(form, textvariable=self.summary_var, font=MONO).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._register(self.goal_combo, self.template_combo, steps_spin, seeds_spin)

        btns = ttk.Frame(pane)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.btn_search = ttk.Button(btns, text="Start search", command=self.start_search)
        self.btn_search.pack(side="left")
        self.btn_search_stop = ttk.Button(btns, text="Stop", command=self.app.stop_tuning,
                                          state="disabled")
        self.btn_search_stop.pack(side="left", padx=6)
        self.btn_skip = ttk.Button(btns, text="Skip search, use preset as is →",
                                   command=self.skip_search)
        self.btn_skip.pack(side="left", padx=(18, 0))
        self.btn_use_best = ttk.Button(btns, text="Continue with best →", command=self.use_best,
                                       state="disabled")
        self.btn_use_best.pack(side="right")

        prog = ttk.Frame(pane)
        prog.grid(row=3, column=0, sticky="ew")
        prog.columnconfigure(0, weight=1)
        ttk.Progressbar(prog, mode="determinate", maximum=100,
                        variable=self.app.tune_progress_var).grid(row=0, column=0, sticky="ew")
        ttk.Label(prog, textvariable=self.app.tune_progress_text, width=10, anchor="e").grid(
            row=0, column=1, padx=(6, 0))
        ttk.Label(pane, textvariable=self.app.tune_live_var, font=MONO).grid(
            row=4, column=0, sticky="w", pady=(4, 6))

        res = ttk.LabelFrame(pane, text="Candidates (best first)", padding=4)
        res.grid(row=5, column=0, sticky="nsew")
        pane.rowconfigure(5, weight=1)
        self.tree = make_table(res, [
            ("rank", "#", 32, "e", False), ("config", "hyperparameters", 300, "w", True),
            ("score", "best eval reward", 130, "e", False), ("time", "time", 64, "e", False),
        ], height=6)

    # ----- pane 2: preset -----

    def _build_preset(self, pane: ttk.Frame) -> None:
        self._intro(pane, 1, 0)
        box = ttk.LabelFrame(pane, text="Configuration", padding=8)
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
        ttk.Label(name_row, text="Preset name:").grid(row=0, column=0, sticky="w")
        self.preset_name_var = tk.StringVar()
        name_entry = ttk.Entry(name_row, textvariable=self.preset_name_var)
        name_entry.grid(row=0, column=1, sticky="ew", padx=6)
        self._register(name_entry)

        btns = ttk.Frame(pane)
        btns.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="← Back", command=lambda: self.goto(0)).pack(side="left")
        self.btn_save_preset = ttk.Button(btns, text="Save preset & continue →",
                                          command=self.save_and_continue)
        self.btn_save_preset.pack(side="right")
        ttk.Button(btns, text="Continue without saving →",
                   command=lambda: self.goto(2)).pack(side="right", padx=6)

    # ----- pane 3: train -----

    def _build_train(self, pane: ttk.Frame) -> None:
        self._intro(pane, 2, 0)
        form = ttk.LabelFrame(pane, text="Run", padding=8)
        form.grid(row=1, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Run name:").grid(row=0, column=0, sticky="w", pady=2)
        self.run_name_var = tk.StringVar()
        run_entry = ttk.Entry(form, textvariable=self.run_name_var, width=32)
        run_entry.grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(form, text="Total timesteps:").grid(row=1, column=0, sticky="w", pady=2)
        self.timesteps_var = tk.IntVar(value=2_000_000)
        steps_spin = ttk.Spinbox(form, from_=10_000, to=200_000_000, increment=100_000, width=14,
                                 textvariable=self.timesteps_var, command=self._update_train_eta)
        steps_spin.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        steps_spin.bind("<KeyRelease>", lambda e: self._update_train_eta())
        self.train_eta_var = tk.StringVar()
        ttk.Label(form, textvariable=self.train_eta_var, foreground=THEME.muted).grid(
            row=2, column=1, sticky="w", padx=6)
        self.preview_var = tk.BooleanVar(value=False)
        preview_cb = ttk.Checkbutton(form, text="Show live preview while training (slower)",
                                     variable=self.preview_var)
        preview_cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._register(run_entry, steps_spin, preview_cb)

        btns = ttk.Frame(pane)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.btn_train_back = ttk.Button(btns, text="← Back", command=lambda: self.goto(1))
        self.btn_train_back.pack(side="left")
        self.btn_train = ttk.Button(btns, text="Start training", command=self.start_training)
        self.btn_train.pack(side="left", padx=(18, 6))
        self.btn_train_stop = ttk.Button(btns, text="Stop", command=self.app.stop_training,
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
        ttk.Label(prog, textvariable=self.app.train_progress_text, width=24, anchor="e").grid(
            row=0, column=1, padx=(6, 0))

        stats = ttk.LabelFrame(pane, text="Live stats", padding=8)
        stats.grid(row=4, column=0, sticky="w", pady=(8, 0))
        for i, key in enumerate(("status", "total_timesteps", "ep_rew_mean", "ep_len_mean",
                                 "fps", "time_elapsed")):
            ttk.Label(stats, text=f"{key}:").grid(row=i % 3, column=(i // 3) * 2, sticky="w",
                                                  padx=(0 if i < 3 else 24, 10))
            ttk.Label(stats, textvariable=self.app.stat_vars[key], font=MONO_BOLD, width=22,
                      anchor="w").grid(row=i % 3, column=(i // 3) * 2 + 1, sticky="w")
        self.train_result_var = tk.StringVar()
        ttk.Label(pane, textvariable=self.train_result_var, wraplength=640).grid(
            row=5, column=0, sticky="w", pady=(10, 0))

    # ----- pane 4: watch -----

    def _build_watch(self, pane: ttk.Frame) -> None:
        self._intro(pane, 3, 0)
        box = ttk.LabelFrame(pane, text="Trained model", padding=8)
        box.grid(row=1, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.model_var = tk.StringVar(value="—")
        ttk.Label(box, textvariable=self.model_var, font=MONO, wraplength=600).grid(
            row=0, column=0, sticky="w")
        ttk.Label(box, foreground=THEME.muted, wraplength=600,
                  text="Episodes and speed are set on the screen panel to the right; "
                       "the live episode stats update there while it plays.").grid(
            row=1, column=0, sticky="w", pady=(4, 0))

        btns = ttk.Frame(pane)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.btn_play = ttk.Button(btns, text="▶ Play again", command=self.start_watch)
        self.btn_play.pack(side="left")
        self.btn_play_stop = ttk.Button(btns, text="■ Stop", command=self.app.stop_playing,
                                        state="disabled")
        self.btn_play_stop.pack(side="left", padx=4)
        ttk.Button(btns, text="Start over with a new agent", command=self.restart).pack(
            side="right")
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
            if not self.run_name_var.get().strip():
                self.run_name_var.set(time.strftime("wizard-%Y%m%d-%H%M"))
            self.timesteps_var.set(int(self.config.get("timesteps", 2_000_000)))
            self._update_train_eta()
            self.train_result_var.set("")
            self.btn_watch.config(state="normal" if self._best_model() else "disabled")
        elif step == 3:
            best = self._best_model()
            self.model_var.set(str(best) if best is not None else "—")
            self.watch_result_var.set("")
            if best is not None and self.phase == "idle" and not self.app.playing_active():
                self.start_watch()

    def restart(self) -> None:
        if self.app.busy() or self.app.playing_active():
            self.app.stop_playing()
        self.config, self.overrides, self.preset_name, self.run_name = {}, {}, "", ""
        self.preset_name_var.set("")
        self.run_name_var.set("")
        self.train_result_var.set("")
        self.watch_result_var.set("")
        self.btn_use_best.config(state="disabled")
        self.btn_watch.config(state="disabled")
        self.status_var.set("")
        self.goto(0)

    def set_inputs_disabled(self, disabled: bool) -> None:
        for w in self._inputs:
            try:
                if disabled:
                    w.config(state="disabled")
                else:
                    w.config(state="readonly" if isinstance(w, ttk.Combobox) else "normal")
            except tk.TclError:
                pass
        busy = self.app.busy()
        self.btn_skip.config(state="disabled" if busy else "normal")
        self.btn_save_preset.config(state="disabled" if busy else "normal")
        self.btn_train_back.config(state="disabled" if busy else "normal")
        self.btn_play.config(state="disabled" if busy else "normal")

    # ---------- step 1: tune ----------

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
        self._sync_tune_tab()

    def _on_template_changed(self) -> None:
        self.template_note.config(text=tuning.TEMPLATE_NOTES.get(self.template_var.get(), ""))
        self._sync_tune_tab()

    def _sync_tune_tab(self) -> None:
        """Mirror the wizard's choices into the Tune tab, which owns the plan."""
        app = self.app
        if app.busy():
            return
        app.tune_preset_var.set(self.goal_var.get())
        if app.tune_template_var.get() != self.template_var.get():
            app.tune_template_var.set(self.template_var.get())
            app.apply_tune_template()
        try:
            app.tune_trial_steps_var.set(int(self.trial_steps_var.get()))
            app.tune_seeds_var.set(int(self.seeds_var.get()))
        except (tk.TclError, ValueError):
            pass
        app.tune_run_prefix_var.set(WIZARD_PREFIX)
        app.tune_metric_var.set("best eval reward")
        app.tune_search_type_var.set("grid")
        app.tune_skip_done_var.set(True)
        app.tune_update_summary()

    def on_tune_plan_changed(self) -> None:
        self.template_note.config(text=tuning.TEMPLATE_NOTES.get(self.template_var.get(), ""))
        self.summary_var.set(self.app.tune_summary_var.get())

    def on_game_changed(self, runnable: bool) -> None:
        state = "normal" if runnable and not self.app.busy() else "disabled"
        if self.phase == "idle":
            self.btn_search.config(state=state)
            self.btn_skip.config(state=state)
            self.btn_train.config(state=state)
            self.btn_play.config(state=state)

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
        self.status_var.set("Searching… each candidate is a short training run "
                            "(details on the Tune tab).")

    def skip_search(self) -> None:
        ok, why = self.app.game_runnable()
        if not ok:
            messagebox.showerror("Wizard", why)
            return
        cfg = self._presets.get(self.goal_var.get())
        if cfg is None:
            messagebox.showwarning("Wizard", "Pick a training goal first.")
            return
        self.goal_name = self.goal_var.get()
        self.overrides = {}
        self.config = presets.normalize(cfg)
        self.preset_name_var.set("")
        self.status_var.set(f"Using '{self.goal_name}' unchanged.")
        self.goto(1)

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
        best = tuning.best_result(self.app.tune_results())
        if best is None:
            self.status_var.set("Search " + ("cancelled" if cancelled else "finished")
                                + " without a scored candidate. Check the Train tab log.")
            return
        self.btn_use_best.config(state="normal")
        note = " (search cancelled early)" if cancelled else ""
        self.status_var.set(f"Best so far{note}: {best.label or 'base config'} → "
                            f"{best.score_text()}. Click 'Continue with best'.")

    def _render_candidates(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        for rank, r in enumerate(self.app.tune_results(), start=1):
            self.tree.insert("", "end", iid=str(r.index),
                             values=(rank, r.label or "(base config)", r.score_text(),
                                     tuning.format_duration(r.duration)),
                             tags=("best",) if rank == 1 and r.values else (() if r.values else ("muted",)))

    def use_best(self) -> None:
        best = tuning.best_result(self.app.tune_results())
        if best is None:
            return
        self.overrides = dict(best.overrides)
        self.config = presets.normalize(best.config)
        self.preset_name_var.set("")
        self.goto(1)

    # ---------- step 2: preset ----------

    def _render_config(self) -> None:
        self.config_text.config(state="normal")
        self.config_text.delete("1.0", "end")
        if self.goal_name:
            self.config_text.insert("end", f"# based on: {self.goal_name}\n")
        if self.overrides:
            self.config_text.insert("end", "# tuned values are highlighted\n")
        self.config_text.insert("end", "\n")
        width = max((len(k) for k in self.config), default=10)
        for key in presets.PRESET_FIELDS:
            if key not in self.config:
                continue
            line = f"{key:<{width}} = {self.config[key]}\n"
            self.config_text.insert("end", line, ("tuned",) if key in self.overrides else ())
        self.config_text.config(state="disabled")

    def save_and_continue(self) -> None:
        name = self.preset_name_var.get().strip()
        if not name:
            messagebox.showwarning("Wizard", "Enter a preset name (or continue without saving).")
            return
        if not self.app.save_preset_named(name, self.config):
            return
        self.preset_name = name
        self.status_var.set(f"Preset '{name}' saved. It is now selected on the Train tab.")
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
            messagebox.showwarning("Wizard", "Enter a run name.")
            return
        if not runs.is_run_name(run_name):
            messagebox.showwarning("Wizard", "Run names may not start with '_'.")
            return
        try:
            timesteps = int(self.timesteps_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Wizard", "Total timesteps must be a whole number.")
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
            self.status_var.set("No model to play.")
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
            messagebox.showwarning("Wizard", "No trained model for this run yet.")
            return
        app = self.app
        app.refresh_models()
        if not app.select_model(best):
            messagebox.showerror("Wizard", f"Model not listed on the Play tab: {best}")
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
                f"{len(summary)} episode(s): mean reward {sum(rewards) / len(rewards):.0f}, "
                f"best {max(rewards):.0f}.")
        self.status_var.set("Done. Play again, or start over to train the next agent.")

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
