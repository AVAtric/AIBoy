"""Tkinter GUI for gameboyEnv.

Tabs, in workflow order:
  Wizard   guided Tune -> Preset -> Train -> Watch flow (see wizard.py)
  Tune     hyperparameter sweeps over short training trials
  Train    a single headless training run (subprocess) with live stats
  Play     watch any saved model inside an embedded Game Boy canvas
  Presets  browse / edit / organise presets (see presets_tab.py)

Training and tuning run `main.py train` as subprocesses so the window
stays responsive; playback runs on a background thread (player.py).
All cross-thread traffic goes through two queues drained by `_pump()`.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

import presets
import runs
import tuning
from env import RomInfo, discover_roms, level_choices, probe_rom
from player import CANVAS_H, CANVAS_W, GAME_H, GAME_W, SCALE, SPEED_CHOICES, EmbeddedPlayer
from presets_tab import PresetsTab
from widgets import MONO, MONO_BOLD, MUTED, ConfigForm, make_table, setup_styles

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GAME = "mario"
TENSORBOARD_PORT = 6006
STATUS_COLORS = {"ok": "#1e7e34", "experimental": "#b26a00", "unsupported": "#c0392b",
                 "checking": MUTED}

# SB3's verbose=1 table: "|    ep_rew_mean    | 123     |"
STAT_LINE = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([\S]+)\s+\|")
TRACKED_STATS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "fps", "time_elapsed")

LEVEL_CHOICES = level_choices()

STOP_GRACE_SECONDS = 20     # SIGINT -> trainer saves final.zip; SIGKILL after this


def interrupt(proc: subprocess.Popen | None) -> None:
    """Ask a trainer to stop the way Ctrl-C would.

    SIGINT lets `cmd_train`'s `finally` save `final.zip` and close the
    SubprocVecEnv workers. SIGTERM would kill the trainer outright and
    leave its emulator workers running at full speed.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        pass


def kill_if_alive(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


class GameBoyAIGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Game Boy AI — Super Mario Land")
        root.geometry("1060x900")
        root.minsize(980, 840)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Cross-thread channels
        self.stats_queue: queue.Queue = queue.Queue()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.player = EmbeddedPlayer(self.frame_queue, self.stats_queue)

        # Subprocess / thread state
        self.train_proc: subprocess.Popen | None = None
        self._train_stop_requested = False
        self.play_stop = threading.Event()
        self.play_thread: threading.Thread | None = None
        self.preview_stop = threading.Event()
        self.preview_thread: threading.Thread | None = None
        self.tune_stop = threading.Event()
        self.tune_thread: threading.Thread | None = None
        self.tune_proc: subprocess.Popen | None = None
        self.tb_proc: subprocess.Popen | None = None
        self._train_target_steps = 1
        self._train_rollout_size = 1
        self._train_baseline_steps: int | None = None

        self._model_paths: dict[str, Path] = {}
        self._all_presets: dict[str, dict] = {}
        self._tune_results: dict[int, tuning.ConfigResult] = {}
        self._canvases: list[tuple[tk.Canvas, int]] = []
        self._tk_img: ImageTk.PhotoImage | None = None
        self._closing = False
        self._train_run_name = ""
        self.roms: dict[str, RomInfo] = {}
        self.wizard = None
        self.presets_tab = None

        self._build_ui()
        self.rescan_roms()
        self.refresh_run_names()
        self.refresh_models()
        self.refresh_presets()
        self._pump()

    # ---------- game selection ----------

    @property
    def game(self) -> str:
        return self.game_var.get() or DEFAULT_GAME

    def rescan_roms(self) -> None:
        """List the ROM folder and probe every ROM in the background."""
        infos = discover_roms(PROJECT_DIR / "ROMs")
        self.roms = {info.name: info for info in infos}
        names = [info.name for info in infos]
        self.game_combo["values"] = names
        if self.game_var.get() not in names:
            self.game_var.set(DEFAULT_GAME if DEFAULT_GAME in names else (names[0] if names else ""))
        self._update_game_status()
        if not infos:
            self.game_status_var.set("no .gb files in ROMs/ — see README → Install")
            self.game_status_label.config(foreground=STATUS_COLORS["unsupported"])
            return

        def _probe_all(paths: list[tuple[str, Path]]) -> None:
            for name, path in paths:
                title, wrapper, error = probe_rom(path)
                self.stats_queue.put(("rom_probed", name, title, wrapper, error))

        threading.Thread(target=_probe_all, daemon=True,
                         args=([(i.name, i.path) for i in infos],)).start()

    def current_rom(self) -> RomInfo | None:
        return self.roms.get(self.game)

    def game_runnable(self) -> tuple[bool, str]:
        """(True, "") if the selected game can be trained and played here,
        else (False, reason)."""
        info = self.current_rom()
        if info is None:
            return False, f"No ROM for '{self.game}' in ROMs/."
        level, text = info.status()
        if level == "ok":
            return True, ""
        if level == "checking":
            return False, "Still checking whether this ROM can run; try again in a moment."
        return False, f"'{self.game}' cannot be used in this version: {text}"

    def _update_game_status(self) -> None:
        info = self.current_rom()
        if info is None:
            level, text = "unsupported", "no ROM selected"
        else:
            level, text = info.status()
        self.game_status_var.set(text)
        self.game_status_label.config(foreground=STATUS_COLORS[level])
        self._update_start_buttons()

    def _update_start_buttons(self) -> None:
        """Start buttons are only enabled for a runnable game (and no run active)."""
        ok, _ = self.game_runnable()
        state = "normal" if ok and not self.busy() else "disabled"
        self.btn_train_start.config(state=state)
        self.btn_tune_start.config(state=state)
        if not self.playing_active():
            self.btn_play_start.config(state=state)
        if self.wizard is not None:
            self.wizard.on_game_changed(ok)

    def _on_game_changed(self) -> None:
        self._update_game_status()
        self.refresh_run_names()
        self.refresh_models()
        self.refresh_presets()

    # ---------- state queries ----------

    def training_active(self) -> bool:
        return self.train_proc is not None and self.train_proc.poll() is None

    def tuning_active(self) -> bool:
        return self.tune_thread is not None and self.tune_thread.is_alive()

    def playing_active(self) -> bool:
        return self.play_thread is not None and self.play_thread.is_alive()

    def busy(self) -> bool:
        """A CPU-hungry subprocess (training or tuning) is running."""
        return self.training_active() or self.tuning_active()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        from wizard import WizardTab

        setup_styles(self.root)

        # Top bar: which ROM to work with, and whether it can run here.
        top = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Game:").pack(side="left")
        self.game_var = tk.StringVar(value=DEFAULT_GAME)
        self.game_combo = ttk.Combobox(top, textvariable=self.game_var, state="readonly", width=12)
        self.game_combo.pack(side="left", padx=6)
        self.game_combo.bind("<<ComboboxSelected>>", lambda e: self._on_game_changed())
        self.btn_rescan = ttk.Button(top, text="Rescan ROMs", command=self.rescan_roms)
        self.btn_rescan.pack(side="left", padx=(0, 12))
        self.game_status_var = tk.StringVar(value="")
        self.game_status_label = ttk.Label(top, textvariable=self.game_status_var, font=MONO)
        self.game_status_label.pack(side="left")

        # Status bar (bottom): what is running right now.
        self.status_bar_var = tk.StringVar(value="Idle")
        bar = ttk.Frame(self.root, padding=(10, 4))
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status_bar_var, font=MONO, foreground="#444").pack(
            side="left")
        ttk.Label(bar, text="Super Mario Land · PPO", foreground=MUTED).pack(side="right")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        self.wizard_tab = ttk.Frame(self.nb, padding=10)
        self.tune_tab = ttk.Frame(self.nb, padding=10)
        self.train_tab = ttk.Frame(self.nb, padding=10)
        self.play_tab = ttk.Frame(self.nb, padding=10)
        self.presets_frame = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.wizard_tab, text="Wizard")
        self.nb.add(self.tune_tab, text="Tune")
        self.nb.add(self.train_tab, text="Train")
        self.nb.add(self.play_tab, text="Play")
        self.nb.add(self.presets_frame, text="Presets")
        # Train defines the variables the other tabs reference; build it first.
        self._build_train(self.train_tab)
        self._build_play(self.play_tab)
        self._build_tune(self.tune_tab)
        self.wizard = WizardTab(self, self.wizard_tab)
        self.presets_tab = PresetsTab(self, self.presets_frame)
        self.nb.select(self.wizard_tab)

    def show_tab(self, tab: ttk.Frame) -> None:
        self.nb.select(tab)

    def register_canvas(self, canvas: tk.Canvas) -> None:
        """Every registered canvas shows the live emulator frame."""
        placeholder = np.full((GAME_H, GAME_W, 3), 34, dtype=np.uint8)
        if self._tk_img is None:
            self._tk_img = ImageTk.PhotoImage(
                Image.fromarray(placeholder).resize((CANVAS_W, CANVAS_H), Image.NEAREST))
        img_id = canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self._canvases.append((canvas, img_id))

    # ---------- Train tab ----------

    def _build_train(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        self.resume_var = tk.BooleanVar(value=False)
        self.preview_var = tk.BooleanVar(value=False)

        # ---- Presets bar ----
        preset_frame = ttk.LabelFrame(parent, text="Preset", padding=6)
        preset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        preset_frame.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value="")
        self.preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                          state="readonly")
        self.preset_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_preset())
        self.btn_preset_save = ttk.Button(preset_frame, text="Save as…", command=self._save_preset)
        self.btn_preset_save.grid(row=0, column=1, padx=2)
        self.btn_preset_manage = ttk.Button(preset_frame, text="Manage…",
                                            command=lambda: self.show_tab(self.presets_frame))
        self.btn_preset_manage.grid(row=0, column=2, padx=2)

        # ---- Parameters (Basic | Advanced) ----
        self.form = ConfigForm(parent)
        self.form.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        # ---- Checkboxes + buttons + progress bar ----
        controls = ttk.LabelFrame(parent, text="Run", padding=6)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        controls.columnconfigure(0, weight=1)
        run_row = ttk.Frame(controls)
        run_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(run_row, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(value="")
        # Pick an existing run (for resume) OR type a new name. Blank = 'default'.
        self.run_name_combo = ttk.Combobox(run_row, textvariable=self.run_name_var, width=24)
        self.run_name_combo.pack(side="left", padx=(6, 4))
        self.run_name_combo.bind("<KeyRelease>", lambda e: self.refresh_models())
        self.run_name_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_models())
        ttk.Label(run_row, text="(blank = 'default'; pick an existing run to resume it)",
                  foreground=MUTED).pack(side="left")
        self.resume_check = ttk.Checkbutton(
            run_row, text="Resume from newest checkpoint", variable=self.resume_var)
        self.resume_check.pack(side="left", padx=(18, 0))
        self.preview_check = ttk.Checkbutton(
            controls, text="Show live preview on the Play tab while training (slower)",
            variable=self.preview_var)
        self.preview_check.grid(row=1, column=0, sticky="w", pady=(4, 0))

        btns = ttk.Frame(controls)
        btns.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.btn_train_start = ttk.Button(btns, text="Start training", command=self.start_training)
        self.btn_train_start.pack(side="left", padx=(0, 4))
        self.btn_train_stop = ttk.Button(btns, text="Stop", command=self.stop_training,
                                         state="disabled")
        self.btn_train_stop.pack(side="left", padx=4)
        ttk.Button(btns, text="TensorBoard", command=self.open_tensorboard).pack(
            side="left", padx=(16, 4))
        ttk.Label(btns, text="(all runs, incl. tune trials)", foreground="#888").pack(side="left")

        pb_frame = ttk.Frame(controls)
        pb_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        pb_frame.columnconfigure(0, weight=1)
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.train_progress_text = tk.StringVar(value="—")
        ttk.Progressbar(pb_frame, mode="determinate", maximum=100,
                        variable=self.train_progress_var).grid(row=0, column=0, sticky="ew",
                                                               padx=(0, 6))
        ttk.Label(pb_frame, textvariable=self.train_progress_text, width=24).grid(
            row=0, column=1, sticky="e")

        # ---- Live stats + training log, side by side ----
        monitor = ttk.Frame(parent)
        monitor.grid(row=3, column=0, sticky="nsew", pady=(0, 4))
        monitor.columnconfigure(0, weight=0)
        monitor.columnconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        stats = ttk.LabelFrame(monitor, text="Live stats", padding=8)
        stats.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.stat_vars: dict[str, tk.StringVar] = {}
        # Left column: trainer stats. Right column: game state, fed by the
        # preview thread when live preview is on.
        columns = [
            [("status", "idle"), ("total_timesteps", "—"), ("ep_rew_mean", "—"),
             ("ep_len_mean", "—"), ("fps", "—"), ("time_elapsed", "—")],
            [("world", "—"), ("lives", "—"), ("coins", "—"), ("max_x", "—")],
        ]
        for col, items in enumerate(columns):
            for row, (k, v) in enumerate(items):
                self.stat_vars[k] = tk.StringVar(value=v)
                ttk.Label(stats, text=f"{k}:").grid(row=row, column=col * 2, sticky="w",
                                                    padx=(0 if col == 0 else 18, 8))
                ttk.Label(stats, textvariable=self.stat_vars[k], font=MONO_BOLD, width=12,
                          anchor="w").grid(row=row, column=col * 2 + 1, sticky="w")

        log_frame = ttk.LabelFrame(monitor, text="Training log", padding=4)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.log_text = tk.Text(log_frame, height=6, wrap="none", font=MONO,
                                background="#111", foreground="#ddd", insertbackground="#ddd")
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        yscroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=yscroll.set)

    # ---------- Play tab ----------

    def _build_play(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, width=320)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        model_frame = ttk.LabelFrame(left, text="Model", padding=8)
        model_frame.pack(fill="x", pady=(0, 6))
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                        state="readonly", width=32)
        self.model_combo.pack(fill="x", pady=(0, 4))
        self.btn_model_refresh = ttk.Button(model_frame, text="Refresh model list",
                                            command=self.refresh_models)
        self.btn_model_refresh.pack(fill="x")
        self._play_widgets: list[tk.Widget] = [self.model_combo, self.btn_model_refresh]

        opts_frame = ttk.LabelFrame(left, text="Options", padding=8)
        opts_frame.pack(fill="x", pady=(0, 6))
        opts_frame.columnconfigure(1, weight=1)
        self.play_episodes_var = tk.IntVar(value=3)
        self.play_max_steps_var = tk.IntVar(value=0)
        self.play_stochastic_var = tk.BooleanVar(value=False)
        self.play_action_repeat_var = tk.IntVar(value=4)
        self.play_frame_stack_var = tk.IntVar(value=4)
        self.play_obs_type_var = tk.StringVar(value="tiles")
        self.play_level_var = tk.StringVar(value="default")
        self.play_speed_label_var = tk.StringVar(value=SPEED_CHOICES[1][0])

        row = [0]

        def add(label, widget):
            ttk.Label(opts_frame, text=label).grid(row=row[0], column=0, sticky="w", pady=2)
            widget.grid(row=row[0], column=1, sticky="ew", pady=2, padx=6)
            row[0] += 1
            self._play_widgets.append(widget)

        add("Episodes:", ttk.Spinbox(opts_frame, from_=1, to=100,
                                     textvariable=self.play_episodes_var, width=10))
        add("Max steps (0=∞):", ttk.Spinbox(opts_frame, from_=0, to=1_000_000, increment=100,
                                            textvariable=self.play_max_steps_var, width=10))
        add("Action repeat:", ttk.Spinbox(opts_frame, from_=1, to=16,
                                          textvariable=self.play_action_repeat_var, width=10))
        add("Frame stack:", ttk.Spinbox(opts_frame, from_=1, to=8,
                                        textvariable=self.play_frame_stack_var, width=10))
        add("Obs type (match training):",
            ttk.Combobox(opts_frame, textvariable=self.play_obs_type_var,
                         values=["tiles", "pixels"], state="readonly", width=10))
        add("Start level:", ttk.Combobox(opts_frame, textvariable=self.play_level_var,
                                         values=LEVEL_CHOICES, state="readonly", width=10))
        add("Speed:", ttk.Combobox(opts_frame, textvariable=self.play_speed_label_var,
                                   values=[c[0] for c in SPEED_CHOICES],
                                   state="readonly", width=15))
        stoch_check = ttk.Checkbutton(opts_frame, text="Stochastic (sample actions)",
                                      variable=self.play_stochastic_var)
        stoch_check.grid(row=row[0], column=0, columnspan=2, sticky="w", pady=4)
        self._play_widgets.append(stoch_check)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        self.btn_play_start = ttk.Button(btns, text="Start playing", command=self.start_playing)
        self.btn_play_start.pack(fill="x", pady=2)
        self.btn_play_stop = ttk.Button(btns, text="Stop", command=self.stop_playing,
                                        state="disabled")
        self.btn_play_stop.pack(fill="x", pady=2)
        self._play_widgets.append(self.btn_play_start)

        pstats = ttk.LabelFrame(left, text="Live episode", padding=8)
        pstats.pack(fill="x", pady=6)
        pstats.columnconfigure(1, weight=1)
        self.play_stat_vars = {k: tk.StringVar(value="—") for k in
                               ("episode", "reward", "world", "x", "steps", "lives", "coins", "action")}
        for i, key in enumerate(self.play_stat_vars):
            ttk.Label(pstats, text=f"{key}:").grid(row=i, column=0, sticky="w", padx=(0, 8))
            ttk.Label(pstats, textvariable=self.play_stat_vars[key], font=MONO_BOLD).grid(
                row=i, column=1, sticky="w")

        self.play_status_var = tk.StringVar(value="idle")
        ttk.Label(left, textvariable=self.play_status_var, font=MONO, wraplength=280).pack(
            fill="x", pady=(6, 0))

        canvas_frame = ttk.LabelFrame(right, text=f"Game Boy ({SCALE}×)", padding=6)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H,
                                bg="#222", highlightthickness=0)
        self.canvas.pack()
        self.register_canvas(self.canvas)

    # ---------- Tune tab ----------

    def _build_tune(self, parent: ttk.Frame) -> None:
        """Hyperparameter search. Every trial is a `main.py train` subprocess
        scored with the same checkpoint/eval cadence. Configs come from a
        sweep template (or hand-edited JSON) expanded as a grid or a random
        sample, optionally repeated over several seeds and averaged. Results
        stream into a best-first table, are auto-saved to
        models/<game>/_tune/<prefix>.json, and can be promoted to a preset
        or straight into the Train tab.
        """
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)
        self._tune_config_widgets: list[tk.Widget] = []

        cfg_frame = ttk.LabelFrame(parent, text="Trial config", padding=6)
        cfg_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for c in (1, 3, 5):
            cfg_frame.columnconfigure(c, weight=1)

        def _field(row, col, label, widget):
            ttk.Label(cfg_frame, text=label).grid(
                row=row, column=col * 2, sticky="w", pady=(2, 0), padx=(0 if col == 0 else 12, 0))
            widget.grid(row=row, column=col * 2 + 1, sticky="ew", padx=6, pady=(2, 0))
            self._tune_config_widgets.append(widget)

        self.tune_preset_var = tk.StringVar(value="")
        self.tune_preset_combo = ttk.Combobox(cfg_frame, textvariable=self.tune_preset_var,
                                              state="readonly")
        self.tune_preset_combo.bind("<<ComboboxSelected>>", lambda e: self.tune_update_summary())
        _field(0, 0, "Base preset:", self.tune_preset_combo)
        self.tune_trial_steps_var = tk.IntVar(value=100_000)
        _field(0, 1, "Steps / trial:", ttk.Spinbox(
            cfg_frame, from_=2000, to=10_000_000, increment=10_000, width=10,
            textvariable=self.tune_trial_steps_var, command=self.tune_update_summary))
        self.tune_seeds_var = tk.IntVar(value=1)
        _field(0, 2, "Seeds / config:", ttk.Spinbox(
            cfg_frame, from_=1, to=10, width=6,
            textvariable=self.tune_seeds_var, command=self.tune_update_summary))
        self.tune_run_prefix_var = tk.StringVar(value="tune")
        _field(1, 0, "Run-name prefix:", ttk.Entry(cfg_frame, textvariable=self.tune_run_prefix_var))
        self.tune_metric_var = tk.StringVar(value="best eval reward")
        _field(1, 1, "Metric:", ttk.Combobox(
            cfg_frame, textvariable=self.tune_metric_var, state="readonly", width=16,
            values=["best eval reward", "final eval reward", "mean eval reward"]))
        self.tune_evals_var = tk.IntVar(value=5)
        _field(1, 2, "Evals / trial:", ttk.Spinbox(
            cfg_frame, from_=1, to=50, width=6, textvariable=self.tune_evals_var))
        self.tune_skip_done_var = tk.BooleanVar(value=True)
        skip_cb = ttk.Checkbutton(
            cfg_frame, variable=self.tune_skip_done_var,
            text="Reuse finished trials with the same run-name prefix and config "
                 "(resumes an interrupted sweep)")
        skip_cb.grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self._tune_config_widgets.append(skip_cb)

        sweep_frame = ttk.LabelFrame(parent, text="Sweep", padding=6)
        sweep_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        sweep_frame.columnconfigure(1, weight=1)
        ttk.Label(sweep_frame, text="Template:").grid(row=0, column=0, sticky="w")
        self.tune_template_var = tk.StringVar(value=tuning.DEFAULT_TEMPLATE)
        tmpl_combo = ttk.Combobox(sweep_frame, textvariable=self.tune_template_var,
                                  values=list(tuning.SWEEP_TEMPLATES), state="readonly")
        tmpl_combo.grid(row=0, column=1, sticky="ew", padx=6)
        tmpl_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_tune_template())
        self.btn_tune_template = ttk.Button(sweep_frame, text="Reset to template",
                                            command=self.apply_tune_template)
        self.btn_tune_template.grid(row=0, column=2)
        self._tune_config_widgets.extend([tmpl_combo, self.btn_tune_template])
        self.tune_template_note = ttk.Label(sweep_frame, text="", foreground="#888", wraplength=900)
        self.tune_template_note.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 4))

        self.tune_sweep_text = tk.Text(sweep_frame, height=6, wrap="word", font=MONO, undo=True)
        self.tune_sweep_text.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.tune_sweep_text.bind("<<Modified>>", self._on_sweep_modified)
        self._tune_config_widgets.append(self.tune_sweep_text)

        search_row = ttk.Frame(sweep_frame)
        search_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.tune_search_type_var = tk.StringVar(value="grid")
        rb_grid = ttk.Radiobutton(search_row, text="Grid (all combinations)",
                                  variable=self.tune_search_type_var, value="grid",
                                  command=self.tune_update_summary)
        rb_grid.pack(side="left", padx=(0, 12))
        rb_random = ttk.Radiobutton(search_row, text="Random sample of",
                                    variable=self.tune_search_type_var, value="random",
                                    command=self.tune_update_summary)
        rb_random.pack(side="left")
        self.tune_n_random_var = tk.IntVar(value=12)
        n_random_spin = ttk.Spinbox(search_row, from_=1, to=1000, width=5,
                                    textvariable=self.tune_n_random_var,
                                    command=self.tune_update_summary)
        n_random_spin.pack(side="left", padx=4)
        ttk.Label(search_row, text="configs").pack(side="left")
        self.tune_summary_var = tk.StringVar(value="")
        self.tune_summary_label = ttk.Label(search_row, textvariable=self.tune_summary_var, font=MONO)
        self.tune_summary_label.pack(side="right")
        self._tune_config_widgets.extend([rb_grid, rb_random, n_random_spin])

        ctrl = ttk.Frame(parent)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(0, 2))
        ctrl.columnconfigure(3, weight=1)
        self.btn_tune_start = ttk.Button(ctrl, text="Start tuning", command=self.start_tuning)
        self.btn_tune_start.grid(row=0, column=0, padx=(0, 4))
        self.btn_tune_stop = ttk.Button(ctrl, text="Stop", command=self.stop_tuning, state="disabled")
        self.btn_tune_stop.grid(row=0, column=1, padx=4)
        self.btn_tune_load = ttk.Button(ctrl, text="Load results…", command=self._tune_load_results)
        self.btn_tune_load.grid(row=0, column=2, padx=4)
        self.tune_progress_var = tk.DoubleVar(value=0.0)
        self.tune_progress_text = tk.StringVar(value="—")
        ttk.Progressbar(ctrl, mode="determinate", maximum=100,
                        variable=self.tune_progress_var).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Label(ctrl, textvariable=self.tune_progress_text, width=12, anchor="e").grid(
            row=0, column=4, sticky="e")
        self.tune_live_var = tk.StringVar(value="idle")
        ttk.Label(parent, textvariable=self.tune_live_var, font=MONO).grid(
            row=3, column=0, sticky="w", pady=(0, 4))

        res_frame = ttk.LabelFrame(parent, text="Results (best first)", padding=4)
        res_frame.grid(row=4, column=0, sticky="nsew")
        self.tune_tree = make_table(res_frame, [
            ("rank", "#", 36, "e", False), ("config", "config (overrides)", 300, "w", True),
            ("score", "score", 110, "e", False), ("ep_len", "ep len", 60, "e", False),
            ("steps", "steps", 80, "e", False), ("time", "time", 60, "e", False),
            ("runs", "runs", 200, "w", True),
        ], height=8)

        actions = ttk.Frame(parent)
        actions.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.btn_tune_save_preset = ttk.Button(actions, text="Save selected as preset",
                                               command=self._tune_save_as_preset)
        self.btn_tune_save_preset.pack(side="left", padx=(0, 4))
        self.btn_tune_to_train = ttk.Button(actions, text="Load selected into Train tab",
                                            command=self._tune_load_into_train)
        self.btn_tune_to_train.pack(side="left", padx=4)
        ttk.Label(actions, text="→ then switch to the Train tab, set a run name and Start",
                  foreground="#888").pack(side="left", padx=8)

        self.apply_tune_template()

    # ---------- Tuning: sweep editing ----------

    def apply_tune_template(self) -> None:
        name = self.tune_template_var.get()
        sweep = tuning.SWEEP_TEMPLATES.get(name)
        if sweep is None:
            return
        self.tune_sweep_text.delete("1.0", "end")
        self.tune_sweep_text.insert("1.0", json.dumps(sweep, indent=2))
        self.tune_sweep_text.edit_modified(False)
        self.tune_template_note.config(text=tuning.TEMPLATE_NOTES.get(name, ""))
        self.tune_update_summary()

    def _on_sweep_modified(self, _event=None) -> None:
        if self.tune_sweep_text.edit_modified():
            self.tune_sweep_text.edit_modified(False)
            self.tune_update_summary()

    def tune_plan(self) -> tuple[dict | None, str | None]:
        """Resolve every Tune-tab input into a plan dict, or an error string."""
        preset_name = self.tune_preset_var.get()
        base = self._all_presets.get(preset_name)
        if base is None:
            return None, "pick a base preset"
        sweep, err = tuning.validate_sweep(self.tune_sweep_text.get("1.0", "end"))
        if err:
            return None, err
        try:
            trial_steps = int(self.tune_trial_steps_var.get())
            n_seeds = max(1, int(self.tune_seeds_var.get()))
            n_random = max(1, int(self.tune_n_random_var.get()))
            evals = max(1, int(self.tune_evals_var.get()))
        except (tk.TclError, ValueError):
            return None, "steps / seeds / evals / N must be whole numbers"
        if trial_steps < 2000:
            return None, "steps per trial must be at least 2000"
        search = self.tune_search_type_var.get()
        combos = (tuning.expand_grid(sweep) if search == "grid"
                  else tuning.sample_random(sweep, n_random))
        return {
            "preset_name": preset_name, "base": dict(base), "sweep": sweep,
            "search": search, "combos": combos, "trial_steps": trial_steps,
            "n_seeds": n_seeds, "evals_per_trial": evals,
            "prefix": self.tune_run_prefix_var.get().strip() or "tune",
            "metric": self.tune_metric_var.get(),
            "skip_done": bool(self.tune_skip_done_var.get()),
        }, None

    def tune_update_summary(self, *_args) -> None:
        if not hasattr(self, "tune_summary_label"):
            return  # widget callback during construction
        plan, err = self.tune_plan()
        if err:
            self.tune_summary_var.set(f"⚠ {err}")
            self.tune_summary_label.config(foreground="#c0392b")
        else:
            n_cfg, n_seeds = len(plan["combos"]), plan["n_seeds"]
            n_trials = n_cfg * n_seeds
            eta = tuning.estimate_seconds(n_trials, plan["trial_steps"], plan["base"])
            seeds_txt = f" × {n_seeds} seeds" if n_seeds > 1 else ""
            self.tune_summary_var.set(
                f"{n_cfg} configs{seeds_txt} = {n_trials} trials · ≈ {tuning.format_duration(eta)}")
            self.tune_summary_label.config(foreground="#555")
        if self.wizard is not None:
            self.wizard.on_tune_plan_changed()

    # ---------- Tuning: execution ----------

    def start_tuning(self) -> bool:
        """Start the sweep described by the Tune tab. Returns True if it started."""
        if self.tuning_active():
            messagebox.showwarning("Tune", "Tuning already running.")
            return False
        if self.training_active():
            messagebox.showwarning("Tune", "Training is currently running. Stop training first.")
            return False
        ok, why = self.game_runnable()
        if not ok:
            messagebox.showerror("Tune", why)
            return False
        plan, err = self.tune_plan()
        if err:
            messagebox.showerror("Tune", f"Cannot start: {err}")
            return False
        n_trials = len(plan["combos"]) * plan["n_seeds"]
        if n_trials > 40 and not messagebox.askyesno(
                "Tune", f"This sweep has {n_trials} trials (≈ "
                        f"{tuning.format_duration(tuning.estimate_seconds(n_trials, plan['trial_steps'], plan['base']))}). "
                        f"Start anyway?"):
            return False
        if self.playing_active():
            self.play_stop.set()

        self._tune_results.clear()
        self.render_tune_results()
        self.tune_tree.heading("score", text=plan["metric"])
        self.tune_progress_var.set(0)
        self.tune_progress_text.set(f"0/{n_trials}")
        self.tune_live_var.set("starting…")
        self.btn_tune_start.config(state="disabled")
        self.btn_tune_stop.config(state="normal")
        self.btn_train_start.config(state="disabled")
        self.set_inputs_disabled(True)
        self.tune_stop.clear()
        self.tune_thread = threading.Thread(target=self._tuning_loop, args=(plan,), daemon=True)
        self.tune_thread.start()
        return True

    def stop_tuning(self) -> None:
        self.tune_stop.set()
        proc = self.tune_proc
        interrupt(proc)
        self.root.after(STOP_GRACE_SECONDS * 1000, lambda: kill_if_alive(proc))

    def _run_trial(self, cmd: list[str], label: str, trial_steps: int,
                   done_trials: int, total: int) -> int | None:
        """Run one training subprocess, streaming its SB3 stats into the
        live-status line and the progress bar. Returns the exit code."""
        try:
            self.tune_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJECT_DIR),
            )
        except OSError as e:
            self.stats_queue.put(("tune_status", f"{label}: could not start: {e}"))
            return None
        assert self.tune_proc.stdout is not None
        stats: dict[str, str] = {}
        for line in self.tune_proc.stdout:
            m = STAT_LINE.search(line)
            if m is None:
                if line.startswith(("Traceback", "[train] warning")) or "Error" in line:
                    self.stats_queue.put(("log", f"[tune] {line}"))
                continue
            stats[m.group(1)] = m.group(2)
            if m.group(1) != "total_timesteps":
                continue
            try:
                steps = int(m.group(2))
            except ValueError:
                continue
            frac = (done_trials + min(1.0, steps / trial_steps)) / total
            self.stats_queue.put(("tune_progress", frac, f"{done_trials}/{total}"))
            self.stats_queue.put(("tune_live",
                f"{label} · {steps:,}/{trial_steps:,} steps · "
                f"ep_rew_mean {stats.get('ep_rew_mean', '—')} · "
                f"ep_len_mean {stats.get('ep_len_mean', '—')} · {stats.get('fps', '—')} fps"))
        rc = self.tune_proc.wait()
        self.tune_proc = None
        return rc

    def _tuning_loop(self, plan: dict) -> None:
        base, combos = plan["base"], plan["combos"]
        trial_steps, n_seeds = plan["trial_steps"], plan["n_seeds"]
        prefix, metric = plan["prefix"], plan["metric"]
        game = base.get("game", self.game)
        eval_freq = max(1000, trial_steps // plan["evals_per_trial"])
        total = len(combos) * n_seeds
        done_trials = 0
        results: list[tuning.ConfigResult] = []
        out_path = tuning.results_file(PROJECT_DIR, game, prefix)
        meta = {"game": game, "base_preset": plan["preset_name"], "metric": metric,
                "trial_steps": trial_steps, "seeds": n_seeds, "search": plan["search"],
                "sweep": plan["sweep"]}
        cancelled = False
        for i, overrides in enumerate(combos, start=1):
            cfg = tuning.full_config(base, overrides)
            res = tuning.ConfigResult(index=i, overrides=overrides, config=cfg, metric=metric)
            t0 = time.time()
            for s in range(n_seeds):
                if self.tune_stop.is_set():
                    cancelled = True
                    break
                run_name = tuning.trial_run_name(prefix, i, s, n_seeds)
                run_dir = runs.run_paths(game, run_name)["base"]
                seed_cfg = {**cfg, "seed": int(cfg.get("seed", 0)) + s}
                label = f"config {i}/{len(combos)}" + (f" seed {s + 1}/{n_seeds}" if n_seeds > 1 else "")
                self.stats_queue.put(("tune_progress", done_trials / total, f"{done_trials}/{total}"))
                if plan["skip_done"] and tuning.trial_is_complete(run_dir, trial_steps, seed_cfg):
                    res.reused += 1
                    self.stats_queue.put(("tune_status", f"{label}: reusing finished run {run_name}"))
                else:
                    cmd = runs.build_train_cmd(seed_cfg, run_name, timesteps=trial_steps,
                                               checkpoint_freq=trial_steps, eval_freq=eval_freq)
                    try:
                        tuning.write_trial_manifest(run_dir, seed_cfg, trial_steps)
                    except OSError as e:
                        self.stats_queue.put(("tune_status", f"{label}: could not write manifest: {e}"))
                    self.stats_queue.put(("tune_status",
                                          f"{label}: {res.label or 'base config'}  →  {run_name}"))
                    rc = self._run_trial(cmd, label, trial_steps, done_trials, total)
                    if self.tune_stop.is_set():
                        cancelled = True
                        break
                    if rc not in (0, None):
                        self.stats_queue.put(("tune_status", f"{label}: trainer exited with rc={rc}"))
                m = tuning.read_trial_metrics(run_dir / "logs" / "evaluations.npz")
                if m is not None:
                    res.values.append(m.by_name(metric))
                    res.ep_lens.append(m.ep_len)
                    res.timesteps = m.timesteps
                else:
                    self.stats_queue.put(("tune_status", f"{label}: no evaluations.npz in {run_name}"))
                res.runs.append(run_name)
                done_trials += 1
            res.duration = time.time() - t0
            if res.runs:
                results.append(res)
                self.stats_queue.put(("tune_result", res))
                try:
                    tuning.save_results(out_path, meta, results)
                except OSError as e:
                    self.stats_queue.put(("tune_status", f"could not save results: {e}"))
            if cancelled:
                break
        self.stats_queue.put(("tune_progress", done_trials / max(1, total), f"{done_trials}/{total}"))
        rel = out_path.relative_to(PROJECT_DIR)
        text = "cancelled" if cancelled else f"done — results in {rel}"
        self.stats_queue.put(("tune_done", text, cancelled))

    # ---------- Tuning: results ----------

    def tune_results(self) -> list[tuning.ConfigResult]:
        return tuning.rank_results(self._tune_results.values())

    def render_tune_results(self) -> None:
        for row in self.tune_tree.get_children():
            self.tune_tree.delete(row)
        for rank, r in enumerate(self.tune_results(), start=1):
            run_txt = ", ".join(r.runs) + (f"  ({r.reused} reused)" if r.reused else "")
            self.tune_tree.insert(
                "", "end", iid=str(r.index),
                values=(rank, r.label or "(base config)", r.score_text(),
                        f"{r.ep_len:.0f}" if r.values else "—",
                        f"{r.timesteps:,}", tuning.format_duration(r.duration), run_txt),
                tags=("best",) if rank == 1 and r.values else (() if r.values else ("muted",)),
            )

    def _selected_tune_result(self) -> tuning.ConfigResult | None:
        sel = self.tune_tree.selection()
        if not sel:
            messagebox.showwarning("Tune", "Select a row in the results table first.")
            return None
        return self._tune_results.get(int(sel[0]))

    def _tune_load_results(self) -> None:
        """Reload a saved sweep (models/<game>/_tune/<prefix>.json)."""
        initial = PROJECT_DIR / "models" / self.game / "_tune"
        path = filedialog.askopenfilename(
            title="Load tuning results", parent=self.root,
            initialdir=str(initial if initial.exists() else PROJECT_DIR),
            filetypes=[("Tuning results", "*.json")])
        if not path:
            return
        try:
            meta, results = tuning.load_results(Path(path))
        except (OSError, ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Tune", f"Could not read {path}:\n{e}")
            return
        self._tune_results = {r.index: r for r in results}
        if meta.get("metric"):
            self.tune_metric_var.set(meta["metric"])
            self.tune_tree.heading("score", text=meta["metric"])
        if meta.get("sweep"):
            self.tune_sweep_text.delete("1.0", "end")
            self.tune_sweep_text.insert("1.0", json.dumps(meta["sweep"], indent=2))
        self.render_tune_results()
        self.tune_live_var.set(f"loaded {len(results)} results from {Path(path).name}")

    def _tune_save_as_preset(self) -> None:
        res = self._selected_tune_result()
        if res is None:
            return
        name = simpledialog.askstring(
            "Save tune result as preset",
            f"Name for this preset ({res.label or 'base config'}):", parent=self.root)
        if not self.save_preset_named((name or "").strip(), res.config):
            return
        messagebox.showinfo("Saved", f"Preset '{name}' saved and selected on the Train tab.")

    def _tune_load_into_train(self) -> None:
        res = self._selected_tune_result()
        if res is None:
            return
        self.load_config_into_train(res.config)
        self.preset_var.set("")  # custom config
        messagebox.showinfo("Loaded", "Config copied into the Train tab. Switch tabs, "
                                      "set a run name and click Start.")

    # ---------- Presets ----------

    def refresh_presets(self) -> None:
        self._all_presets = presets.load_all()
        names = presets.sorted_names(self._all_presets, self.game)
        self.tune_preset_combo["values"] = names
        if self.tune_preset_var.get() not in names and names:
            self.tune_preset_var.set(names[0])
        prev = self.preset_var.get()
        self.preset_combo["values"] = names
        if prev in names:
            self.preset_var.set(prev)
        elif names:
            self.preset_var.set(names[0])
            self.apply_preset()
        if self.wizard is not None:
            self.wizard.on_presets_changed(names)
        if self.presets_tab is not None:
            self.presets_tab.refresh()
        self.tune_update_summary()

    def presets_by_name(self) -> dict[str, dict]:
        return dict(self._all_presets)

    def load_config_into_train(self, cfg: dict) -> None:
        """Populate the Train tab (and the Play tab's obs / level / input
        shape) from a preset-style dict. Optional fields missing from `cfg`
        fall back to PRESET_DEFAULTS so applying a preset is deterministic."""
        self.form.set_config(cfg)
        self.sync_play_options(cfg)
        self.refresh_models()

    def sync_play_options(self, cfg: dict) -> None:
        """Playback must use the observation setup the model was trained with."""
        if "obs_type" in cfg:
            self.play_obs_type_var.set(cfg["obs_type"])
        if "start_level" in cfg:
            self.play_level_var.set(cfg["start_level"])
        if "action_repeat" in cfg:
            self.play_action_repeat_var.set(int(cfg["action_repeat"]))
        if "frame_stack" in cfg:
            self.play_frame_stack_var.set(int(cfg["frame_stack"]))

    def apply_preset(self) -> None:
        cfg = self._all_presets.get(self.preset_var.get())
        if cfg is not None:
            self.load_config_into_train(cfg)

    def current_config(self) -> dict:
        """Train-tab values as a preset dict. Raises ValueError naming the
        field if a value is not a number."""
        return {"game": self.game, **self.form.get_config()}

    def save_preset_named(self, name: str, cfg: dict, *, confirm_overwrite: bool = True) -> bool:
        """Persist `cfg` as user preset `name` and select it on the Train tab."""
        if not name:
            return False
        if confirm_overwrite and presets.is_user(name) and not messagebox.askyesno(
                "Save preset", f"A preset named '{name}' already exists. Overwrite it?"):
            return False
        try:
            presets.upsert(name, cfg)
        except (ValueError, OSError) as e:
            messagebox.showerror("Save preset", str(e))
            return False
        self.refresh_presets()
        self.preset_var.set(name)
        self.apply_preset()
        return True

    def _save_preset(self) -> None:
        try:
            cfg = self.current_config()
        except ValueError as e:
            messagebox.showerror("Save preset", f"A training field has an invalid value:\n{e}")
            return
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        self.save_preset_named((name or "").strip(), cfg)

    # ---------- Runs & models ----------

    def refresh_run_names(self) -> None:
        self.run_name_combo["values"] = runs.list_runs(self.game)

    def refresh_models(self) -> None:
        self._model_paths = dict(runs.list_models(self.game))
        labels = list(self._model_paths)
        prev = self.model_var.get()
        self.model_combo["values"] = labels
        if prev in labels:
            self.model_var.set(prev)
        else:
            self.model_var.set(labels[0] if labels else "")

    def select_model(self, path: Path) -> bool:
        """Select the Play-tab model whose file is `path`. False if unknown."""
        for label, p in self._model_paths.items():
            if p == path:
                self.model_var.set(label)
                return True
        return False

    # ---------- Training ----------

    def start_training(self) -> bool:
        """Launch `main.py train` with the Train tab's config. True if started."""
        if self.training_active():
            messagebox.showwarning("Training", "Training is already running.")
            return False
        if self.tuning_active():
            messagebox.showwarning("Training", "Tuning is currently running. Stop it first.")
            return False
        ok, why = self.game_runnable()
        if not ok:
            messagebox.showerror("Training", why)
            return False
        try:
            cfg = self.current_config()
        except ValueError as e:
            messagebox.showerror("Training", f"A training field has an invalid value:\n{e}")
            return False
        # Playback would compete for CPU and for the shared canvas.
        if self.playing_active():
            self.play_stop.set()
        run_name = self.run_name_var.get().strip() or "default"
        cmd = runs.build_train_cmd(cfg, run_name, resume=bool(self.resume_var.get()))

        self.append_log(f"$ {' '.join(cmd)}\n")
        try:
            self.train_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJECT_DIR),
            )
        except OSError as e:
            messagebox.showerror("Training", f"Could not start the trainer:\n{e}")
            return False
        self._train_stop_requested = False
        self._train_run_name = run_name
        self.btn_train_start.config(state="disabled")
        self.btn_train_stop.config(state="normal")
        self.btn_tune_start.config(state="disabled")
        self.set_inputs_disabled(True)
        self.stat_vars["status"].set("running")
        for k in TRACKED_STATS[1:] + ("world", "lives", "coins", "max_x"):
            self.stat_vars[k].set("—")
        self._train_target_steps = max(1, cfg["timesteps"])
        # Baseline = the model's prior step count (0 unless resuming). The
        # first reported total_timesteps is baseline + one rollout.
        self._train_rollout_size = max(1, cfg["n_steps"] * cfg["n_envs"])
        self._train_baseline_steps = None
        self.train_progress_var.set(0)
        self.train_progress_text.set(f"0 / {self._train_target_steps:,}")
        threading.Thread(target=self._read_train_stdout, daemon=True).start()

        if self.preview_var.get():
            self._start_preview(self.game, run_name, cfg)
        return True

    def _read_train_stdout(self) -> None:
        proc = self.train_proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            self.stats_queue.put(("log", line))
            m = STAT_LINE.search(line)
            if m and m.group(1) in TRACKED_STATS:
                self.stats_queue.put(("stat", m.group(1), m.group(2)))
        rc = proc.wait()
        self.stats_queue.put(("train_done", rc))

    def stop_training(self) -> None:
        """Interrupt the trainer (it saves final.zip) without blocking the UI."""
        proc = self.train_proc
        if proc is None or proc.poll() is not None:
            return
        self._train_stop_requested = True
        self.append_log("[gui] stopping training (saving final.zip)…\n")
        self.stat_vars["status"].set("stopping…")
        self.preview_stop.set()
        interrupt(proc)
        self.root.after(STOP_GRACE_SECONDS * 1000, lambda: kill_if_alive(proc))

    def _start_preview(self, game: str, run_name: str, cfg: dict) -> None:
        """Watch the newest best_model.zip play alongside training."""
        self.preview_stop.clear()
        raw_level = cfg["start_level"]
        self.sync_play_options(cfg)
        self.preview_thread = self.player.preview(
            model_path=runs.run_paths(game, run_name)["logs"] / "best_model.zip",
            game=game, obs_type=cfg["obs_type"], action_repeat=int(cfg["action_repeat"]),
            frame_stack=int(cfg["frame_stack"]),
            start_level=None if raw_level == "default" else raw_level,
            stop=self.preview_stop, training_active=self.training_active,
        )

    # ---------- Playing ----------

    def start_playing(self) -> bool:
        """Play the selected model in the embedded canvas. True if started."""
        if self.playing_active():
            messagebox.showwarning("Play", "Playback already running.")
            return False
        if self.busy():
            messagebox.showwarning(
                "Play", "Training or tuning is running. Stop it first, or wait for it to finish.")
            return False
        ok, why = self.game_runnable()
        if not ok:
            messagebox.showerror("Play", why)
            return False
        label = self.model_var.get()
        if not label:
            self.refresh_models()
            label = self.model_var.get()
        if not label:
            messagebox.showerror(
                "Play",
                f"No trained model found for '{self.game}'. Train first, or place a .zip at\n"
                f"models/{self.game}/<run>/logs/best_model.zip or "
                f"models/{self.game}/<run>/checkpoints/final.zip")
            return False
        model_path = self._model_paths.get(label)
        if model_path is None or not model_path.exists():
            messagebox.showerror("Play", f"Model file missing: {model_path}")
            return False
        try:
            episodes = max(1, int(self.play_episodes_var.get()))
            max_steps = max(0, int(self.play_max_steps_var.get()))
            action_repeat = int(self.play_action_repeat_var.get())
            frame_stack = int(self.play_frame_stack_var.get())
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Play", f"A play option has an invalid value:\n{e}")
            return False
        raw_level = self.play_level_var.get()

        self.play_stop.clear()
        self.btn_play_start.config(state="disabled")
        self.btn_play_stop.config(state="normal")
        self.play_status_var.set(f"loading {model_path.name}…")
        for v in self.play_stat_vars.values():
            v.set("—")
        self.play_thread = self.player.play(
            model_path=model_path, game=self.game, obs_type=self.play_obs_type_var.get(),
            action_repeat=action_repeat, frame_stack=frame_stack,
            start_level=None if raw_level == "default" else raw_level,
            episodes=episodes, max_steps=max_steps,
            deterministic=not self.play_stochastic_var.get(),
            speed_mult=dict(SPEED_CHOICES)[self.play_speed_label_var.get()],
            stop=self.play_stop,
        )
        return True

    def stop_playing(self) -> None:
        self.play_stop.set()

    # ---------- Event loop ----------

    def _pump(self) -> None:
        if self._closing:
            return
        try:
            while True:
                self._handle_event(self.stats_queue.get_nowait())
        except queue.Empty:
            pass

        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            img = Image.fromarray(latest).resize((CANVAS_W, CANVAS_H), Image.NEAREST)
            self._tk_img = ImageTk.PhotoImage(img)
            for canvas, img_id in self._canvases:
                canvas.itemconfig(img_id, image=self._tk_img)

        self._update_status_bar()
        try:
            self.root.after(33, self._pump)
        except tk.TclError:
            pass

    def _update_status_bar(self) -> None:
        if self.training_active():
            text = (f"Training '{self._train_run_name}' · {self.train_progress_text.get()} steps · "
                    f"{self.stat_vars['fps'].get()} fps · ep_rew_mean {self.stat_vars['ep_rew_mean'].get()}")
        elif self.tuning_active():
            text = f"Tuning · {self.tune_progress_text.get()} trials · {self.tune_live_var.get()}"
        elif self.playing_active():
            text = f"Playing · {self.play_status_var.get()}"
        else:
            text = "Idle"
        if self.status_bar_var.get() != text:
            self.status_bar_var.set(text)

    def _handle_event(self, item: tuple) -> None:
        kind = item[0]
        if kind == "rom_probed":
            _, name, title, wrapper, error = item
            info = self.roms.get(name)
            if info is not None:
                info.probed, info.title, info.has_wrapper, info.error = True, title, wrapper, error
                if name == self.game:
                    self._update_game_status()
        elif kind == "log":
            self.append_log(item[1])
        elif kind == "stat":
            _, key, val = item
            if key in self.stat_vars:
                self.stat_vars[key].set(val)
            if key == "total_timesteps":
                self._update_train_progress(val)
        elif kind == "train_done":
            self._on_train_done(item[1])
        elif kind == "play_status":
            self.play_status_var.set(item[1])
        elif kind == "play_stat":
            _, key, val = item
            if key in self.play_stat_vars:
                self.play_stat_vars[key].set(val)
        elif kind == "play_done":
            self.play_status_var.set("done")
            self._play_finished()
            if self.wizard is not None:
                self.wizard.on_play_done(item[1])
        elif kind == "play_error":
            self.play_status_var.set("error")
            self._play_finished()
            if self.wizard is not None:
                self.wizard.on_play_error(item[1])
            messagebox.showerror("Play error", item[1])
        elif kind == "tune_live":
            self.tune_live_var.set(item[1])
        elif kind == "tune_progress":
            _, frac, label = item
            self.tune_progress_var.set(100.0 * frac)
            self.tune_progress_text.set(label)
        elif kind == "tune_result":
            r = item[1]
            self._tune_results[r.index] = r
            self.render_tune_results()
            self.append_log(f"[tune] config {r.index} ({r.label or 'base'}) → "
                            f"{r.score_text()}  {tuning.format_duration(r.duration)}\n")
            if self.wizard is not None:
                self.wizard.on_tune_result()
        elif kind == "tune_status":
            self.append_log(f"[tune] {item[1]}\n")
        elif kind == "tune_done":
            _, text, cancelled = item
            self.btn_tune_stop.config(state="disabled")
            self.set_inputs_disabled(False)
            self._update_start_buttons()
            self.tune_proc = None
            self.tune_live_var.set(text)
            self.append_log(f"[tune] {text}\n")
            self.refresh_models()
            self.refresh_run_names()
            if self.wizard is not None:
                self.wizard.on_tune_done(text, cancelled)

    def _update_train_progress(self, val: str) -> None:
        try:
            done_abs = int(val)
        except ValueError:
            return
        if self._train_baseline_steps is None:
            # Fresh run: first report = one rollout, baseline 0.
            # Resume: first report = previous total + one rollout.
            self._train_baseline_steps = max(0, done_abs - self._train_rollout_size)
        new_steps = done_abs - self._train_baseline_steps
        pct = min(100.0, 100.0 * new_steps / max(1, self._train_target_steps))
        self.train_progress_var.set(pct)
        self.train_progress_text.set(f"{new_steps:,} / {self._train_target_steps:,}")

    def _on_train_done(self, rc: int) -> None:
        if rc == 0:
            status = "finished"
            self.train_progress_var.set(100)
        elif self._train_stop_requested:
            status = "stopped (final.zip saved)"
            self.train_progress_text.set("stopped")
        else:
            status = f"failed (rc={rc}) — see log"
            self.train_progress_text.set("failed")
        self.stat_vars["status"].set(status)
        self.btn_train_stop.config(state="disabled")
        self.set_inputs_disabled(False)
        self.train_proc = None
        self._update_start_buttons()
        self.preview_stop.set()
        self.refresh_models()       # new checkpoints may have arrived
        self.refresh_run_names()    # a new run dir may have appeared
        if self.wizard is not None:
            self.wizard.on_train_done(rc, self._train_stop_requested)

    def _play_finished(self) -> None:
        self.btn_play_stop.config(state="disabled")
        self._update_start_buttons()

    def set_inputs_disabled(self, disabled: bool) -> None:
        """Lock every input on every tab while training or tuning runs, so a
        config cannot change mid-run and a second CPU-hungry session cannot
        be started. Start / Stop buttons are managed by their own callers."""
        def _state_for(w):
            if disabled:
                return "disabled"
            return "readonly" if isinstance(w, ttk.Combobox) else "normal"

        widgets: list[tk.Widget] = [
            self.game_combo, self.btn_rescan,
            self.preset_combo, self.btn_preset_save, self.run_name_combo,
            self.resume_check, self.preview_check,
            *self._play_widgets,
            *self._tune_config_widgets,
            self.btn_tune_save_preset, self.btn_tune_to_train, self.btn_tune_load,
        ]
        for w in widgets:
            try:
                w.config(state=_state_for(w))
            except tk.TclError:
                pass
        self.form.set_enabled(not disabled)
        if self.wizard is not None:
            self.wizard.set_inputs_disabled(disabled)
        if self.presets_tab is not None:
            self.presets_tab.set_inputs_disabled(disabled)

    def open_tensorboard(self) -> None:
        """Serve models/<game>/ (every run, incl. tune trials) and open a browser tab."""
        url = f"http://localhost:{TENSORBOARD_PORT}"
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            webbrowser.open(url)
            return
        logdir = PROJECT_DIR / "models" / self.game
        exe = shutil.which("tensorboard")
        cmd = [exe] if exe else [sys.executable, "-m", "tensorboard.main"]
        cmd += ["--logdir", str(logdir), "--port", str(TENSORBOARD_PORT)]
        try:
            self.tb_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, cwd=str(PROJECT_DIR))
        except OSError as e:
            messagebox.showerror("TensorBoard", f"Could not start TensorBoard:\n{e}\n\n"
                                                f"Install it with: pip install tensorboard")
            return
        self.append_log(f"[gui] tensorboard --logdir {logdir} → {url}\n")
        self.root.after(2500, lambda: webbrowser.open(url))

    def append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 2000:
            self.log_text.delete("1.0", "500.0")

    def _on_close(self) -> None:
        self._closing = True
        self.play_stop.set()
        self.preview_stop.set()
        self.tune_stop.set()
        for proc in (self.tune_proc, self.train_proc):
            if proc is not None and proc.poll() is None:
                interrupt(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    kill_if_alive(proc)
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            self.tb_proc.terminate()
        self.root.destroy()


def run() -> None:
    os.chdir(PROJECT_DIR)
    root = tk.Tk()
    GameBoyAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
