"""Tkinter GUI for gameboyEnv: train (subprocess) or play (embedded canvas).

Play runs the emulator with rendering enabled but no SDL2 window, grabs
frames via `screen_ndarray()` and paints them onto an embedded Canvas.
Pacing is done manually via `time.sleep()` because PyBoy's built-in
`set_emulation_speed` only paces when using the SDL2 window.
"""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog, ttk, messagebox

import numpy as np
from PIL import Image, ImageTk
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

import presets
from env import GAMES, MarioEnv, SML_ALL_LEVELS, ensure_level_states

SCALE = 3
GAME_W, GAME_H = 160, 144
CANVAS_W, CANVAS_H = GAME_W * SCALE, GAME_H * SCALE

STAT_LINE = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([\S]+)\s+\|")
TRACKED_STATS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "fps", "time_elapsed")

SPEED_CHOICES = [("0.5×", 0.5), ("1× (real time)", 1.0), ("2×", 2.0),
                 ("4×", 4.0), ("Unlimited", 0.0)]

LEVEL_CHOICES = (["default", "random", "sequential", "marathon"]
                 + [f"{w}-{l}" for (w, l) in SML_ALL_LEVELS])


def build_train_cmd(cfg: dict, run_name: str, *, resume: bool = False,
                    timesteps: int | None = None, checkpoint_freq: int | None = None,
                    eval_freq: int | None = None) -> list[str]:
    """`main.py train` argv for a preset-style config dict (see
    presets.PRESET_FIELDS). Shared by the Train tab and the Tune tab so
    both launch training with exactly the same flag set."""
    cmd = [
        sys.executable, "-u", "main.py", "train",
        "--game", str(cfg.get("game", "mario")),
        "--n-envs", str(cfg["n_envs"]),
        "--timesteps", str(timesteps if timesteps is not None else cfg["timesteps"]),
        "--ent-coef", str(cfg["ent_coef"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--n-steps", str(cfg["n_steps"]),
        "--batch-size", str(cfg["batch_size"]),
        "--obs-type", str(cfg.get("obs_type", "tiles")),
        "--start-level", str(cfg.get("start_level", "default")),
        "--device", str(cfg.get("device", "cpu")),
        "--action-repeat", str(cfg.get("action_repeat", 4)),
        "--frame-stack", str(cfg.get("frame_stack", 4)),
        "--n-epochs", str(cfg.get("n_epochs", 4)),
        "--seed", str(cfg.get("seed", 0)),
        "--checkpoint-freq", str(checkpoint_freq if checkpoint_freq is not None
                                 else cfg.get("checkpoint_freq", 25_000)),
        "--eval-freq", str(eval_freq if eval_freq is not None
                           else cfg.get("eval_freq", 10_000)),
        "--n-eval-episodes", str(cfg.get("n_eval_episodes", 3)),
        "--run-name", run_name,
    ]
    if resume:
        cmd.append("--resume")
    return cmd


class GameBoyAIGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Game Boy AI — Train & Play")
        root.geometry("1040x880")
        root.minsize(920, 720)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.project_dir = Path(__file__).resolve().parent

        # Cross-thread channels
        self.stats_queue: queue.Queue = queue.Queue()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Subprocess / thread state
        self.train_proc: subprocess.Popen | None = None
        self.play_stop = threading.Event()
        self.play_thread: threading.Thread | None = None
        self.preview_stop = threading.Event()
        self.preview_thread: threading.Thread | None = None
        self.tune_stop = threading.Event()
        self.tune_thread: threading.Thread | None = None
        self.tune_proc: subprocess.Popen | None = None
        self._train_target_steps = 1
        self._train_baseline_steps: int | None = None

        self._model_paths: dict[str, Path] = {}
        self._closing = False

        self._build_ui()
        self._refresh_run_names()
        self._refresh_models()
        self._refresh_presets()
        self._pump()

    def _on_game_changed(self) -> None:
        self._refresh_run_names()
        self._refresh_models()

    def _refresh_run_names(self) -> None:
        """Populate the run-name combobox with existing runs for the current game.
        Ignores names starting with '_' (reserved for internal caches like
        `_level_states/` which holds per-level save-state files).
        """
        game = self.game_var.get()
        game_dir = self.project_dir / "models" / game
        if game_dir.exists():
            runs = sorted(
                d.name for d in game_dir.glob("*")
                if d.is_dir() and not d.name.startswith("_")
            )
        else:
            runs = []
        self.run_name_combo["values"] = runs

    # ---------- UI ----------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Game:").pack(side="left")
        self.game_var = tk.StringVar(value="mario")
        game_combo = ttk.Combobox(top, textvariable=self.game_var, values=sorted(GAMES),
                                   state="readonly", width=10)
        game_combo.pack(side="left", padx=6)
        game_combo.bind("<<ComboboxSelected>>", lambda e: self._on_game_changed())
        ttk.Label(top, text="Run name (blank = 'default'):").pack(side="left", padx=(20, 4))
        self.run_name_var = tk.StringVar(value="")
        # Combobox lets the user pick an existing run (for resume) OR type a new name
        self.run_name_combo = ttk.Combobox(top, textvariable=self.run_name_var, width=16)
        self.run_name_combo.pack(side="left")
        self.run_name_combo.bind("<KeyRelease>", lambda e: self._refresh_models())
        self.run_name_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_models())

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        # Workflow order: 1) Tune → find good hyperparams,
        #                 2) Train → full training run with those,
        #                 3) Play → watch the trained agent.
        tune_tab = ttk.Frame(nb, padding=10)
        train_tab = ttk.Frame(nb, padding=10)
        play_tab = ttk.Frame(nb, padding=10)
        nb.add(tune_tab, text="Tune")
        nb.add(train_tab, text="Train")
        nb.add(play_tab, text="Play")
        # Build Train first (defines vars the other tabs may reference)
        self._build_train(train_tab)
        self._build_play(play_tab)
        self._build_tune(tune_tab)
        # Train is the primary workflow — open focused on it, even though
        # Tune sits to its left in the tab strip (Tune → Train → Play).
        nb.select(train_tab)

    def _build_train(self, parent: ttk.Frame) -> None:
        # 2-column params layout so the tab fits on a laptop screen.
        parent.columnconfigure(0, weight=1)

        self.tsteps_var = tk.IntVar(value=500_000)
        self.n_envs_var = tk.IntVar(value=10)
        self.ent_coef_var = tk.DoubleVar(value=0.01)
        self.lr_var = tk.DoubleVar(value=2.5e-4)
        self.nsteps_var = tk.IntVar(value=256)
        self.batch_var = tk.IntVar(value=128)
        self.device_var = tk.StringVar(value="auto")
        self.resume_var = tk.BooleanVar(value=False)
        self.action_repeat_var = tk.IntVar(value=4)
        self.frame_stack_var = tk.IntVar(value=4)
        self.n_epochs_var = tk.IntVar(value=4)
        self.seed_var = tk.IntVar(value=0)
        self.ckpt_freq_var = tk.IntVar(value=25_000)
        self.eval_freq_var = tk.IntVar(value=10_000)
        self.n_eval_var = tk.IntVar(value=3)
        self.obs_type_var = tk.StringVar(value="tiles")
        self.start_level_var = tk.StringVar(value="default")
        self.preview_var = tk.BooleanVar(value=False)

        # ---- Presets bar ----
        preset_frame = ttk.LabelFrame(parent, text="Preset", padding=6)
        preset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        preset_frame.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value="")
        self.preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                          state="readonly")
        self.preset_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        self.btn_preset_save = ttk.Button(preset_frame, text="Save as…",
                                           command=self._save_preset)
        self.btn_preset_save.grid(row=0, column=1, padx=2)
        self.btn_preset_delete = ttk.Button(preset_frame, text="Delete",
                                             command=self._delete_preset)
        self.btn_preset_delete.grid(row=0, column=2, padx=2)
        self.btn_preset_reload = ttk.Button(preset_frame, text="Reload",
                                             command=self._refresh_presets)
        self.btn_preset_reload.grid(row=0, column=3, padx=2)

        # ---- Basic + Advanced params, side by side ----
        params_frame = ttk.Frame(parent)
        params_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        params_frame.columnconfigure(0, weight=1)
        params_frame.columnconfigure(1, weight=1)

        basic = ttk.LabelFrame(params_frame, text="Basic", padding=8)
        basic.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        basic.columnconfigure(1, weight=1)
        advanced = ttk.LabelFrame(params_frame, text="Advanced", padding=8)
        advanced.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        advanced.columnconfigure(1, weight=1)

        # Widgets that must be disabled while training is running
        self._train_config_widgets: list[tk.Widget] = []

        def _add(parent_frame, row_ref, label, widget):
            ttk.Label(parent_frame, text=label).grid(row=row_ref[0], column=0,
                                                     sticky="w", pady=1)
            widget.grid(row=row_ref[0], column=1, sticky="ew", pady=1, padx=(6, 0))
            row_ref[0] += 1
            self._train_config_widgets.append(widget)

        r = [0]
        _add(basic, r, "Total timesteps:", ttk.Spinbox(basic, from_=1000, to=100_000_000,
              increment=10_000, textvariable=self.tsteps_var, width=14))
        _add(basic, r, "Parallel envs:", ttk.Spinbox(basic, from_=1, to=16,
              textvariable=self.n_envs_var, width=14))
        _add(basic, r, "Learning rate:", ttk.Entry(basic, textvariable=self.lr_var, width=16))
        _add(basic, r, "Entropy coef:", ttk.Spinbox(basic, from_=0.0, to=0.5,
              increment=0.01, textvariable=self.ent_coef_var, width=14))
        _add(basic, r, "PPO n_steps:", ttk.Spinbox(basic, from_=32, to=8192, increment=32,
              textvariable=self.nsteps_var, width=14))
        _add(basic, r, "PPO batch size:", ttk.Spinbox(basic, from_=8, to=4096, increment=8,
              textvariable=self.batch_var, width=14))
        _add(basic, r, "Obs type:", ttk.Combobox(basic, textvariable=self.obs_type_var,
              values=["tiles", "pixels"], state="readonly", width=13))
        _add(basic, r, "Start level:", ttk.Combobox(basic, textvariable=self.start_level_var,
              values=LEVEL_CHOICES, state="readonly", width=13))
        _add(basic, r, "Device:", ttk.Combobox(basic, textvariable=self.device_var,
              values=["auto", "cpu", "mps", "cuda"], state="readonly", width=13))

        r = [0]
        _add(advanced, r, "Action repeat:", ttk.Spinbox(advanced, from_=1, to=16,
              textvariable=self.action_repeat_var, width=14))
        _add(advanced, r, "Frame stack:", ttk.Spinbox(advanced, from_=1, to=16,
              textvariable=self.frame_stack_var, width=14))
        _add(advanced, r, "PPO n_epochs:", ttk.Spinbox(advanced, from_=1, to=30,
              textvariable=self.n_epochs_var, width=14))
        _add(advanced, r, "Seed:", ttk.Spinbox(advanced, from_=0, to=2_147_483_647,
              textvariable=self.seed_var, width=14))
        _add(advanced, r, "Checkpoint / N steps:", ttk.Spinbox(advanced, from_=100, to=10_000_000,
              increment=1000, textvariable=self.ckpt_freq_var, width=14))
        _add(advanced, r, "Eval / N steps:", ttk.Spinbox(advanced, from_=100, to=10_000_000,
              increment=1000, textvariable=self.eval_freq_var, width=14))
        _add(advanced, r, "Eval episodes:", ttk.Spinbox(advanced, from_=1, to=100,
              textvariable=self.n_eval_var, width=14))

        # ---- Checkboxes + buttons + progress bar ----
        controls = ttk.Frame(parent)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        controls.columnconfigure(0, weight=1)

        self.resume_check = ttk.Checkbutton(controls,
            text="Resume from newest checkpoint (uses selected run name)",
            variable=self.resume_var)
        self.resume_check.grid(row=0, column=0, sticky="w")
        self.preview_check = ttk.Checkbutton(controls,
            text="Show live preview on Play tab (slower training)",
            variable=self.preview_var)
        self.preview_check.grid(row=1, column=0, sticky="w")

        btns = ttk.Frame(controls)
        btns.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.btn_train_start = ttk.Button(btns, text="Start training",
                                           command=self.start_training)
        self.btn_train_start.pack(side="left", padx=(0, 4))
        self.btn_train_stop = ttk.Button(btns, text="Stop",
                                          command=self.stop_training, state="disabled")
        self.btn_train_stop.pack(side="left", padx=4)

        pb_frame = ttk.Frame(controls)
        pb_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        pb_frame.columnconfigure(0, weight=1)
        self.train_progress = ttk.Progressbar(pb_frame, mode="determinate", maximum=100)
        self.train_progress.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.train_progress_label = ttk.Label(pb_frame, text="—", width=22)
        self.train_progress_label.grid(row=0, column=1, sticky="e")

        # ---- Live stats + training log, side by side ----
        monitor = ttk.Frame(parent)
        monitor.grid(row=3, column=0, sticky="nsew", pady=(0, 4))
        monitor.columnconfigure(0, weight=0)
        monitor.columnconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        stats = ttk.LabelFrame(monitor, text="Live stats", padding=8)
        stats.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.stat_labels: dict[str, ttk.Label] = {}
        stats.columnconfigure(1, weight=1)
        rows_config = [
            ("status", "idle"),
            ("total_timesteps", "—"),
            ("ep_rew_mean", "—"),
            ("ep_len_mean", "—"),
            ("fps", "—"),
            ("time_elapsed", "—"),
            # Game state — populated from the preview thread when preview is on
            ("world", "—"),
            ("lives", "—"),
            ("coins", "—"),
            ("max_x", "—"),
        ]
        for i, (k, v) in enumerate(rows_config):
            ttk.Label(stats, text=f"{k}:").grid(row=i, column=0, sticky="w", padx=(0, 12))
            lbl = ttk.Label(stats, text=v, font=("Menlo", 11, "bold"))
            lbl.grid(row=i, column=1, sticky="w")
            self.stat_labels[k] = lbl

        log_frame = ttk.LabelFrame(monitor, text="Training log", padding=4)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="none", font=("Menlo", 10),
                                background="#111", foreground="#ddd", insertbackground="#ddd")
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        yscroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=yscroll.set)

    def _build_play(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, width=320)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        # Model selector
        model_frame = ttk.LabelFrame(left, text="Model", padding=8)
        model_frame.pack(fill="x", pady=(0, 6))
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                         state="readonly", width=32)
        self.model_combo.pack(fill="x", pady=(0, 4))
        self.btn_model_refresh = ttk.Button(model_frame, text="Refresh model list",
                                             command=self._refresh_models)
        self.btn_model_refresh.pack(fill="x")

        # Play-tab widgets that must be disabled while training is running
        self._play_widgets: list[tk.Widget] = [self.model_combo, self.btn_model_refresh]

        # Options
        opts_frame = ttk.LabelFrame(left, text="Options", padding=8)
        opts_frame.pack(fill="x", pady=(0, 6))
        opts_frame.columnconfigure(1, weight=1)

        self.play_episodes_var = tk.IntVar(value=3)
        self.play_max_steps_var = tk.IntVar(value=2000)
        self.play_stochastic_var = tk.BooleanVar(value=False)
        self.play_action_repeat_var = tk.IntVar(value=4)
        self.play_frame_stack_var = tk.IntVar(value=4)
        self.play_speed_label_var = tk.StringVar(value=SPEED_CHOICES[1][0])

        row = 0
        def add(label, widget):
            nonlocal row
            ttk.Label(opts_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            widget.grid(row=row, column=1, sticky="ew", pady=2, padx=6)
            row += 1
            self._play_widgets.append(widget)

        add("Episodes:", ttk.Spinbox(opts_frame, from_=1, to=100,
                                       textvariable=self.play_episodes_var, width=10))
        add("Max steps (0=∞):", ttk.Spinbox(opts_frame, from_=0, to=1_000_000, increment=100,
                                              textvariable=self.play_max_steps_var, width=10))
        add("Action repeat:", ttk.Spinbox(opts_frame, from_=1, to=16,
                                            textvariable=self.play_action_repeat_var, width=10))
        add("Frame stack:", ttk.Spinbox(opts_frame, from_=1, to=8,
                                          textvariable=self.play_frame_stack_var, width=10))
        self.play_obs_type_var = tk.StringVar(value="tiles")
        add("Obs type (match training):",
            ttk.Combobox(opts_frame, textvariable=self.play_obs_type_var,
                          values=["tiles", "pixels"], state="readonly", width=10))
        self.play_level_var = tk.StringVar(value="default")
        add("Start level:", ttk.Combobox(opts_frame, textvariable=self.play_level_var,
                                          values=LEVEL_CHOICES, state="readonly", width=10))
        add("Speed:", ttk.Combobox(opts_frame, textvariable=self.play_speed_label_var,
                                    values=[c[0] for c in SPEED_CHOICES],
                                    state="readonly", width=15))
        stoch_check = ttk.Checkbutton(opts_frame, text="Stochastic (sample actions)",
                                       variable=self.play_stochastic_var)
        stoch_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        self._play_widgets.append(stoch_check)

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        self.btn_play_start = ttk.Button(btns, text="Start playing", command=self.start_playing)
        self.btn_play_start.pack(fill="x", pady=2)
        self.btn_play_stop = ttk.Button(btns, text="Stop", command=self.stop_playing, state="disabled")
        self.btn_play_stop.pack(fill="x", pady=2)
        # Start button gets disabled while training runs (Stop stays as-is
        # since it only enables if a play episode is actually in progress).
        self._play_widgets.append(self.btn_play_start)

        # Live stats
        pstats = ttk.LabelFrame(left, text="Live episode", padding=8)
        pstats.pack(fill="x", pady=6)
        pstats.columnconfigure(1, weight=1)

        self.play_stat_vars = {
            "episode": tk.StringVar(value="—"),
            "reward": tk.StringVar(value="—"),
            "world": tk.StringVar(value="—"),
            "x": tk.StringVar(value="—"),
            "steps": tk.StringVar(value="—"),
            "lives": tk.StringVar(value="—"),
            "coins": tk.StringVar(value="—"),
            "action": tk.StringVar(value="—"),
        }
        for i, key in enumerate(["episode", "reward", "world", "x", "steps",
                                  "lives", "coins", "action"]):
            ttk.Label(pstats, text=f"{key}:").grid(row=i, column=0, sticky="w", padx=(0, 8))
            ttk.Label(pstats, textvariable=self.play_stat_vars[key],
                      font=("Menlo", 11, "bold")).grid(row=i, column=1, sticky="w")

        self.play_status_var = tk.StringVar(value="idle")
        ttk.Label(left, textvariable=self.play_status_var, font=("Menlo", 10),
                  wraplength=280).pack(fill="x", pady=(6, 0))

        # Game canvas
        canvas_frame = ttk.LabelFrame(right, text=f"Game Boy ({SCALE}×)", padding=6)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H,
                                bg="#222", highlightthickness=0)
        self.canvas.pack()
        placeholder = np.full((GAME_H, GAME_W, 3), 34, dtype=np.uint8)
        self._tk_img = ImageTk.PhotoImage(
            Image.fromarray(placeholder).resize((CANVAS_W, CANVAS_H), Image.NEAREST)
        )
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    # ---------- Tuning tab ----------

    def _build_tune(self, parent: ttk.Frame) -> None:
        """Hyperparameter tuning tab. Runs each trial as a `main.py train`
        subprocess (grid search over the sweep JSON, or random search of N
        combos), then reads the trial's best eval reward from
        evaluations.npz. Best-first results, one-click save-as-preset or
        load-into-train-tab from any row.
        """
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)

        # Widgets on the Tune tab that must be disabled while a train or
        # tune subprocess is running (prevents mid-run edits + confused UX).
        self._tune_config_widgets: list[tk.Widget] = []

        # ---- Base config (loaded from preset) ----
        cfg_frame = ttk.LabelFrame(parent, text="Trial config", padding=6)
        cfg_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        cfg_frame.columnconfigure(1, weight=1)
        cfg_frame.columnconfigure(3, weight=1)

        ttk.Label(cfg_frame, text="Base preset:").grid(row=0, column=0, sticky="w")
        self.tune_preset_var = tk.StringVar(value="")
        self.tune_preset_combo = ttk.Combobox(cfg_frame, textvariable=self.tune_preset_var,
                                               state="readonly")
        self.tune_preset_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self._tune_config_widgets.append(self.tune_preset_combo)

        ttk.Label(cfg_frame, text="Timesteps / trial:").grid(row=0, column=2, sticky="w")
        self.tune_trial_steps_var = tk.IntVar(value=50_000)
        tune_steps_spin = ttk.Spinbox(cfg_frame, from_=2000, to=10_000_000, increment=10000,
                                       textvariable=self.tune_trial_steps_var, width=12)
        tune_steps_spin.grid(row=0, column=3, sticky="w", padx=6)
        self._tune_config_widgets.append(tune_steps_spin)

        ttk.Label(cfg_frame, text="Run-name prefix:").grid(row=1, column=0, sticky="w",
                                                             pady=(4, 0))
        self.tune_run_prefix_var = tk.StringVar(value="tune")
        tune_prefix_entry = ttk.Entry(cfg_frame, textvariable=self.tune_run_prefix_var, width=20)
        tune_prefix_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
        self._tune_config_widgets.append(tune_prefix_entry)

        ttk.Label(cfg_frame, text="Metric:").grid(row=1, column=2, sticky="w", pady=(4, 0))
        self.tune_metric_var = tk.StringVar(value="best eval reward")
        tune_metric_combo = ttk.Combobox(cfg_frame, textvariable=self.tune_metric_var,
                                          values=["best eval reward", "final eval reward",
                                                  "mean eval reward"],
                                          state="readonly", width=18)
        tune_metric_combo.grid(row=1, column=3, sticky="w", padx=6, pady=(4, 0))
        self._tune_config_widgets.append(tune_metric_combo)

        # Search-type row
        ttk.Label(cfg_frame, text="Search:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        search_frame = ttk.Frame(cfg_frame)
        search_frame.grid(row=2, column=1, columnspan=3, sticky="w",
                           padx=6, pady=(4, 0))
        self.tune_search_type_var = tk.StringVar(value="grid")
        rb_grid = ttk.Radiobutton(search_frame, text="Grid (all combinations)",
                                    variable=self.tune_search_type_var, value="grid")
        rb_grid.pack(side="left", padx=(0, 12))
        rb_random = ttk.Radiobutton(search_frame, text="Random",
                                     variable=self.tune_search_type_var, value="random")
        rb_random.pack(side="left", padx=(0, 4))
        ttk.Label(search_frame, text="N trials:").pack(side="left", padx=(4, 2))
        self.tune_n_random_var = tk.IntVar(value=10)
        n_random_spin = ttk.Spinbox(search_frame, from_=1, to=1000,
                                     textvariable=self.tune_n_random_var, width=6)
        n_random_spin.pack(side="left")
        self._tune_config_widgets.extend([rb_grid, rb_random, n_random_spin])

        # ---- Sweep spec ----
        sweep_frame = ttk.LabelFrame(parent, text="Sweep (JSON — param → list of values)",
                                       padding=6)
        sweep_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        sweep_frame.columnconfigure(0, weight=1)
        self.tune_sweep_text = tk.Text(sweep_frame, height=5, wrap="word",
                                        font=("Menlo", 10))
        self.tune_sweep_text.grid(row=0, column=0, sticky="ew")
        self.tune_sweep_text.insert("1.0",
            '{\n  "ent_coef": [0.005, 0.01, 0.02, 0.05],\n'
            '  "learning_rate": [1e-4, 2.5e-4, 5e-4]\n}')
        self._tune_config_widgets.append(self.tune_sweep_text)
        ttk.Label(sweep_frame, text="Cartesian product of all values → total trials",
                   foreground="#888").grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ---- Buttons + progress ----
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ctrl.columnconfigure(2, weight=1)
        self.btn_tune_start = ttk.Button(ctrl, text="Start tuning",
                                          command=self.start_tuning)
        self.btn_tune_start.grid(row=0, column=0, padx=(0, 4))
        self.btn_tune_stop = ttk.Button(ctrl, text="Stop",
                                         command=self.stop_tuning, state="disabled")
        self.btn_tune_stop.grid(row=0, column=1, padx=4)
        self.tune_progress = ttk.Progressbar(ctrl, mode="determinate", maximum=100)
        self.tune_progress.grid(row=0, column=2, sticky="ew", padx=6)
        self.tune_progress_label = ttk.Label(ctrl, text="—", width=18)
        self.tune_progress_label.grid(row=0, column=3, sticky="e")

        # ---- Results table ----
        res_frame = ttk.LabelFrame(parent, text="Results (best first)", padding=4)
        res_frame.grid(row=4, column=0, sticky="nsew")
        res_frame.columnconfigure(0, weight=1)
        res_frame.rowconfigure(0, weight=1)
        self.tune_tree = ttk.Treeview(res_frame,
                                       columns=("trial", "config", "reward",
                                                "steps", "duration", "run"),
                                       show="headings", height=8)
        for c, w, anc in [("trial", 50, "e"), ("config", 320, "w"),
                          ("reward", 90, "e"), ("steps", 80, "e"),
                          ("duration", 80, "e"), ("run", 160, "w")]:
            self.tune_tree.heading(c, text=c)
            self.tune_tree.column(c, width=w, anchor=anc)
        self.tune_tree.grid(row=0, column=0, sticky="nsew")
        tune_scroll = ttk.Scrollbar(res_frame, command=self.tune_tree.yview)
        tune_scroll.grid(row=0, column=1, sticky="ns")
        self.tune_tree.config(yscrollcommand=tune_scroll.set)

        # ---- Actions on selected row ----
        actions = ttk.Frame(parent)
        actions.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.btn_tune_save_preset = ttk.Button(
            actions, text="Save selected as preset",
            command=self._tune_save_as_preset)
        self.btn_tune_save_preset.pack(side="left", padx=(0, 4))
        self.btn_tune_to_train = ttk.Button(
            actions, text="Load selected into Train tab",
            command=self._tune_load_into_train)
        self.btn_tune_to_train.pack(side="left", padx=4)
        ttk.Label(actions, text="→ then switch to the Train tab and Start",
                   foreground="#888").pack(side="left", padx=8)

        # Per-trial full-config cache, keyed by trial number (int).
        # Populated by _tuning_loop, consumed by save/load buttons.
        self._tune_configs: dict[int, dict] = {}

    # ---------- Tuning execution ----------

    def start_tuning(self) -> None:
        if self.tune_thread is not None and self.tune_thread.is_alive():
            messagebox.showwarning("Tune", "Tuning already running.")
            return
        if self.train_proc is not None and self.train_proc.poll() is None:
            messagebox.showwarning(
                "Tune",
                "Training is currently running. Stop training first."
            )
            return
        preset_name = self.tune_preset_var.get()
        if not preset_name or preset_name not in self._all_presets:
            messagebox.showerror("Tune", "Pick a base preset first.")
            return
        base = dict(self._all_presets[preset_name])
        try:
            trial_steps = int(self.tune_trial_steps_var.get())
            n_random = max(1, int(self.tune_n_random_var.get()))
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Tune", f"Invalid trial-steps / N-trials value:\n{e}")
            return

        # Parse sweep JSON
        try:
            import json as _json
            sweep = _json.loads(self.tune_sweep_text.get("1.0", "end"))
            assert isinstance(sweep, dict), "sweep must be a dict"
            for k, v in sweep.items():
                assert isinstance(v, list) and v, f"{k}: must be non-empty list"
        except Exception as e:
            messagebox.showerror("Tune", f"Bad sweep JSON: {e}")
            return

        keys = list(sweep.keys())
        search_type = self.tune_search_type_var.get()
        if search_type == "grid":
            from itertools import product
            combos = list(product(*(sweep[k] for k in keys)))
        else:  # random
            import random as _random
            rng = _random.Random()
            combos = [tuple(rng.choice(sweep[k]) for k in keys) for _ in range(n_random)]
        if not combos:
            messagebox.showerror("Tune", "Sweep produced 0 configs.")
            return

        # Clear results + per-trial cache
        for row in self.tune_tree.get_children():
            self.tune_tree.delete(row)
        self._tune_configs.clear()
        self.tune_progress["value"] = 0
        self.tune_progress_label.config(text=f"0 / {len(combos)}")
        self.btn_tune_start.config(state="disabled")
        self.btn_tune_stop.config(state="normal")
        # Lock the train-start button so the user can't kick off a training
        # session that would collide with the tuner's own train subprocess.
        self.btn_train_start.config(state="disabled")
        self._set_subprocess_widgets_disabled(True)
        self.tune_stop.clear()

        self.tune_thread = threading.Thread(
            target=self._tuning_loop,
            args=(base, keys, combos, trial_steps,
                  self.tune_run_prefix_var.get().strip() or "tune",
                  self.tune_metric_var.get()),
            daemon=True,
        )
        self.tune_thread.start()

    def stop_tuning(self) -> None:
        self.tune_stop.set()
        if self.tune_proc is not None and self.tune_proc.poll() is None:
            try:
                self.tune_proc.terminate()
            except Exception:
                pass

    def _tuning_loop(self, base_cfg: dict, keys: list, combos: list,
                     trial_steps: int, run_prefix: str, metric_name: str) -> None:
        """Runs each config as a subprocess. After completion, reads eval
        reward from `models/mario/<run>/logs/evaluations.npz`. Emits progress
        + rows via stats_queue.
        """
        import numpy as np
        from itertools import chain
        game = base_cfg.get("game", "mario")
        best_eval_freq = min(trial_steps, base_cfg.get("eval_freq", trial_steps))
        for i, combo in enumerate(combos):
            if self.tune_stop.is_set():
                self.stats_queue.put(("tune_status", "cancelled"))
                break
            overrides = dict(zip(keys, combo))
            cfg = {**base_cfg, **overrides}
            # Snapshot the exact config that will run so save/load buttons can
            # recover it later. Trial number is 1-indexed to match display.
            self._tune_configs[i + 1] = dict(cfg)
            run_name = f"{run_prefix}-{i + 1:03d}"
            cfg_summary = ", ".join(f"{k}={overrides[k]}" for k in keys)
            self.stats_queue.put(("tune_progress", i, len(combos),
                                   f"trial {i + 1}/{len(combos)}: {cfg_summary}"))
            cmd = build_train_cmd(cfg, run_name, timesteps=trial_steps,
                                  checkpoint_freq=max(trial_steps, 1),  # only save at end
                                  eval_freq=best_eval_freq)
            t0 = time.time()
            try:
                self.tune_proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=str(self.project_dir),
                )
                rc = self.tune_proc.wait()
            except Exception as e:
                self.stats_queue.put(("log", f"[tune] trial {i + 1} err: {e}\n"))
                continue
            duration = time.time() - t0
            self.tune_proc = None
            if self.tune_stop.is_set():
                break
            # Read eval reward
            eval_file = (self.project_dir / "models" / game / run_name
                          / "logs" / "evaluations.npz")
            reward = float("-inf")
            actual_steps = trial_steps
            if eval_file.exists():
                try:
                    data = np.load(eval_file)
                    means = data["results"].mean(axis=1)
                    if metric_name == "best eval reward":
                        reward = float(means.max())
                    elif metric_name == "final eval reward":
                        reward = float(means[-1])
                    else:  # mean
                        reward = float(means.mean())
                    actual_steps = int(data["timesteps"][-1])
                except Exception:
                    pass
            self.stats_queue.put((
                "tune_result",
                {
                    "trial": i + 1,
                    "config": cfg_summary,
                    "reward": reward,
                    "steps": actual_steps,
                    "duration": duration,
                    "run": run_name,
                },
            ))
        self.stats_queue.put(("tune_done", None))

    def _selected_tune_config(self) -> dict | None:
        """Return the full config of the currently-selected tune-tab row,
        or None if nothing selected."""
        sel = self.tune_tree.selection()
        if not sel:
            messagebox.showwarning("Tune",
                                    "Select a row in the results table first.")
            return None
        vals = self.tune_tree.item(sel[0], "values")
        try:
            trial_num = int(vals[0])
        except (ValueError, IndexError):
            return None
        cfg = self._tune_configs.get(trial_num)
        if cfg is None:
            messagebox.showerror("Tune", f"No cached config for trial {trial_num}.")
            return None
        return cfg

    def _tune_save_as_preset(self) -> None:
        """Save the selected trial's full config as a user preset so it
        appears in the Train tab's preset dropdown."""
        cfg = self._selected_tune_config()
        if cfg is None:
            return
        name = simpledialog.askstring(
            "Save tune result as preset",
            "Name this preset (this exact config, including tuned hyperparams,\n"
            "will be saved for use on the Train tab):",
            parent=self.root,
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return
        try:
            presets.upsert(name, cfg)
        except ValueError as e:
            messagebox.showerror("Save preset", str(e))
            return
        self._refresh_presets()
        # Auto-select the just-saved preset on the Train tab too
        self.preset_var.set(name)
        self._apply_preset()
        messagebox.showinfo(
            "Saved",
            f"Preset '{name}' saved. It's now selected on the Train tab —\n"
            f"switch there and click Start to train with these hyperparams.",
        )

    def _tune_load_into_train(self) -> None:
        """Populate the Train tab's widgets from the selected trial's
        config, without saving a preset. One-click promotion from tuning
        to a full training run."""
        cfg = self._selected_tune_config()
        if cfg is None:
            return
        var_map = {
            "game": self.game_var,
            "timesteps": self.tsteps_var,
            "n_envs": self.n_envs_var,
            "ent_coef": self.ent_coef_var,
            "learning_rate": self.lr_var,
            "n_steps": self.nsteps_var,
            "batch_size": self.batch_var,
            "device": self.device_var,
            "obs_type": self.obs_type_var,
            "start_level": self.start_level_var,
            "action_repeat": self.action_repeat_var,
            "frame_stack": self.frame_stack_var,
            "n_epochs": self.n_epochs_var,
            "seed": self.seed_var,
            "checkpoint_freq": self.ckpt_freq_var,
            "eval_freq": self.eval_freq_var,
            "n_eval_episodes": self.n_eval_var,
        }
        for key, var in var_map.items():
            if key in cfg:
                var.set(cfg[key])
        self._refresh_run_names()
        # Clear the Train-tab preset selection to signal "custom config"
        self.preset_var.set("")
        # Play tab should track the same obs/level for a smooth handoff
        self.play_obs_type_var.set(cfg.get("obs_type", self.play_obs_type_var.get()))
        if "start_level" in cfg:
            self.play_level_var.set(cfg["start_level"])
        self._refresh_models()
        messagebox.showinfo(
            "Loaded",
            "Trial config copied into the Train tab. Switch tabs and click Start.",
        )

    # ---------- Model discovery ----------

    def _list_models(self, game_filter: str | None = None) -> list[tuple[str, Path]]:
        """Return (label, path) tuples for models under models/<game>/<run>/."""
        entries: list[tuple[str, Path]] = []

        # Skip internal caches like `_level_states/` (per-level save-states)
        def _is_run(name: str) -> bool:
            return not name.startswith("_")

        # models/<game>/<run>/logs/best_model.zip
        for p in sorted(self.project_dir.glob("models/*/*/logs/best_model.zip")):
            run = p.parent.parent.name
            game = p.parent.parent.parent.name
            if not _is_run(run):
                continue
            if game_filter is None or game == game_filter:
                entries.append((f"{game}/{run} — best", p))

        # models/<game>/<run>/checkpoints/final.zip
        for p in sorted(self.project_dir.glob("models/*/*/checkpoints/final.zip")):
            run = p.parent.parent.name
            game = p.parent.parent.parent.name
            if not _is_run(run):
                continue
            if game_filter is None or game == game_filter:
                entries.append((f"{game}/{run} — final", p))

        # models/<game>/<run>/checkpoints/ppo_*_steps.zip
        for run_dir in sorted(self.project_dir.glob("models/*/*/")):
            game = run_dir.parent.name
            run = run_dir.name
            if not _is_run(run):
                continue
            if game_filter is not None and game != game_filter:
                continue
            snaps = sorted((run_dir / "checkpoints").glob("ppo_*_steps.zip"),
                           key=lambda p: int(p.stem.split("_")[1]))
            for p in snaps:
                steps = int(p.stem.split("_")[1])
                entries.append((f"{game}/{run} — {steps:,} steps", p))

        return entries

    # ---------- Presets ----------

    def _refresh_presets(self) -> None:
        self._all_presets = presets.load_all()
        recommended = "Mario — Campaign, recommended (~15 min)"
        def sort_key(name):
            return (name != recommended, not presets.is_builtin(name), name.lower())
        names = sorted(self._all_presets.keys(), key=sort_key)
        # Also feed the Tune tab's preset combo
        if hasattr(self, "tune_preset_combo"):
            self.tune_preset_combo["values"] = names
            if not self.tune_preset_var.get() and names:
                self.tune_preset_var.set(names[0])
        prev = self.preset_var.get()
        self.preset_combo["values"] = names
        if prev in names:
            self.preset_var.set(prev)
        elif names:
            self.preset_var.set(names[0])
            self._apply_preset()

    def _apply_preset(self) -> None:
        name = self.preset_var.get()
        cfg = self._all_presets.get(name)
        if cfg is None:
            return
        if "game" in cfg and cfg["game"] in GAMES and cfg["game"] != self.game_var.get():
            self.game_var.set(cfg["game"])
            self._refresh_run_names()
        var_map = {
            "timesteps": self.tsteps_var,
            "n_envs": self.n_envs_var,
            "ent_coef": self.ent_coef_var,
            "learning_rate": self.lr_var,
            "n_steps": self.nsteps_var,
            "batch_size": self.batch_var,
            "device": self.device_var,
            "obs_type": self.obs_type_var,
            "start_level": self.start_level_var,
            "action_repeat": self.action_repeat_var,
            "frame_stack": self.frame_stack_var,
            "n_epochs": self.n_epochs_var,
            "seed": self.seed_var,
            "checkpoint_freq": self.ckpt_freq_var,
            "eval_freq": self.eval_freq_var,
            "n_eval_episodes": self.n_eval_var,
        }
        for key, var in var_map.items():
            if key in cfg:
                var.set(cfg[key])
        # Keep the Play tab in sync so playing a just-trained model works
        if "obs_type" in cfg:
            self.play_obs_type_var.set(cfg["obs_type"])
        if "start_level" in cfg:
            self.play_level_var.set(cfg["start_level"])
        self._refresh_models()

    def _current_config(self) -> dict:
        return {
            "game": self.game_var.get(),
            "timesteps": int(self.tsteps_var.get()),
            "n_envs": int(self.n_envs_var.get()),
            "ent_coef": float(self.ent_coef_var.get()),
            "learning_rate": float(self.lr_var.get()),
            "n_steps": int(self.nsteps_var.get()),
            "batch_size": int(self.batch_var.get()),
            "device": self.device_var.get(),
            "obs_type": self.obs_type_var.get(),
            "start_level": self.start_level_var.get(),
            "action_repeat": int(self.action_repeat_var.get()),
            "frame_stack": int(self.frame_stack_var.get()),
            "n_epochs": int(self.n_epochs_var.get()),
            "seed": int(self.seed_var.get()),
            "checkpoint_freq": int(self.ckpt_freq_var.get()),
            "eval_freq": int(self.eval_freq_var.get()),
            "n_eval_episodes": int(self.n_eval_var.get()),
        }

    def _save_preset(self) -> None:
        name = simpledialog.askstring("Save preset",
                                       "Preset name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        try:
            presets.upsert(name, self._current_config())
        except ValueError as e:
            messagebox.showerror("Save preset", str(e))
            return
        self._refresh_presets()
        self.preset_var.set(name)

    def _delete_preset(self) -> None:
        name = self.preset_var.get()
        if not name:
            return
        if presets.is_builtin(name):
            messagebox.showinfo("Delete preset",
                                f"'{name}' is a built-in preset and cannot be deleted.")
            return
        if not messagebox.askyesno("Delete preset", f"Delete preset '{name}'?"):
            return
        try:
            presets.delete(name)
        except ValueError as e:
            messagebox.showerror("Delete preset", str(e))
            return
        self.preset_var.set("")
        self._refresh_presets()

    # ---------- Models ----------

    def _refresh_models(self) -> None:
        # Prioritize models for the currently-selected game
        game_filter = self.game_var.get()
        priority = self._list_models(game_filter)
        others = [e for e in self._list_models(None) if e not in priority]
        entries = priority + others

        self._model_paths = {label: path for label, path in entries}
        labels = list(self._model_paths.keys())
        prev = self.model_var.get()
        self.model_combo["values"] = labels
        if prev in labels:
            self.model_var.set(prev)
        elif labels:
            self.model_var.set(labels[0])
        else:
            self.model_var.set("")

    # ---------- Training ----------

    def start_training(self) -> None:
        if self.train_proc is not None and self.train_proc.poll() is None:
            messagebox.showwarning("Training", "Training is already running.")
            return
        # Stop any running play session — it would compete with training
        # for CPU and both threads pushing to frame_queue would clash on
        # the shared canvas.
        if self.play_thread is not None and self.play_thread.is_alive():
            self.play_stop.set()
        try:
            cfg = self._current_config()
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Training", f"A training field has an invalid value:\n{e}")
            return
        game = cfg["game"]
        run_name = self.run_name_var.get().strip() or "default"
        cmd = build_train_cmd(cfg, run_name, resume=bool(self.resume_var.get()))

        self._append_log(f"$ {' '.join(cmd)}\n")
        self.train_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(self.project_dir),
        )
        self.btn_train_start.config(state="disabled")
        self.btn_train_stop.config(state="normal")
        # Lock the tune-start button so the user can't kick off a tuning
        # session that would collide with the running trainer.
        self.btn_tune_start.config(state="disabled")
        self._set_subprocess_widgets_disabled(True)
        self.stat_labels["status"].config(text="running")
        self._train_target_steps = max(1, cfg["timesteps"])
        # Baseline = model's prior step count. First observed total_timesteps
        # will be (baseline + rollout_size), so we subtract rollout_size to find it.
        self._train_rollout_size = max(1, cfg["n_steps"] * cfg["n_envs"])
        self._train_baseline_steps = None
        self.train_progress["value"] = 0
        self.train_progress_label.config(text=f"0 / {self._train_target_steps:,}")
        threading.Thread(target=self._read_train_stdout, daemon=True).start()

        if self.preview_var.get():
            self._start_preview(game, run_name)

    def _read_train_stdout(self) -> None:
        assert self.train_proc is not None and self.train_proc.stdout is not None
        for line in self.train_proc.stdout:
            self.stats_queue.put(("log", line))
            m = STAT_LINE.search(line)
            if m and m.group(1) in TRACKED_STATS:
                self.stats_queue.put(("stat", m.group(1), m.group(2)))
        rc = self.train_proc.wait()
        self.stats_queue.put(("train_done", rc))

    def stop_training(self) -> None:
        if self.train_proc is None:
            return
        self._append_log("[gui] stopping training...\n")
        self.preview_stop.set()
        self.train_proc.terminate()
        try:
            self.train_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.train_proc.kill()

    def _start_preview(self, game: str, run_name: str) -> None:
        """Watch the newest saved model play alongside training, in the Play tab canvas."""
        self.preview_stop.clear()
        action_repeat = int(self.action_repeat_var.get())
        frame_stack = int(self.frame_stack_var.get())
        obs_type = self.obs_type_var.get()
        # Preview mirrors the training's start_level so what you see matches
        # what the model is training on: "random" → new random level per
        # preview episode; specific level → that level; default → 1-1.
        raw_level = self.start_level_var.get()
        preview_level = None if raw_level == "default" else raw_level

        def _preview_loop():
            from stable_baselines3.common.vec_env import DummyVecEnv as DVE
            from stable_baselines3.common.monitor import Monitor as Mon
            model_path = self.project_dir / "models" / game / run_name / "logs" / "best_model.zip"
            model = None
            model_mtime = 0.0
            vec = None
            self.stats_queue.put(("log", f"[preview] waiting for first best_model at {model_path}\n"))
            while (
                not self.preview_stop.is_set()
                and self.train_proc is not None
                and self.train_proc.poll() is None
            ):
                if not model_path.exists():
                    time.sleep(2)
                    continue
                mt = model_path.stat().st_mtime
                if model is None or mt > model_mtime:
                    try:
                        if vec is not None:
                            vec.close()
                        spec = GAMES[game]
                        rom = self.project_dir / "ROMs" / spec.rom_file
                        pyboy = PyBoy(str(rom), window_type="null",
                                      game_wrapper=True, disable_renderer=False)
                        pyboy.set_emulation_speed(0)

                        def _grab():
                            arr = pyboy.botsupport_manager().screen().screen_ndarray()
                            try:
                                self.frame_queue.put_nowait(np.asarray(arr, dtype=np.uint8))
                            except queue.Full:
                                pass

                        base = MarioEnv(pyboy, frame_skip=action_repeat,
                                        obs_type=obs_type, tick_callback=_grab,
                                        start_level=preview_level)
                        vec = DVE([lambda env=Mon(base): env])
                        if obs_type == "pixels":
                            vec = VecTransposeImage(vec)
                        if frame_stack > 1:
                            order = "first" if obs_type == "pixels" else "last"
                            vec = VecFrameStack(vec, n_stack=frame_stack, channels_order=order)
                        model = PPO.load(str(model_path), env=vec, device="cpu")
                        model_mtime = mt
                        obs = vec.reset()
                        done = [False]
                        # Per-episode tracking so the Play tab's live-episode
                        # panel gets the same info as a real play session
                        ep_num = 1
                        ep_reward = 0.0
                        ep_steps = 0
                        self.stats_queue.put(("log", f"[preview] loaded {model_path.name}\n"))
                        self.stats_queue.put(("play_status",
                                              f"preview — {model_path.name}"))
                        self.stats_queue.put(("play_stat", "episode",
                                              f"{ep_num} (preview)"))
                    except Exception as e:
                        self.stats_queue.put(("log", f"[preview] load err: {e}\n"))
                        time.sleep(3)
                        continue

                # Play one step, mildly paced so we don't spin CPU
                action, _ = model.predict(obs, deterministic=False)
                obs, r, done, info = vec.step(action)
                ep_reward += float(r[0])
                ep_steps += 1
                # Push game state to BOTH the Train tab stats and the Play
                # tab's live-episode panel (the canvas shows the preview).
                if info and isinstance(info[0], dict):
                    i0 = info[0]
                    if "world" in i0:
                        w_str = f"{i0['world'][0]}-{i0['world'][1]}"
                        self.stats_queue.put(("stat", "world", w_str))
                        self.stats_queue.put(("play_stat", "world", w_str))
                    if "lives" in i0:
                        v = str(i0["lives"])
                        self.stats_queue.put(("stat", "lives", v))
                        self.stats_queue.put(("play_stat", "lives", v))
                    if "coins" in i0:
                        v = str(i0["coins"])
                        self.stats_queue.put(("stat", "coins", v))
                        self.stats_queue.put(("play_stat", "coins", v))
                    if "max_x" in i0:
                        self.stats_queue.put(("stat", "max_x", str(i0["max_x"])))
                    if "x" in i0:
                        self.stats_queue.put((
                            "play_stat", "x",
                            f"{i0['x']} (max {i0.get('max_x', '?')})"
                        ))
                act_id = int(np.asarray(action).flat[0])
                if game == "mario" and act_id < len(MARIO_ACTION_NAMES):
                    self.stats_queue.put(("play_stat", "action",
                                          MARIO_ACTION_NAMES[act_id]))
                self.stats_queue.put(("play_stat", "reward", f"{ep_reward:.1f}"))
                self.stats_queue.put(("play_stat", "steps", str(ep_steps)))
                if done[0]:
                    obs = vec.reset()
                    ep_num += 1
                    ep_reward = 0.0
                    ep_steps = 0
                    self.stats_queue.put(("play_stat", "episode",
                                          f"{ep_num} (preview)"))
                time.sleep(0.02)  # cap preview at ~50 env-steps/sec

            if vec is not None:
                try:
                    vec.close()
                except Exception:
                    pass
            self.stats_queue.put(("log", "[preview] stopped\n"))
            self.stats_queue.put(("play_status", "idle"))

        self.preview_thread = threading.Thread(target=_preview_loop, daemon=True)
        self.preview_thread.start()

    # ---------- Playing ----------

    def start_playing(self) -> None:
        if self.play_thread is not None and self.play_thread.is_alive():
            messagebox.showwarning("Play", "Playback already running.")
            return
        # Defensive: the Start button is disabled while training runs, but
        # a rogue keypress or scripted event shouldn't be able to bypass it.
        if self.train_proc is not None and self.train_proc.poll() is None:
            messagebox.showwarning(
                "Play",
                "Training is currently running. Stop training first, or wait for it to finish."
            )
            return
        game = self.game_var.get()
        label = self.model_var.get()
        if not label:
            self._refresh_models()
            label = self.model_var.get()
        if not label:
            messagebox.showerror(
                "Play",
                f"No trained model found for '{self.game_var.get()}'. Train first, or\n"
                f"place a .zip at models/{self.game_var.get()}/<run>/logs/best_model.zip\n"
                f"or models/{self.game_var.get()}/<run>/checkpoints/final.zip"
            )
            return
        model_path = self._model_paths.get(label)
        if model_path is None or not model_path.exists():
            messagebox.showerror("Play", f"Model file missing: {model_path}")
            return

        speed_mult = dict(SPEED_CHOICES)[self.play_speed_label_var.get()]
        action_repeat = int(self.play_action_repeat_var.get())
        frame_stack = int(self.play_frame_stack_var.get())
        obs_type = self.play_obs_type_var.get()
        raw_level = self.play_level_var.get()
        start_level = None if raw_level == "default" else raw_level
        episodes = int(self.play_episodes_var.get())
        max_steps = int(self.play_max_steps_var.get())
        deterministic = not self.play_stochastic_var.get()

        self.play_stop.clear()
        self.btn_play_start.config(state="disabled")
        self.btn_play_stop.config(state="normal")
        self.play_status_var.set(f"loading {model_path.name}...")
        for v in self.play_stat_vars.values():
            v.set("—")

        self.play_thread = threading.Thread(
            target=self._play_loop,
            args=(game, model_path, speed_mult, action_repeat, frame_stack,
                  obs_type, start_level, episodes, max_steps, deterministic),
            daemon=True,
        )
        self.play_thread.start()

    def stop_playing(self) -> None:
        self.play_stop.set()

    def _play_loop(
        self, game: str, model_path: Path, speed_mult: float,
        action_repeat: int, frame_stack: int, obs_type: str,
        start_level, episodes: int, max_steps: int, deterministic: bool,
    ) -> None:
        try:
            spec = GAMES[game]
            rom_path = self.project_dir / "ROMs" / spec.rom_file
            # Bootstrap save-state file for the requested level if missing
            # (subprocess with timeout, so a stuck level can't wedge the GUI).
            if game == "mario" and start_level is not None:
                if start_level in ("random", "sequential", "marathon"):
                    ensure_level_states(rom_path)
                else:
                    w, l = (int(x) for x in start_level.split("-"))
                    ensure_level_states(rom_path, [(w, l)])
            # For tiles we still need rendering enabled — we grab pixel frames
            # for the canvas even though the model sees tiles.
            pyboy = PyBoy(str(rom_path), window_type="null",
                          game_wrapper=True, disable_renderer=False)
            pyboy.set_emulation_speed(0)  # we pace manually

            # frame grabber runs after every emulator tick → smooth ~60Hz canvas
            def _grab_frame():
                arr = pyboy.botsupport_manager().screen().screen_ndarray()
                try:
                    self.frame_queue.put_nowait(np.asarray(arr, dtype=np.uint8))
                except queue.Full:
                    pass

            if game == "mario":
                base = MarioEnv(pyboy, frame_skip=action_repeat,
                                obs_type=obs_type, tick_callback=_grab_frame,
                                start_level=start_level)
            else:
                if pyboy.game_wrapper() is None:
                    pyboy.stop(save=False)
                    raise RuntimeError(
                        f"Game '{game}' has no PyBoy game_wrapper; only games with a "
                        f"built-in wrapper (mario, kirby) can be played."
                    )
                if obs_type != "pixels":
                    pyboy.stop(save=False)
                    raise RuntimeError(f"obs_type={obs_type!r} only supported for mario")
                base = pyboy.openai_gym(observation_type="raw", action_type="press")

            vec = DummyVecEnv([lambda env=Monitor(base): env])
            if obs_type == "pixels":
                vec = VecTransposeImage(vec)
            if frame_stack > 1:
                order = "first" if obs_type == "pixels" else "last"
                vec = VecFrameStack(vec, n_stack=frame_stack, channels_order=order)

            self.stats_queue.put(("play_status", f"loaded {model_path.name}"))
            model = PPO.load(str(model_path), env=vec, device="cpu")

            # target wall-clock time per env step; 0 = no throttle
            step_period = (action_repeat / 60.0) / speed_mult if speed_mult > 0 else 0.0

            for ep in range(episodes):
                if self.play_stop.is_set():
                    break
                obs = vec.reset()
                _grab_frame()
                total = 0.0
                steps = 0
                done = [False]
                self.stats_queue.put(("play_stat", "episode", f"{ep + 1}/{episodes}"))
                while not done[0] and not self.play_stop.is_set():
                    t0 = time.time()
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = vec.step(action)
                    total += float(reward[0])
                    steps += 1
                    act_id = int(np.asarray(action).flat[0])
                    self.stats_queue.put(("play_stat", "action", MARIO_ACTION_NAMES[act_id]
                                          if game == "mario" and act_id < len(MARIO_ACTION_NAMES)
                                          else str(act_id)))
                    self.stats_queue.put(("play_stat", "reward", f"{total:.1f}"))
                    self.stats_queue.put(("play_stat", "steps", str(steps)))
                    if info and isinstance(info[0], dict):
                        i0 = info[0]
                        if "x" in i0:
                            self.stats_queue.put(("play_stat", "x",
                                                  f"{i0['x']} (max {i0.get('max_x', '?')})"))
                        if "world" in i0:
                            w = i0["world"]
                            self.stats_queue.put(("play_stat", "world", f"{w[0]}-{w[1]}"))
                        if "lives" in i0:
                            self.stats_queue.put(("play_stat", "lives", str(i0["lives"])))
                        if "coins" in i0:
                            self.stats_queue.put(("play_stat", "coins", str(i0["coins"])))
                    if max_steps and steps >= max_steps:
                        break
                    # Pace to target speed
                    if step_period > 0:
                        remaining = step_period - (time.time() - t0)
                        if remaining > 0:
                            time.sleep(remaining)
                self.stats_queue.put((
                    "play_status",
                    f"Episode {ep + 1} done: reward={total:.1f} steps={steps}",
                ))

            vec.close()
            self.stats_queue.put(("play_done", None))
        except Exception:
            import traceback
            self.stats_queue.put(("play_error", traceback.format_exc()))

    # ---------- Event loop ----------

    def _pump(self) -> None:
        if self._closing:
            return
        # Drain stats queue
        try:
            while True:
                item = self.stats_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "stat":
                    _, key, val = item
                    if key in self.stat_labels:
                        self.stat_labels[key].config(text=val)
                    if key == "total_timesteps":
                        try:
                            done_abs = int(val)
                            if self._train_baseline_steps is None:
                                # Fresh: first obs = rollout_size, baseline = 0
                                # Resume: first obs = prev + rollout_size, baseline = prev
                                self._train_baseline_steps = max(
                                    0, done_abs - self._train_rollout_size
                                )
                            new_steps = done_abs - self._train_baseline_steps
                            pct = min(100.0, 100.0 * new_steps / max(1, self._train_target_steps))
                            self.train_progress["value"] = pct
                            self.train_progress_label.config(
                                text=f"{new_steps:,} / {self._train_target_steps:,}"
                            )
                        except ValueError:
                            pass
                elif kind == "train_done":
                    rc = item[1]
                    self.stat_labels["status"].config(text=f"finished (rc={rc})")
                    self.btn_train_start.config(state="normal")
                    self.btn_train_stop.config(state="disabled")
                    self.btn_tune_start.config(state="normal")
                    self._set_subprocess_widgets_disabled(False)
                    self.train_proc = None
                    self.preview_stop.set()
                    # Reset progress bar unless training reached the target cleanly
                    if rc == 0:
                        self.train_progress["value"] = 100
                    else:
                        self.train_progress["value"] = 0
                        self.train_progress_label.config(text="cancelled")
                    self._refresh_models()      # new checkpoints may have arrived
                    self._refresh_run_names()   # new run dir may have appeared
                elif kind == "play_status":
                    self.play_status_var.set(item[1])
                elif kind == "play_stat":
                    _, key, val = item
                    if key in self.play_stat_vars:
                        self.play_stat_vars[key].set(val)
                elif kind == "play_done":
                    self.play_status_var.set("done")
                    # Don't re-enable Play Start if training is still running —
                    # the whole play tab should stay disabled in that case.
                    training_active = (self.train_proc is not None
                                       and self.train_proc.poll() is None)
                    if not training_active:
                        self.btn_play_start.config(state="normal")
                    self.btn_play_stop.config(state="disabled")
                elif kind == "play_error":
                    messagebox.showerror("Play error", item[1])
                    self.play_status_var.set("error")
                    training_active = (self.train_proc is not None
                                       and self.train_proc.poll() is None)
                    if not training_active:
                        self.btn_play_start.config(state="normal")
                    self.btn_play_stop.config(state="disabled")
                elif kind == "tune_progress":
                    _, done_n, total, label = item
                    pct = 100.0 * done_n / max(1, total)
                    self.tune_progress["value"] = pct
                    self.tune_progress_label.config(text=f"{done_n}/{total}")
                    self._append_log(f"[tune] {label}\n")
                elif kind == "tune_result":
                    r = item[1]
                    reward_str = f"{r['reward']:.1f}" if r['reward'] != float("-inf") else "—"
                    # Insert sorted by reward (best first)
                    inserted = False
                    for existing in self.tune_tree.get_children():
                        existing_vals = self.tune_tree.item(existing, "values")
                        try:
                            existing_r = float(existing_vals[2])
                        except (ValueError, IndexError):
                            existing_r = float("-inf")
                        if r["reward"] > existing_r:
                            self.tune_tree.insert("", self.tune_tree.index(existing),
                                                   values=(r["trial"], r["config"],
                                                           reward_str, r["steps"],
                                                           f"{r['duration']:.0f}s",
                                                           r["run"]))
                            inserted = True
                            break
                    if not inserted:
                        self.tune_tree.insert("", "end",
                                               values=(r["trial"], r["config"],
                                                       reward_str, r["steps"],
                                                       f"{r['duration']:.0f}s",
                                                       r["run"]))
                    self._append_log(
                        f"[tune] trial {r['trial']} done → reward={reward_str} "
                        f"steps={r['steps']} ({r['duration']:.0f}s)\n"
                    )
                elif kind == "tune_status":
                    self._append_log(f"[tune] {item[1]}\n")
                elif kind == "tune_done":
                    self.btn_tune_start.config(state="normal")
                    self.btn_tune_stop.config(state="disabled")
                    self.btn_train_start.config(state="normal")
                    self._set_subprocess_widgets_disabled(False)
                    self.tune_proc = None
                    self._append_log("[tune] done\n")
        except queue.Empty:
            pass

        # Drain latest frame
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            img = Image.fromarray(latest).resize((CANVAS_W, CANVAS_H), Image.NEAREST)
            self._tk_img = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

        try:
            self.root.after(33, self._pump)
        except tk.TclError:
            pass

    def _set_subprocess_widgets_disabled(self, disabled: bool) -> None:
        """Enable/disable every input widget on the Train, Play, and Tune
        tabs while any training or tuning subprocess is running. Prevents
        mid-run edits and prevents starting another CPU-hungry session
        that would collide with the one already going. Action buttons
        (Start / Stop / other-tab Start) are managed by their own callers
        — this method only touches inputs and secondary buttons.
        """
        # Comboboxes with state='readonly' need 'readonly' (not 'normal') to
        # re-enable properly; other widgets use 'normal'.
        def _state_for(w, on: bool):
            if not on:
                return "disabled"
            if isinstance(w, ttk.Combobox):
                return "readonly"
            return "normal"

        enabled = not disabled
        all_widgets: list[tk.Widget] = [
            *self._train_config_widgets,
            self.preset_combo, self.btn_preset_save, self.btn_preset_delete,
            self.btn_preset_reload, self.resume_check, self.preview_check,
            *self._play_widgets,
            *self._tune_config_widgets,
            self.btn_tune_save_preset, self.btn_tune_to_train,
        ]
        for w in all_widgets:
            try:
                w.config(state=_state_for(w, enabled))
            except tk.TclError:
                pass

    def _append_log(self, text: str) -> None:
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
        if self.tune_proc is not None and self.tune_proc.poll() is None:
            try:
                self.tune_proc.terminate()
                self.tune_proc.wait(timeout=5)
            except Exception:
                try:
                    self.tune_proc.kill()
                except Exception:
                    pass
        if self.train_proc is not None and self.train_proc.poll() is None:
            self.train_proc.terminate()
            try:
                self.train_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.train_proc.kill()
        self.root.destroy()


MARIO_ACTION_NAMES = [
    "NOOP", "RIGHT", "LEFT", "JUMP",
    "RIGHT+JUMP", "RIGHT+RUN", "RIGHT+RUN+JUMP",
    "LEFT+JUMP", "LEFT+RUN", "LEFT+RUN+JUMP",
    "DOWN",
]


def run() -> None:
    root = tk.Tk()
    GameBoyAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
