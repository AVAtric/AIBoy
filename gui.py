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
from env import GAMES, MarioEnv

SCALE = 3
GAME_W, GAME_H = 160, 144
CANVAS_W, CANVAS_H = GAME_W * SCALE, GAME_H * SCALE

STAT_LINE = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([\S]+)\s+\|")
TRACKED_STATS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "fps", "time_elapsed")

SPEED_CHOICES = [("0.5×", 0.5), ("1× (real time)", 1.0), ("2×", 2.0),
                 ("4×", 4.0), ("Unlimited", 0.0)]


class GameBoyAIGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Game Boy AI — Train & Play")
        root.geometry("1080x760")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.project_dir = Path(__file__).resolve().parent

        # Cross-thread channels
        self.stats_queue: queue.Queue = queue.Queue()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Subprocess / thread state
        self.train_proc: subprocess.Popen | None = None
        self.play_stop = threading.Event()
        self.play_thread: threading.Thread | None = None

        self._model_paths: dict[str, Path] = {}
        self._closing = False

        self._build_ui()
        self._refresh_models()
        self._refresh_presets()
        self._pump()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Game:").pack(side="left")
        self.game_var = tk.StringVar(value="mario")
        game_combo = ttk.Combobox(top, textvariable=self.game_var, values=sorted(GAMES),
                                   state="readonly", width=10)
        game_combo.pack(side="left", padx=6)
        game_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_models())
        ttk.Label(top, text="Run name (blank = same as game):").pack(side="left", padx=(20, 4))
        self.run_name_var = tk.StringVar(value="")
        run_entry = ttk.Entry(top, textvariable=self.run_name_var, width=16)
        run_entry.pack(side="left")
        run_entry.bind("<KeyRelease>", lambda e: self._refresh_models())

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        train_tab = ttk.Frame(nb, padding=10)
        play_tab = ttk.Frame(nb, padding=10)
        nb.add(train_tab, text="Train")
        nb.add(play_tab, text="Play")
        self._build_train(train_tab)
        self._build_play(play_tab)

    def _build_train(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        self.tsteps_var = tk.IntVar(value=500_000)
        self.n_envs_var = tk.IntVar(value=8)
        self.ent_coef_var = tk.DoubleVar(value=0.01)
        self.lr_var = tk.DoubleVar(value=2.5e-4)
        self.nsteps_var = tk.IntVar(value=256)
        self.batch_var = tk.IntVar(value=64)
        self.device_var = tk.StringVar(value="auto")
        self.resume_var = tk.BooleanVar(value=False)

        # ---- Presets bar ----
        preset_frame = ttk.LabelFrame(parent, text="Preset", padding=6)
        preset_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        preset_frame.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value="")
        self.preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                          state="readonly")
        self.preset_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        ttk.Button(preset_frame, text="Save as…",
                   command=self._save_preset).grid(row=0, column=1, padx=2)
        ttk.Button(preset_frame, text="Delete",
                   command=self._delete_preset).grid(row=0, column=2, padx=2)
        ttk.Button(preset_frame, text="Reload list",
                   command=self._refresh_presets).grid(row=0, column=3, padx=2)

        row = 1
        def add(label, widget):
            nonlocal row
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3, padx=6)
            row += 1

        add("Total timesteps:", ttk.Spinbox(parent, from_=1000, to=20_000_000,
                                             increment=10_000, textvariable=self.tsteps_var, width=18))
        add("Parallel envs:", ttk.Spinbox(parent, from_=1, to=16,
                                           textvariable=self.n_envs_var, width=18))
        add("Learning rate:", ttk.Entry(parent, textvariable=self.lr_var, width=20))
        add("PPO n_steps:", ttk.Spinbox(parent, from_=32, to=8192, increment=32,
                                          textvariable=self.nsteps_var, width=18))
        add("PPO batch size:", ttk.Spinbox(parent, from_=8, to=4096, increment=8,
                                            textvariable=self.batch_var, width=18))
        add("Entropy coef:", ttk.Spinbox(parent, from_=0.0, to=0.5,
                                          increment=0.01, textvariable=self.ent_coef_var, width=18))
        self.obs_type_var = tk.StringVar(value="tiles")
        add("Obs type:", ttk.Combobox(parent, textvariable=self.obs_type_var,
                                       values=["tiles", "pixels"], state="readonly", width=17))
        add("Device:", ttk.Combobox(parent, textvariable=self.device_var,
                                     values=["auto", "cpu", "mps", "cuda"], state="readonly", width=17))
        ttk.Checkbutton(parent, text="Resume from newest checkpoint",
                        variable=self.resume_var).grid(row=row, column=0, columnspan=2,
                                                       sticky="w", pady=6)
        row += 1

        btns = ttk.Frame(parent)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        self.btn_train_start = ttk.Button(btns, text="Start training", command=self.start_training)
        self.btn_train_start.pack(side="left", padx=4)
        self.btn_train_stop = ttk.Button(btns, text="Stop", command=self.stop_training, state="disabled")
        self.btn_train_stop.pack(side="left", padx=4)
        row += 1

        stats = ttk.LabelFrame(parent, text="Live stats", padding=8)
        stats.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        self.stat_labels: dict[str, ttk.Label] = {}
        stats.columnconfigure(1, weight=1)
        rows_config = [("status", "idle"), ("total_timesteps", "—"), ("ep_rew_mean", "—"),
                       ("ep_len_mean", "—"), ("fps", "—"), ("time_elapsed", "—")]
        for i, (k, v) in enumerate(rows_config):
            ttk.Label(stats, text=f"{k}:").grid(row=i, column=0, sticky="w", padx=(0, 12))
            lbl = ttk.Label(stats, text=v, font=("Menlo", 11, "bold"))
            lbl.grid(row=i, column=1, sticky="w")
            self.stat_labels[k] = lbl

        log_frame = ttk.LabelFrame(parent, text="Training log", padding=4)
        log_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=6)
        parent.rowconfigure(row, weight=1)
        self.log_text = tk.Text(log_frame, height=10, wrap="none", font=("Menlo", 10),
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
        ttk.Button(model_frame, text="Refresh model list",
                   command=self._refresh_models).pack(fill="x")

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
        add("Speed:", ttk.Combobox(opts_frame, textvariable=self.play_speed_label_var,
                                    values=[c[0] for c in SPEED_CHOICES],
                                    state="readonly", width=15))
        ttk.Checkbutton(opts_frame, text="Stochastic (sample actions)",
                        variable=self.play_stochastic_var).grid(row=row, column=0,
                                                                columnspan=2, sticky="w", pady=4)

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        self.btn_play_start = ttk.Button(btns, text="Start playing", command=self.start_playing)
        self.btn_play_start.pack(fill="x", pady=2)
        self.btn_play_stop = ttk.Button(btns, text="Stop", command=self.stop_playing, state="disabled")
        self.btn_play_stop.pack(fill="x", pady=2)

        # Live stats
        pstats = ttk.LabelFrame(left, text="Live episode", padding=8)
        pstats.pack(fill="x", pady=6)
        pstats.columnconfigure(1, weight=1)

        self.play_stat_vars = {
            "episode": tk.StringVar(value="—"),
            "reward": tk.StringVar(value="—"),
            "x": tk.StringVar(value="—"),
            "steps": tk.StringVar(value="—"),
            "action": tk.StringVar(value="—"),
        }
        for i, key in enumerate(["episode", "reward", "x", "steps", "action"]):
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

    # ---------- Model discovery ----------

    def _list_models(self, game_filter: str | None = None) -> list[tuple[str, Path]]:
        """Return (label, path) tuples for models under models/<game>/<run>/."""
        entries: list[tuple[str, Path]] = []

        # models/<game>/<run>/logs/best_model.zip
        for p in sorted(self.project_dir.glob("models/*/*/logs/best_model.zip")):
            run = p.parent.parent.name
            game = p.parent.parent.parent.name
            if game_filter is None or game == game_filter:
                entries.append((f"{game}/{run} — best", p))

        # models/<game>/<run>/checkpoints/final.zip
        for p in sorted(self.project_dir.glob("models/*/*/checkpoints/final.zip")):
            run = p.parent.parent.name
            game = p.parent.parent.parent.name
            if game_filter is None or game == game_filter:
                entries.append((f"{game}/{run} — final", p))

        # models/<game>/<run>/checkpoints/ppo_*_steps.zip
        for run_dir in sorted(self.project_dir.glob("models/*/*/")):
            game = run_dir.parent.name
            run = run_dir.name
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
        recommended = "Mario — Balanced tiles (recommended)"
        def sort_key(name):
            return (name != recommended, not presets.is_builtin(name), name.lower())
        names = sorted(self._all_presets.keys(), key=sort_key)
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
        if "game" in cfg and cfg["game"] in GAMES:
            self.game_var.set(cfg["game"])
        var_map = {
            "timesteps": self.tsteps_var,
            "n_envs": self.n_envs_var,
            "ent_coef": self.ent_coef_var,
            "learning_rate": self.lr_var,
            "n_steps": self.nsteps_var,
            "batch_size": self.batch_var,
            "device": self.device_var,
            "obs_type": self.obs_type_var,
        }
        for key, var in var_map.items():
            if key in cfg:
                var.set(cfg[key])
        # Keep the Play tab's obs_type in sync so playing a just-trained model works
        if "obs_type" in cfg:
            self.play_obs_type_var.set(cfg["obs_type"])
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
        game = self.game_var.get()
        run_name = self.run_name_var.get().strip() or game
        cmd = [
            sys.executable, "-u", "main.py", "train",
            "--game", game,
            "--n-envs", str(self.n_envs_var.get()),
            "--timesteps", str(self.tsteps_var.get()),
            "--ent-coef", str(self.ent_coef_var.get()),
            "--learning-rate", str(self.lr_var.get()),
            "--n-steps", str(self.nsteps_var.get()),
            "--batch-size", str(self.batch_var.get()),
            "--obs-type", self.obs_type_var.get(),
            "--device", self.device_var.get(),
            "--run-name", run_name,
        ]
        if self.resume_var.get():
            cmd.append("--resume")

        self._append_log(f"$ {' '.join(cmd)}\n")
        self.train_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(self.project_dir),
        )
        self.btn_train_start.config(state="disabled")
        self.btn_train_stop.config(state="normal")
        self.stat_labels["status"].config(text="running")
        threading.Thread(target=self._read_train_stdout, daemon=True).start()

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
        self.train_proc.terminate()
        try:
            self.train_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.train_proc.kill()

    # ---------- Playing ----------

    def start_playing(self) -> None:
        if self.play_thread is not None and self.play_thread.is_alive():
            messagebox.showwarning("Play", "Playback already running.")
            return
        game = self.game_var.get()
        label = self.model_var.get()
        if not label:
            self._refresh_models()
            label = self.model_var.get()
        if not label:
            messagebox.showerror(
                "Play",
                f"No trained model found. Train first, or place a .zip in\n"
                f"logs/<run>/best_model.zip or checkpoints/<run>/final.zip"
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
                  obs_type, episodes, max_steps, deterministic),
            daemon=True,
        )
        self.play_thread.start()

    def stop_playing(self) -> None:
        self.play_stop.set()

    def _play_loop(
        self, game: str, model_path: Path, speed_mult: float,
        action_repeat: int, frame_stack: int, obs_type: str,
        episodes: int, max_steps: int, deterministic: bool,
    ) -> None:
        try:
            spec = GAMES[game]
            rom_path = self.project_dir / "ROMs" / spec.rom_file
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
                                obs_type=obs_type, tick_callback=_grab_frame)
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
                    if info and isinstance(info[0], dict) and "x" in info[0]:
                        self.stats_queue.put(("play_stat", "x",
                                              f"{info[0]['x']} (max {info[0].get('max_x', '?')})"))
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
                elif kind == "train_done":
                    rc = item[1]
                    self.stat_labels["status"].config(text=f"finished (rc={rc})")
                    self.btn_train_start.config(state="normal")
                    self.btn_train_stop.config(state="disabled")
                    self.train_proc = None
                    self._refresh_models()  # new checkpoints may have arrived
                elif kind == "play_status":
                    self.play_status_var.set(item[1])
                elif kind == "play_stat":
                    _, key, val = item
                    if key in self.play_stat_vars:
                        self.play_stat_vars[key].set(val)
                elif kind == "play_done":
                    self.play_status_var.set("done")
                    self.btn_play_start.config(state="normal")
                    self.btn_play_stop.config(state="disabled")
                elif kind == "play_error":
                    messagebox.showerror("Play error", item[1])
                    self.play_status_var.set("error")
                    self.btn_play_start.config(state="normal")
                    self.btn_play_stop.config(state="disabled")
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

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 2000:
            self.log_text.delete("1.0", "500.0")

    def _on_close(self) -> None:
        self._closing = True
        self.play_stop.set()
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
]


def run() -> None:
    root = tk.Tk()
    GameBoyAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
