"""The AIboy window.

Tabs, in workflow order:
  Wizard      guided Search -> Save -> Train -> Watch flow (see wizard.py)
  Train       a single headless training run (subprocess) with live stats
  Tune        hyperparameter sweeps over short training trials
  Presets     browse / edit / organise presets (see presets_tab.py)
  Experience  everything AIboy has tried so far (see experience_tab.py)
plus the Game Boy screen on the right, which plays any saved model.

Training and tuning run `main.py train` as subprocesses so the window stays
responsive; playback runs on a background thread (player.py). All
cross-thread traffic goes through two queues drained by `_pump()`. Every
finished trainer process appends to the experience file, which the app
re-reads to reuse known results, refresh its estimates and fill the
Experience tab.
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

from aiboy import APP_NAME, presets, runs, tuning
from aiboy.experience import Experience
from aiboy.games import OBS_TYPES, RomInfo, discover_roms, level_choices, probe_rom
from aiboy.gui.experience_tab import ExperienceTab
from aiboy.gui.player import (CANVAS_H, CANVAS_W, GAME_H, GAME_W, SPEED_CHOICES, EmbeddedPlayer,
                              IntroVideo, LatestFrame)
from aiboy.gui.presets_tab import PresetsTab
from aiboy.paths import DATA_DIR, FROZEN
from aiboy.gui.widgets import MONO, MONO_BOLD, THEME, ConfigForm, WidgetLock, make_table, setup_styles

PROJECT_DIR = DATA_DIR      # ROMs/, models/, presets and the error log live here
DEFAULT_GAME = "mario"
TENSORBOARD_PORT = 6006

# SB3's verbose=1 table: "|    ep_rew_mean    | 123     |"
STAT_LINE = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([\S]+)\s+\|")
TRACKED_STATS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "fps", "time_elapsed")

LEVEL_CHOICES = level_choices()

STOP_GRACE_SECONDS = 20     # SIGINT -> trainer saves final.zip; SIGKILL after this
FLASH_SECONDS = 6           # transient status-bar messages


IDLE_SCREEN_TEXT = "No video"
INTRO_DISABLED = os.environ.get("AIBOY_NO_INTRO") == "1"      # tests: silent, no video


def idle_screen(intro: IntroVideo | None = None) -> Image.Image:
    """Screen for the canvas while no video is active: the intro video's
    last frame with the logo, with a pixel-font "No video" under it. Falls
    back to a drawn screen when the intro assets are missing."""
    from PIL import ImageDraw
    if intro is not None:
        base = Image.fromarray(intro.idle_frame())                        # 160x144
        draw = ImageDraw.Draw(base)
        w = draw.textlength(IDLE_SCREEN_TEXT)
        draw.text(((GAME_W - w) / 2, 92), IDLE_SCREEN_TEXT, fill=(2, 10, 7))
        return base.resize((CANVAS_W, CANVAS_H), Image.NEAREST)
    img = Image.new("RGB", (GAME_W, GAME_H), (155, 188, 15))
    draw = ImageDraw.Draw(img)
    dark = (15, 56, 15)
    draw.rectangle([8, 8, GAME_W - 9, GAME_H - 9], outline=dark, width=2)
    for i, line in enumerate((APP_NAME, IDLE_SCREEN_TEXT)):
        w = draw.textlength(line)
        draw.text(((GAME_W - w) / 2, 56 + i * 20), line, fill=dark)
    return img.resize((CANVAS_W, CANVAS_H), Image.NEAREST)


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


class AIboyGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"{APP_NAME} — Super Mario Land")
        root.geometry("1300x900")
        root.minsize(1240, 820)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Cross-thread channels
        self.stats_queue: queue.Queue = queue.Queue()
        self.frames = LatestFrame()
        self.player = EmbeddedPlayer(self.frames, self.stats_queue)

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
        self._train_rate = tuning.RateEstimator()

        self._model_paths: dict[str, Path] = {}
        self._all_presets: dict[str, dict] = {}
        self._tune_results: dict[int, tuning.ConfigResult] = {}
        self._canvas_img_id: int | None = None
        self._tk_img: ImageTk.PhotoImage | None = None
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.train_progress_text = tk.StringVar(value="—")
        self._closing = False
        self._train_run_name = ""
        self._flash: tuple[str, float] | None = None
        self.intro: IntroVideo | None = None
        self.intro_stop = threading.Event()
        if IntroVideo.available() and not INTRO_DISABLED:
            try:
                self.intro = IntroVideo()
            except Exception as e:          # a broken asset must not stop the app
                sys.stderr.write(f"intro video not loaded: {e}\n")
        self.roms: dict[str, RomInfo] = {}
        self.experience = Experience()          # what earlier trials and runs taught AIboy
        self.wizard = None
        self.presets_tab = None
        self.experience_tab = None
        self._input_lock = WidgetLock()

        self._build_ui()
        self.rescan_roms()
        self.refresh_run_names()
        self.refresh_models()
        self.refresh_presets()
        self._pump()
        if self.intro is not None:
            # Boot video with sound once the window is up; ends on the idle
            # frame ("No video") unless something else takes the screen.
            self.root.after(300, lambda: self.intro.play(
                self.frames, self.intro_stop,
                on_done=lambda: self.stats_queue.put(("intro_done",))))

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
            self.game_status_label.config(foreground=THEME.err)
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
        colors = {"ok": THEME.ok, "experimental": THEME.warn, "unsupported": THEME.err,
                  "checking": THEME.muted}
        self.game_status_label.config(foreground=colors[level])
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
        from aiboy.gui.wizard import WizardTab

        setup_styles(self.root)

        # Top bar: which ROM to work with, and whether it can run here.
        top = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Game:").pack(side="left")
        self.game_var = tk.StringVar(value=DEFAULT_GAME)
        self.game_combo = ttk.Combobox(top, textvariable=self.game_var, state="readonly", width=12)
        self.game_combo.pack(side="left", padx=6)
        self.game_combo.bind("<<ComboboxSelected>>", lambda e: self._on_game_changed())
        self.game_status_var = tk.StringVar(value="")
        self.game_status_label = ttk.Label(top, textvariable=self.game_status_var, font=MONO)
        self.game_status_label.pack(side="left", padx=(6, 0))
        self.btn_rescan = ttk.Button(top, text="Rescan ROMs", command=self.rescan_roms)
        self.btn_rescan.pack(side="right")

        # Status bar (bottom): what is running right now.
        self.status_bar_var = tk.StringVar(value="Idle")
        bar = ttk.Frame(self.root, padding=(10, 4))
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status_bar_var, font=MONO,
                  foreground=THEME.text_soft).pack(side="left")
        self.disk_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.disk_var, foreground=THEME.muted, font=MONO).pack(
            side="right")

        # Main area: workflow tabs on the left, the Game Boy screen on the right.
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.nb = ttk.Notebook(main)
        self.nb.grid(row=0, column=0, sticky="nsew")
        screen = ttk.LabelFrame(main, text="Game Boy screen", padding=8)
        screen.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        self.wizard_tab = ttk.Frame(self.nb, padding=8)
        self.train_tab = ttk.Frame(self.nb, padding=8)
        self.tune_tab = ttk.Frame(self.nb, padding=8)
        self.presets_frame = ttk.Frame(self.nb, padding=8)
        self.experience_frame = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.wizard_tab, text="Wizard")
        self.nb.add(self.train_tab, text="Train")
        self.nb.add(self.tune_tab, text="Tune")
        self.nb.add(self.presets_frame, text="Presets")
        self.nb.add(self.experience_frame, text="Experience")
        # Train and Tune define the variables the wizard mirrors; build them first.
        self._build_train(self.train_tab)
        self._build_tune(self.tune_tab)
        self._build_screen(screen)
        self.wizard = WizardTab(self, self.wizard_tab)
        self.presets_tab = PresetsTab(self, self.presets_frame)
        self.experience_tab = ExperienceTab(self, self.experience_frame)
        self.nb.select(self.wizard_tab)

    # ---------- experience ----------

    def fps_for(self, cfg: dict | None) -> float | None:
        """Env-steps/sec this computer measured on earlier runs with the same
        setup (game, observation type, emulators); None until it has any."""
        if not cfg:
            return None
        return self.experience.fps(cfg)

    def known_trials(self, plan: dict) -> int:
        """How many of a sweep plan's trials AIboy already has a result for."""
        if not plan or not plan.get("skip_done", True):
            return 0
        return sum(self.experience.known_seeds(plan["base"], c, plan["trial_steps"], plan["n_seeds"])
                   for c in plan["combos"])

    def on_experience_changed(self) -> None:
        """Re-read the experience file after a trainer finished and tell the
        tabs that show or use it."""
        self.experience.reload()
        if self.experience_tab is not None:
            self.experience_tab.refresh()
        if self.wizard is not None:
            self.wizard.on_experience_changed()
        self.tune_update_summary()

    def show_tab(self, tab: ttk.Frame) -> None:
        self.nb.select(tab)

    def _init_canvas(self, canvas: tk.Canvas) -> None:
        """Idle screen ("No video") when nothing else is on the canvas."""
        self._tk_img = ImageTk.PhotoImage(idle_screen(self.intro))
        if self._canvas_img_id is None:
            self._canvas_img_id = canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        else:
            canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

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
        # Recompute the "modified" marker whenever the selected preset changes
        # (the form is often filled before the name is set).
        self.preset_var.trace_add("write", lambda *_a: self._on_train_form_changed())
        self.btn_preset_save = ttk.Button(preset_frame, text="Save as…", command=self._save_preset)
        self.btn_preset_save.grid(row=0, column=1, padx=2)
        self.btn_preset_manage = ttk.Button(preset_frame, text="Manage…",
                                            command=lambda: self.show_tab(self.presets_frame))
        self.btn_preset_manage.grid(row=0, column=2, padx=2)
        self.preset_state_var = tk.StringVar(value="")
        ttk.Label(preset_frame, textvariable=self.preset_state_var, foreground=THEME.warn).grid(
            row=1, column=0, columnspan=3, sticky="w")

        # ---- Parameters (Basic | Advanced) ----
        self.form = ConfigForm(parent, on_change=self._on_train_form_changed)
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
        self.run_name_combo.bind("<KeyRelease>", lambda e: self._on_run_name_changed())
        self.run_name_combo.bind("<<ComboboxSelected>>", lambda e: self._on_run_name_changed())
        self.resume_check = ttk.Checkbutton(
            run_row, text="Resume from newest checkpoint", variable=self.resume_var)
        self.resume_check.pack(side="left", padx=(18, 0))
        self.run_hint_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.run_hint_var, foreground=THEME.muted, font=MONO,
                  wraplength=640).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.preview_check = ttk.Checkbutton(
            controls, text="Show live preview on the screen while training (slower)",
            variable=self.preview_var)
        self.preview_check.grid(row=2, column=0, sticky="w", pady=(4, 0))

        btns = ttk.Frame(controls)
        btns.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.btn_train_start = ttk.Button(btns, text="Start training", command=self.start_training)
        self.btn_train_start.pack(side="left", padx=(0, 4))
        self.btn_train_stop = ttk.Button(btns, text="Stop", command=self.stop_training,
                                         state="disabled")
        self.btn_train_stop.pack(side="left", padx=4)
        ttk.Button(btns, text="TensorBoard", command=self.open_tensorboard).pack(
            side="left", padx=(16, 4))
        self.btn_run_delete = ttk.Button(btns, text="Delete run…", command=self._delete_run)
        self.btn_run_delete.pack(side="right")
        self.btn_run_compact = ttk.Button(btns, text="Compact run…", command=self._compact_run)
        self.btn_run_compact.pack(side="right", padx=(0, 4))
        self.btn_run_open = ttk.Button(btns, text="Open folder", command=self._open_run_folder)
        self.btn_run_open.pack(side="right", padx=(0, 4))

        pb_frame = ttk.Frame(controls)
        pb_frame.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        pb_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(pb_frame, mode="determinate", maximum=100,
                        variable=self.train_progress_var).grid(row=0, column=0, sticky="ew",
                                                               padx=(0, 6))
        ttk.Label(pb_frame, textvariable=self.train_progress_text, width=32, anchor="e").grid(
            row=0, column=1, sticky="e")

        # ---- Training log (full width, grows). Live stats are on the screen panel. ----
        parent.rowconfigure(3, weight=1)
        log_frame = ttk.LabelFrame(parent, text="Training log", padding=4)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, width=40, wrap="none", font=MONO,
                                background="#111", foreground="#ddd", insertbackground="#ddd")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=yscroll.set)

    # ---------- Screen panel (always visible) ----------

    def _build_screen(self, parent: ttk.Frame) -> None:
        """The emulator view plus the controls to play a saved model. Live
        preview during training and the wizard's Watch step render here too,
        so nothing has to switch tabs to be seen."""
        self.canvas = tk.Canvas(parent, width=CANVAS_W, height=CANVAS_H, bg="#222",
                                highlightthickness=0)
        self.canvas.pack()
        self._init_canvas(self.canvas)
        self.play_status_var = tk.StringVar(value="idle")
        ttk.Label(parent, textvariable=self.play_status_var, font=MONO, wraplength=CANVAS_W).pack(
            fill="x", pady=(4, 2))

        # Live episode, three columns.
        stats = ttk.LabelFrame(parent, text="Live episode", padding=(6, 2))
        stats.pack(fill="x", pady=(0, 4))
        keys = ["episode", "world", "power", "reward", "x", "lives", "steps", "coins", "action"]
        self.play_stat_vars = {k: tk.StringVar(value="—") for k in keys}
        for i, key in enumerate(keys):
            row, col = divmod(i, 3)
            ttk.Label(stats, text=f"{key}:", foreground=THEME.muted).grid(
                row=row, column=col * 2, sticky="w", padx=(0 if col == 0 else 10, 4))
            ttk.Label(stats, textvariable=self.play_stat_vars[key], font=MONO_BOLD,
                      width=8 if key != "action" else 13, anchor="w").grid(
                row=row, column=col * 2 + 1, sticky="w")

        # Training, always visible: status line, progress + ETA, key stats.
        train_box = ttk.LabelFrame(parent, text="Training", padding=(6, 2))
        train_box.pack(fill="x", pady=(0, 4))
        train_box.columnconfigure(0, weight=1)
        self.stat_vars: dict[str, tk.StringVar] = {
            k: tk.StringVar(value=v) for k, v in
            [("status", "idle"), ("total_timesteps", "—"), ("ep_rew_mean", "—"),
             ("ep_len_mean", "—"), ("fps", "—"), ("time_elapsed", "—")]}
        line = ttk.Frame(train_box)
        line.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(line, textvariable=self.stat_vars["status"], font=MONO_BOLD, width=9,
                  anchor="w").pack(side="left", padx=(0, 8))
        for key, label in (("ep_rew_mean", "reward"), ("fps", "fps"), ("time_elapsed", "elapsed")):
            ttk.Label(line, text=f"{label}:", foreground=THEME.muted).pack(side="left")
            ttk.Label(line, textvariable=self.stat_vars[key], font=MONO_BOLD, width=8,
                      anchor="w").pack(side="left", padx=(3, 6))
        ttk.Progressbar(train_box, mode="determinate", maximum=100,
                        variable=self.train_progress_var).grid(row=1, column=0, sticky="ew",
                                                               pady=(2, 0))
        ttk.Label(train_box, textvariable=self.train_progress_text, font=MONO, width=28,
                  anchor="e").grid(row=1, column=1, sticky="e", padx=(6, 0), pady=(2, 0))

        # Play controls.
        ctl = ttk.LabelFrame(parent, text="Play a model", padding=6)
        ctl.pack(fill="x")
        ctl.columnconfigure(1, weight=1)
        ctl.columnconfigure(3, weight=1)
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(ctl, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_selected())
        # Re-scan the model files whenever the list is opened; no Refresh button needed.
        self.model_combo.bind("<Button-1>", lambda e: self.refresh_models(), add="+")
        self.model_note_var = tk.StringVar(value="")
        ttk.Label(ctl, textvariable=self.model_note_var, foreground=THEME.muted,
                  wraplength=CANVAS_W - 20).grid(row=1, column=0, columnspan=4, sticky="w",
                                                 pady=(2, 2))
        self.play_episodes_var = tk.IntVar(value=3)
        self.play_speed_label_var = tk.StringVar(value=SPEED_CHOICES[1][0])
        ttk.Label(ctl, text="Episodes:").grid(row=2, column=0, sticky="w")
        ep_spin = ttk.Spinbox(ctl, from_=1, to=100, textvariable=self.play_episodes_var, width=5)
        ep_spin.grid(row=2, column=1, sticky="w", padx=(4, 12))
        ttk.Label(ctl, text="Speed:").grid(row=2, column=2, sticky="w")
        speed_combo = ttk.Combobox(ctl, textvariable=self.play_speed_label_var, state="readonly",
                                   values=[c[0] for c in SPEED_CHOICES], width=13)
        speed_combo.grid(row=2, column=3, sticky="w", padx=(4, 0))

        btns = ttk.Frame(ctl)
        btns.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.btn_play_start = ttk.Button(btns, text="▶ Play", command=self.start_playing)
        self.btn_play_start.pack(side="left")
        self.btn_play_stop = ttk.Button(btns, text="■ Stop", command=self.stop_playing,
                                        state="disabled")
        self.btn_play_stop.pack(side="left", padx=4)
        self.btn_screen_clear = ttk.Button(btns, text="Clear", command=self.clear_screen)
        self.btn_screen_clear.pack(side="left", padx=4)
        self.btn_play_adv = ttk.Button(btns, text="Advanced…", command=self._open_play_advanced)
        self.btn_play_adv.pack(side="right")

        # Advanced options live in a small dialog (see _open_play_advanced). The
        # observation setup is normally filled in from the model's run.json.
        self.play_max_steps_var = tk.IntVar(value=0)
        self.play_stochastic_var = tk.BooleanVar(value=False)
        self.play_action_repeat_var = tk.IntVar(value=4)
        self.play_frame_stack_var = tk.IntVar(value=4)
        self.play_obs_type_var = tk.StringVar(value="tiles")
        self.play_level_var = tk.StringVar(value="default")
        self.play_marathon_demo_var = tk.BooleanVar(value=True)
        # Attempt limits follow the model's run.json (see sync_play_options).
        self.play_time_budget = presets.PRESET_DEFAULTS["time_budget"]
        self.play_stall_steps = presets.PRESET_DEFAULTS["stall_steps"]
        self._play_adv_win: tk.Toplevel | None = None
        self._play_widgets: list[tk.Widget] = [self.model_combo, ep_spin, speed_combo,
                                               self.btn_play_start, self.btn_play_adv]

    def clear_screen(self) -> None:
        """Blank the emulator view and the live-episode panel (stops playback first)."""
        if self.playing_active():
            self.play_stop.set()
        if self.preview_thread is not None and self.preview_thread.is_alive():
            self.preview_stop.set()
        self.intro_stop.set()
        self.frames.take()
        self._init_canvas(self.canvas)
        for v in self.play_stat_vars.values():
            v.set("—")
        self.play_status_var.set("idle")

    def _open_play_advanced(self) -> None:
        """Rarely needed playback options, in a small window next to the screen."""
        if self._play_adv_win is not None and self._play_adv_win.winfo_exists():
            self._play_adv_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Advanced playback options")
        win.resizable(False, False)
        win.transient(self.root)
        self._play_adv_win = win
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        r = 0

        def row(label, widget):
            nonlocal r
            ttk.Label(body, text=label).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="w", padx=8, pady=2)
            r += 1

        row("Max steps per episode (0 = no cap):", ttk.Spinbox(
            body, from_=0, to=1_000_000, increment=100, textvariable=self.play_max_steps_var,
            width=8))
        ttk.Checkbutton(body, text="Stochastic actions (sample from the policy)",
                        variable=self.play_stochastic_var).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1
        ttk.Checkbutton(body, text="Marathon demo: after a death continue with the next level "
                                   "(shows every level; off = strict one-life marathon)",
                        variable=self.play_marathon_demo_var).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1
        ttk.Separator(body).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        ttk.Label(body, wraplength=420, foreground=THEME.muted,
                  text="Observation setup. Filled in automatically from the model's run.json; "
                       "only change it for models from older runs, and make it match how "
                       "the model was trained.").grid(row=r, column=0, columnspan=2, sticky="w",
                                                      pady=(0, 6))
        r += 1
        row("Obs type:", ttk.Combobox(body, textvariable=self.play_obs_type_var,
                                      values=list(OBS_TYPES), state="readonly", width=8))
        row("Start level:", ttk.Combobox(body, textvariable=self.play_level_var,
                                         values=LEVEL_CHOICES, state="readonly", width=10))
        row("Action repeat:", ttk.Spinbox(body, from_=1, to=16,
                                          textvariable=self.play_action_repeat_var, width=5))
        row("Frame stack:", ttk.Spinbox(body, from_=1, to=8,
                                        textvariable=self.play_frame_stack_var, width=5))
        ttk.Button(body, text="Close", command=win.destroy).grid(row=r, column=1, sticky="e",
                                                                 pady=(10, 0))
        win.update_idletasks()
        win.geometry(f"+{self.canvas.winfo_rootx()}+{self.root.winfo_rooty() + 80}")

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
        for c in (1, 3):
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
        _field(1, 0, "Seeds / config:", ttk.Spinbox(
            cfg_frame, from_=1, to=10, width=6,
            textvariable=self.tune_seeds_var, command=self.tune_update_summary))
        self.tune_evals_var = tk.IntVar(value=5)
        _field(1, 1, "Evals / trial:", ttk.Spinbox(
            cfg_frame, from_=1, to=50, width=6, textvariable=self.tune_evals_var))
        self.tune_run_prefix_var = tk.StringVar(value="tune")
        _field(2, 0, "Run-name prefix:", ttk.Entry(cfg_frame, textvariable=self.tune_run_prefix_var))
        self.tune_metric_var = tk.StringVar(value=tuning.METRIC_BEST)
        metric_combo = ttk.Combobox(cfg_frame, textvariable=self.tune_metric_var, state="readonly",
                                    width=24, values=list(tuning.METRICS))
        metric_combo.bind("<<ComboboxSelected>>", lambda e: self._on_metric_changed())
        _field(2, 1, "Metric:", metric_combo)
        self.tune_metric_note = ttk.Label(cfg_frame, text=tuning.METRIC_NOTES[tuning.METRIC_BEST],
                                          foreground=THEME.muted, wraplength=640)
        self.tune_metric_note.grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.tune_keep_best_var = tk.IntVar(value=9)
        _field(3, 0, "Keep best N trial runs:", ttk.Spinbox(
            cfg_frame, from_=0, to=100, width=6, textvariable=self.tune_keep_best_var))
        ttk.Label(cfg_frame, text="(0 = keep all; scores of deleted runs stay in the table)",
                  foreground=THEME.muted).grid(row=3, column=2, columnspan=2, sticky="w", padx=12)
        self.tune_skip_done_var = tk.BooleanVar(value=True)
        skip_cb = ttk.Checkbutton(
            cfg_frame, variable=self.tune_skip_done_var, command=self.tune_update_summary,
            text="Reuse results AIboy already knows for the same settings (also resumes a sweep)")
        skip_cb.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self._tune_config_widgets.append(skip_cb)
        self.tune_baseline_var = tk.BooleanVar(value=True)
        base_cb = ttk.Checkbutton(
            cfg_frame, variable=self.tune_baseline_var, command=self.tune_update_summary,
            text="Include the base preset unchanged as candidate 1 (a winner must beat it)")
        base_cb.grid(row=7, column=0, columnspan=4, sticky="w")
        self._tune_config_widgets.append(base_cb)


        sweep_frame = ttk.LabelFrame(parent, text="Sweep", padding=6)
        sweep_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        sweep_frame.columnconfigure(1, weight=1)
        ttk.Label(sweep_frame, text="Template:").grid(row=0, column=0, sticky="w")
        self.tune_template_var = tk.StringVar(value=tuning.DEFAULT_TEMPLATE)
        tmpl_combo = ttk.Combobox(sweep_frame, textvariable=self.tune_template_var,
                                  values=list(tuning.SWEEP_TEMPLATES), state="readonly", width=30)
        tmpl_combo.grid(row=0, column=1, sticky="ew", padx=6)
        tmpl_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_tune_template())
        self.btn_tune_template = ttk.Button(sweep_frame, text="Reset to template",
                                            command=self.apply_tune_template)
        self.btn_tune_template.grid(row=0, column=2)
        self.btn_tune_suggest = ttk.Button(sweep_frame, text="Suggest from experience",
                                           command=self.tune_suggest)
        self.btn_tune_suggest.grid(row=0, column=3, padx=(4, 0))
        self._tune_config_widgets.extend([tmpl_combo, self.btn_tune_template, self.btn_tune_suggest])
        self.tune_template_note = ttk.Label(sweep_frame, text="", foreground=THEME.muted,
                                            wraplength=620)
        self.tune_template_note.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 4))

        self.tune_sweep_text = tk.Text(sweep_frame, height=5, wrap="word", font=MONO, undo=True)
        self.tune_sweep_text.grid(row=2, column=0, columnspan=4, sticky="ew")
        self.tune_sweep_text.bind("<<Modified>>", self._on_sweep_modified)
        self._tune_config_widgets.append(self.tune_sweep_text)

        search_row = ttk.Frame(sweep_frame)
        search_row.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
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
        self.tune_summary_label = ttk.Label(sweep_frame, textvariable=self.tune_summary_var,
                                            font=MONO, wraplength=640)
        self.tune_summary_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
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
        ttk.Label(ctrl, textvariable=self.tune_progress_text, width=20, anchor="e").grid(
            row=0, column=4, sticky="e")
        self.tune_live_var = tk.StringVar(value="idle")
        ttk.Label(parent, textvariable=self.tune_live_var, font=MONO).grid(
            row=3, column=0, sticky="w", pady=(0, 4))

        res_frame = ttk.LabelFrame(parent, text="Results (best first)", padding=4)
        res_frame.grid(row=4, column=0, sticky="nsew")
        self.tune_tree = make_table(res_frame, [
            ("rank", "#", 30, "e", False), ("config", "config (overrides)", 190, "w", True),
            ("score", "score", 90, "e", False), ("ep_len", "ep len", 52, "e", False),
            ("steps", "steps", 66, "e", False), ("time", "time", 52, "e", False),
            ("runs", "runs", 110, "w", True),
        ], height=7)
        self.tune_tree.bind("<Double-1>", lambda e: self._tune_load_into_train())

        actions = ttk.Frame(parent)
        actions.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.btn_tune_save_preset = ttk.Button(actions, text="Save selected as preset",
                                               command=self._tune_save_as_preset)
        self.btn_tune_save_preset.pack(side="left", padx=(0, 4))
        self.btn_tune_to_train = ttk.Button(actions, text="Load selected into Train tab",
                                            command=self._tune_load_into_train)
        self.btn_tune_to_train.pack(side="left", padx=4)
        ttk.Label(actions, text="(double-click a row to load it)", foreground=THEME.muted).pack(
            side="left", padx=8)
        self.btn_tune_delete = ttk.Button(actions, text="Delete trial runs…",
                                          command=lambda: self.delete_tune_data(
                                              self.tune_run_prefix_var.get().strip() or "tune"))
        self.btn_tune_delete.pack(side="right")

        self.apply_tune_template()

    def _on_metric_changed(self) -> None:
        """Re-rank the current results under the newly chosen metric."""
        metric = self.tune_metric_var.get()
        self.tune_metric_note.config(text=tuning.METRIC_NOTES.get(metric, ""))
        self.tune_tree.heading("score", text=metric)
        self._rescore_results(metric)

    def _rescore_results(self, metric: str) -> None:
        """Recompute every config's score under `metric` from the eval
        histories stored with the results (no run folder needed). Results
        loaded from an old file without histories keep their numbers."""
        stale = 0
        for res in self._tune_results.values():
            if res.records:
                res.rescore(metric)
            else:
                stale += 1
        self.render_tune_results()
        if stale:
            self.tune_live_var.set(f"{stale} result(s) come from an older file and keep their "
                                   f"original metric")
        if self.wizard is not None:
            self.wizard.on_tune_result()

    # ---------- Tuning: sweep editing ----------

    def apply_tune_template(self) -> None:
        name = self.tune_template_var.get()
        sweep = tuning.SWEEP_TEMPLATES.get(name)
        if sweep is None:
            return
        self.set_sweep(sweep, tuning.TEMPLATE_NOTES.get(name, ""))

    def set_sweep(self, sweep, note: str) -> None:
        """Put a grid or a candidate list into the sweep editor."""
        text = json.dumps(sweep, indent=2) if isinstance(sweep, dict) else json.dumps(sweep)
        if self.tune_sweep_text.get("1.0", "end").strip() != text:
            self.tune_sweep_text.delete("1.0", "end")
            self.tune_sweep_text.insert("1.0", text)
        self.tune_sweep_text.edit_modified(False)
        self.tune_template_note.config(text=note)
        self.tune_update_summary()

    def tune_suggest(self) -> None:
        """Fill the sweep with untested variations around the best known
        settings of the base preset (or around the preset itself)."""
        plan, err = self.tune_plan()
        base = self._all_presets.get(self.tune_preset_var.get())
        if base is None:
            messagebox.showwarning("Tune", "Pick a base preset first.")
            return
        try:
            trial_steps = int(self.tune_trial_steps_var.get())
            n_seeds = max(1, int(self.tune_seeds_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showwarning("Tune", "Steps / trial and seeds must be whole numbers.")
            return
        know = self.experience.best_for_task(tuning.trial_config(base, {}, trial_steps))
        combos = self.experience.suggest(base, trial_steps, n_seeds, 6, tuning.METRIC_LATE, know)
        if not combos:
            self.flash("Nothing left to suggest: every nearby variation is already known.")
            return
        if know is not None:
            around = tuning.describe_overrides(
                tuning.config_diff(know.config, tuning.full_config(base, {}))) or "the preset itself"
            note = (f"{len(combos)} untested variations around the best known settings "
                    f"({around}, {know.n_trials} trials remembered, score {know.score:.0f}).")
        else:
            note = (f"No earlier trials of this task: {len(combos)} untested variations around "
                    f"the preset's own settings.")
        self.set_sweep(combos, note)

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
            keep_best = max(0, int(self.tune_keep_best_var.get()))
        except (tk.TclError, ValueError):
            return None, "steps / seeds / evals / N must be whole numbers"
        if trial_steps < 2000:
            return None, "steps per trial must be at least 2000"
        search = self.tune_search_type_var.get()
        combos = (tuning.expand_grid(sweep) if search == "grid"
                  else tuning.sample_random(sweep, n_random))
        if self.tune_baseline_var.get():
            combos = tuning.with_baseline(combos)
        return {
            "preset_name": preset_name, "base": dict(base), "sweep": sweep,
            "game": str(base.get("game", self.game)),
            "search": search, "combos": combos, "trial_steps": trial_steps,
            "n_seeds": n_seeds, "evals_per_trial": evals,
            "prefix": self.tune_run_prefix_var.get().strip() or "tune",
            "metric": self.tune_metric_var.get(),
            "skip_done": bool(self.tune_skip_done_var.get()),
            "keep_best": keep_best,
        }, None

    def tune_update_summary(self, *_args) -> None:
        if not hasattr(self, "tune_summary_label"):
            return  # widget callback during construction
        plan, err = self.tune_plan()
        if err:
            self.tune_summary_var.set(f"⚠ {err}")
            self.tune_summary_label.config(foreground=THEME.err)
        else:
            n_cfg, n_seeds = len(plan["combos"]), plan["n_seeds"]
            n_trials = n_cfg * n_seeds
            known = self.known_trials(plan)
            fps = self.fps_for(tuning.trial_config(plan["base"], {}, plan["trial_steps"]))
            eta = tuning.estimate_seconds(n_trials - known, plan["trial_steps"], plan["base"], fps)
            seeds_txt = f" × {n_seeds} seeds" if n_seeds > 1 else ""
            base_txt = " (incl. base preset)" if self.tune_baseline_var.get() else ""
            known_txt = f", {known} already known" if known else ""
            self.tune_summary_var.set(
                f"{n_cfg} configs{base_txt}{seeds_txt} = {n_trials} trials{known_txt} · "
                f"≈ {tuning.format_duration(eta)}" + (" (measured speed)" if fps else ""))
            self.tune_summary_label.config(foreground=THEME.text_soft)
        if self.wizard is not None:
            self.wizard.on_tune_plan_changed()

    # ---------- Tuning: execution ----------

    def start_tuning(self, source: str = "tune") -> bool:
        """Start the sweep described by the Tune tab. Returns True if it
        started. `source` (tune | wizard) is recorded with every trial."""
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
        plan["source"] = source
        n_trials = len(plan["combos"]) * plan["n_seeds"]
        to_train = n_trials - self.known_trials(plan)
        if to_train > 40 and not messagebox.askyesno(
                "Tune", f"This sweep trains {to_train} trials (≈ "
                        f"{tuning.format_duration(tuning.estimate_seconds(to_train, plan['trial_steps'], plan['base'], self.fps_for(plan['base'])))}). "
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
                   done_trials: int, total: int, eta: tuning.SweepEta) -> int | None:
        """Run one training subprocess, streaming its SB3 stats into the
        live-status line and the progress bar. Returns the exit code."""
        eta.trial_started(time.time())
        try:
            self.tune_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJECT_DIR),
            )
        except OSError as e:
            self.stats_queue.put(("tune_status", f"{label}: could not start: {e}"))
            return None
        runs.keep_awake(self.tune_proc.pid)
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
            self.stats_queue.put(("tune_progress", frac, f"{done_trials}/{total}",
                                  eta.report(time.time(), steps)))
            self.stats_queue.put(("tune_live",
                f"{label} · {steps:,}/{trial_steps:,} steps · "
                f"ep_rew_mean {stats.get('ep_rew_mean', '—')} · "
                f"ep_len_mean {stats.get('ep_len_mean', '—')} · {stats.get('fps', '—')} fps"))
        rc = self.tune_proc.wait()
        self.tune_proc = None
        eta.trial_finished(time.time())
        return rc

    def _tuning_loop(self, plan: dict) -> None:
        """Background thread. Whatever happens, `tune_done` is emitted so
        the tabs unlock; an unexpected error is reported in the log."""
        try:
            self._run_sweep(plan)
        except Exception:
            import traceback
            self.stats_queue.put(("tune_status", f"sweep aborted:\n{traceback.format_exc()}"))
            self.stats_queue.put(("tune_done", "aborted by an error — see the Train tab log", True))

    def _run_sweep(self, plan: dict) -> None:
        base, combos = plan["base"], plan["combos"]
        trial_steps, n_seeds = plan["trial_steps"], plan["n_seeds"]
        prefix, metric = plan["prefix"], plan["metric"]
        game = plan["game"]          # resolved on the main thread; no Tk access here
        source = plan.get("source", "tune")
        eval_freq = max(1000, trial_steps // plan["evals_per_trial"])
        total = len(combos) * n_seeds
        done_trials = 0
        results: list[tuning.ConfigResult] = []
        out_path = tuning.results_file(PROJECT_DIR, game, prefix)
        meta = {"game": game, "base_preset": plan["preset_name"], "metric": metric,
                "trial_steps": trial_steps, "seeds": n_seeds, "search": plan["search"],
                "sweep": plan["sweep"], "source": source}
        cancelled = False
        exp = self.experience
        exp.reload()

        # Which planned trials does AIboy already know? Those are scored from
        # memory (instant), so the ETA must not count them as work.
        def _known(i: int, s: int) -> "Record | None":
            if not plan["skip_done"]:
                return None
            return exp.find(tuning.trial_config(base, combos[i - 1], trial_steps, s))

        to_train = sum(0 if _known(i, s) else 1
                       for i in range(1, len(combos) + 1) for s in range(n_seeds))
        eta = tuning.SweepEta(to_train, trial_steps)
        if total - to_train:
            self.stats_queue.put(("tune_status", f"{total - to_train} of {total} trials are already "
                                                 f"known from earlier searches; not training them again"))

        for i, overrides in enumerate(combos, start=1):
            cfg = tuning.full_config(base, overrides)
            res = tuning.ConfigResult(index=i, overrides=overrides, config=cfg, metric=metric)
            t0 = time.time()
            for s in range(n_seeds):
                if self.tune_stop.is_set():
                    cancelled = True
                    break
                run_name = tuning.trial_run_name(prefix, i, s, n_seeds)
                seed_cfg = tuning.trial_config(base, overrides, trial_steps, s)
                label = f"config {i}/{len(combos)}" + (f" seed {s + 1}/{n_seeds}" if n_seeds > 1 else "")
                self.stats_queue.put(("tune_progress", done_trials / total, f"{done_trials}/{total}",
                                      eta.between_trials()))
                known = _known(i, s)
                if known is not None:
                    res.reused += 1
                    res.add_record(known.to_dict())
                    self.stats_queue.put(("tune_status", f"{label}: known from "
                                                         f"{known.source} run '{known.run_name}' "
                                                         f"({time.strftime('%Y-%m-%d', time.localtime(known.created_at))})"))
                else:
                    cmd = runs.build_train_cmd(seed_cfg, run_name, timesteps=trial_steps,
                                               checkpoint_freq=trial_steps, eval_freq=eval_freq,
                                               source=source)
                    self.stats_queue.put(("tune_status", f"{label}: {res.label}  →  {run_name}"))
                    started = time.time()
                    rc = self._run_trial(cmd, label, trial_steps, done_trials, total, eta)
                    if self.tune_stop.is_set():
                        cancelled = True
                        break
                    if rc not in (0, None):
                        self.stats_queue.put(("tune_status", f"{label}: trainer exited with rc={rc}"))
                    # Scoring needs only logs/ (eval history, best model).
                    runs.slim_trial_run(game, run_name)
                    exp.reload()
                    record = exp.latest_for_run(run_name, since=started - 1.0)
                    if record is None:
                        self.stats_queue.put(("tune_status", f"{label}: no result was recorded for "
                                                             f"{run_name} (see the log above)"))
                    else:
                        res.add_record(record.to_dict())
                        if not record.complete:
                            self.stats_queue.put(("tune_status", f"{label}: {run_name} did not finish; "
                                                                 f"its partial score is shown but "
                                                                 f"will not be reused"))
                    res.runs.append(run_name)
                done_trials += 1
            res.duration = time.time() - t0
            if res.records or res.runs:
                results.append(res)
                # Keep only the best `keep_best` configs on disk; scores stay.
                deleted = tuning.prune_trial_runs(results, plan["keep_best"], game, runs.delete_run)
                if deleted:
                    self.stats_queue.put(("tune_status",
                                          f"deleted {len(deleted)} trial run(s) outside the best "
                                          f"{plan['keep_best']}: {', '.join(deleted)}"))
                self.stats_queue.put(("tune_result", res))
                try:
                    tuning.save_results(out_path, meta, results)
                except OSError as e:
                    self.stats_queue.put(("tune_status", f"could not save results: {e}"))
            if cancelled:
                break
        self.stats_queue.put(("tune_progress", done_trials / max(1, total), f"{done_trials}/{total}",
                              None))
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
            run_txt = ", ".join(r.runs) + (f"  ({r.reused} known)" if r.reused else "")
            if r.pruned:
                run_txt += "  (deleted)"
            self.tune_tree.insert(
                "", "end", iid=str(r.index),
                values=(rank, r.label, r.score_text(),
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
        if meta.get("metric") in tuning.METRICS:
            self.tune_metric_var.set(meta["metric"])
            self.tune_tree.heading("score", text=meta["metric"])
            self.tune_metric_note.config(text=tuning.METRIC_NOTES[meta["metric"]])
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
            "Save tune result as preset", f"Name for this preset ({res.label}):", parent=self.root)
        if not self.save_preset_named((name or "").strip(), res.config):
            return
        self.flash(f"Preset '{name}' saved and selected on the Train tab.")

    def _tune_load_into_train(self) -> None:
        res = self._selected_tune_result()
        if res is None or self.busy():
            return
        self.load_config_into_train(res.config)
        self.preset_var.set("")  # custom config
        self.show_tab(self.train_tab)
        self.flash(f"Config {res.index} ({res.label}) loaded — set a run name and Start.")

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
        self._on_train_form_changed()

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
        self.play_time_budget = int(cfg.get("time_budget", presets.PRESET_DEFAULTS["time_budget"]))
        self.play_stall_steps = int(cfg.get("stall_steps", presets.PRESET_DEFAULTS["stall_steps"]))

    def apply_preset(self) -> None:
        cfg = self._all_presets.get(self.preset_var.get())
        if cfg is not None:
            self.load_config_into_train(cfg)
        self._on_train_form_changed()

    def _on_train_form_changed(self) -> None:
        """Show whether the Train tab still matches its selected preset."""
        if not hasattr(self, "preset_state_var"):
            return
        name = self.preset_var.get()
        base = self._all_presets.get(name)
        if base is None:
            self.preset_state_var.set("custom configuration (not a saved preset)" if name == ""
                                      else "")
            return
        try:
            current = self.current_config()
        except ValueError:
            self.preset_state_var.set("⚠ a field holds an invalid value")
            return
        base_full = presets.normalize({**base, "game": self.game})
        if all(str(current.get(k)) == str(v) for k, v in base_full.items()):
            self.preset_state_var.set("")
        else:
            self.preset_state_var.set("modified — use Save as… to keep these values as a preset")

    def _on_run_name_changed(self) -> None:
        self.refresh_models()
        self._update_run_hint()

    def _update_run_hint(self) -> None:
        name = self.run_name_var.get().strip() or "default"
        if not runs.is_run_name(name):
            self.run_hint_var.set("⚠ run names may not start with '_'")
            return
        paths = runs.run_paths(self.game, name)
        best = runs.best_model_for_run(self.game, name)
        if best is not None:
            n_ckpt = len(list(paths["checkpoints"].glob("*.zip")))
            self.run_hint_var.set(f"{paths['base']}/ exists ({n_ckpt} checkpoints): tick Resume "
                                  f"to continue it, or choose a new name")
        else:
            self.run_hint_var.set(f"{paths['base']}/ (new run)")

    def current_config(self) -> dict:
        """Train-tab values as a preset dict. Raises ValueError naming the
        field if a value is not a number."""
        return {"game": self.game, **self.form.get_config()}

    def save_preset_named(self, name: str, cfg: dict, *, confirm_overwrite: bool = True) -> bool:
        """Persist `cfg` as user preset `name` and select it on the Train tab."""
        if not name:
            return False
        if confirm_overwrite and presets.is_builtin(name) and not messagebox.askyesno(
                "Save preset",
                f"'{name}' is a built-in preset. Save your values as its new version?\n"
                f"(The shipped values stay available via 'Reset to default' on the Presets tab.)"):
            return False
        if confirm_overwrite and not presets.is_builtin(name) and presets.is_user(name) \
                and not messagebox.askyesno(
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
        self._refresh_disk_usage()

    def _refresh_disk_usage(self) -> None:
        """Size of models/ in the status bar, measured off the UI thread.
        Everything Tk-related is read here, on the main thread."""
        root, game = runs.MODELS_ROOT, self.game

        def _measure() -> None:
            self.stats_queue.put(("disk_usage", runs.dir_size(root), len(runs.list_runs(game))))

        threading.Thread(target=_measure, daemon=True).start()

    # ---------- Housekeeping ----------

    def _open_run_folder(self) -> None:
        name = self.run_name_var.get().strip() or "default"
        base = runs.run_paths(self.game, name)["base"]
        target = base if base.exists() else base.parent
        target.mkdir(parents=True, exist_ok=True)
        try:
            runs.open_in_file_manager(PROJECT_DIR / target)
        except OSError as e:
            messagebox.showerror("Open folder", f"Could not open {target}:\n{e}")

    def _compact_run(self) -> None:
        """Drop redundant step snapshots of the run named on the Train tab."""
        if self.busy():
            messagebox.showwarning("Compact run", "Stop training or tuning first.")
            return
        name = self.run_name_var.get().strip() or "default"
        paths = runs.run_paths(self.game, name)
        if not runs.is_run_name(name) or not paths["base"].exists():
            messagebox.showinfo("Compact run", f"There is no run '{name}'.")
            return
        has_final = (paths["checkpoints"] / "final.zip").exists()
        doomed = runs.redundant_checkpoints(self.game, name)
        if not doomed:
            messagebox.showinfo("Compact run", f"'{name}' is already compact.")
            return
        size = runs.format_size(sum(p.stat().st_size for p in doomed))
        what = ("all step snapshots (final.zip holds the last state)" if has_final
                else f"all but the newest {runs.KEEP_CHECKPOINTS} step snapshots")
        if not messagebox.askyesno(
                "Compact run",
                f"Remove {len(doomed)} file(s), {size}, from '{name}'?\n\nThis deletes {what}. "
                f"The best model, final model, eval history and TensorBoard logs stay."):
            return
        freed = runs.compact_run(self.game, name)
        self.refresh_models()
        self._update_run_hint()
        self.flash(f"Run '{name}' compacted ({runs.format_size(freed)} freed).")

    def _delete_run(self) -> None:
        """Delete the run named on the Train tab (models/<game>/<run>/)."""
        if self.busy():
            messagebox.showwarning("Delete run", "Stop training or tuning first.")
            return
        name = self.run_name_var.get().strip() or "default"
        base = runs.run_paths(self.game, name)["base"]
        if not runs.is_run_name(name) or not base.exists():
            messagebox.showinfo("Delete run", f"There is no run '{name}' to delete.")
            return
        size = runs.format_size(runs.dir_size(base))
        if not messagebox.askyesno(
                "Delete run",
                f"Delete run '{name}' ({size})?\n\nThis removes {base}/ with all its "
                f"checkpoints, the best model and TensorBoard logs. It cannot be undone."):
            return
        if self.playing_active():
            self.play_stop.set()
        try:
            freed = runs.delete_run(self.game, name)
        except (OSError, ValueError) as e:
            messagebox.showerror("Delete run", str(e))
            return
        self.run_name_var.set("")
        self.refresh_run_names()
        self.refresh_models()
        self._update_run_hint()
        self.flash(f"Run '{name}' deleted ({runs.format_size(freed)} freed).")

    def delete_tune_data(self, prefix: str) -> bool:
        """Delete every trial run of a sweep prefix and its results file.
        Used by the Tune tab and the wizard. True if something was deleted."""
        if self.busy():
            messagebox.showwarning("Delete trial runs", "Stop training or tuning first.")
            return False
        n_runs, size = runs.tune_data_size(self.game, prefix)
        if n_runs == 0 and size == 0:
            messagebox.showinfo("Delete trial runs", f"No trial runs with prefix '{prefix}' found.")
            return False
        if not messagebox.askyesno(
                "Delete trial runs",
                f"Delete {n_runs} trial run(s) named '{prefix}-…' and the saved results "
                f"({runs.format_size(size)})?\n\nThe presets you saved from them are kept. "
                f"This cannot be undone."):
            return False
        try:
            n_runs, freed = runs.delete_tune_data(self.game, prefix)
        except (OSError, ValueError) as e:
            messagebox.showerror("Delete trial runs", str(e))
            return False
        if self.tune_run_prefix_var.get().strip() == prefix:
            self._tune_results.clear()
            self.render_tune_results()
            self.tune_live_var.set("idle")
            self.tune_progress_var.set(0)
            self.tune_progress_text.set("—")
        self.refresh_run_names()
        self.refresh_models()
        self.flash(f"Deleted {n_runs} trial run(s) of '{prefix}' ({runs.format_size(freed)} freed).")
        return True

    def refresh_models(self) -> None:
        self._model_paths = dict(runs.list_models(self.game))
        labels = list(self._model_paths)
        prev = self.model_var.get()
        self.model_combo["values"] = labels
        if prev in labels:
            self.model_var.set(prev)
        else:
            self.model_var.set(labels[0] if labels else "")
            self._on_model_selected()
        if hasattr(self, "run_hint_var"):
            self._update_run_hint()

    def _on_model_selected(self) -> None:
        """Apply the selected model's recorded training settings to the Play
        options, so playback always uses the observation setup it needs."""
        if not hasattr(self, "model_note_var"):
            return
        path = self._model_paths.get(self.model_var.get())
        if path is None:
            self.model_note_var.set("")
            return
        game_run = runs.run_of_model(path)
        cfg = runs.read_run_config(*game_run) if game_run else None
        if cfg:
            self.sync_play_options(cfg)
            self.model_note_var.set(
                f"run '{game_run[1]}': {cfg.get('obs_type')} · {cfg.get('start_level')} · "
                f"repeat {cfg.get('action_repeat')} · stack {cfg.get('frame_stack')}")
        else:
            self.model_note_var.set("older run without run.json — check settings under Advanced…")

    def select_model(self, path: Path) -> bool:
        """Select the Play-tab model whose file is `path`. False if unknown."""
        for label, p in self._model_paths.items():
            if p == path:
                self.model_var.set(label)
                self._on_model_selected()
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
        cmd = runs.build_train_cmd(cfg, run_name, resume=bool(self.resume_var.get()),
                                   source="train")

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
        if runs.keep_awake(self.train_proc.pid) is not None:
            self.append_log("[gui] sleep prevented while training (caffeinate)\n")
        self.btn_train_start.config(state="disabled")
        self.btn_train_stop.config(state="normal")
        self.btn_tune_start.config(state="disabled")
        self.set_inputs_disabled(True)
        self.stat_vars["status"].set("running")
        for k in TRACKED_STATS[1:]:
            self.stat_vars[k].set("—")
        self._train_target_steps = max(1, cfg["timesteps"])
        # Baseline = the model's prior step count (0 unless resuming). The
        # first reported total_timesteps is baseline + one rollout.
        self._train_rollout_size = max(1, cfg["n_steps"] * cfg["n_envs"])
        self._train_baseline_steps = None
        self._train_rate.reset()
        self.train_progress_var.set(0)
        self.train_progress_text.set(tuning.progress_text(0, self._train_target_steps, None))
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
        self.intro_stop.set()
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

        self.intro_stop.set()
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
            time_budget=self.play_time_budget, stall_steps=self.play_stall_steps,
            marathon_demo=bool(self.play_marathon_demo_var.get()),
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

        latest = self.frames.take()
        if latest is not None and self._canvas_img_id is not None:
            img = Image.fromarray(latest).resize((CANVAS_W, CANVAS_H), Image.NEAREST)
            self._tk_img = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

        self._update_status_bar()
        try:
            self.root.after(33, self._pump)
        except tk.TclError:
            pass

    def flash(self, text: str, seconds: float = FLASH_SECONDS) -> None:
        """Show a transient message in the status bar (instead of a modal dialog)."""
        self._flash = (text, time.time() + seconds)
        self.status_bar_var.set(text)

    def _update_status_bar(self) -> None:
        if self._flash is not None:
            if time.time() < self._flash[1]:
                return
            self._flash = None
        if self.training_active():
            text = (f"Training '{self._train_run_name}' · {self.train_progress_text.get()} · "
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
        elif kind == "intro_done":
            # Show the idle frame unless playback / preview took the screen.
            if not self.playing_active() and not (
                    self.preview_thread is not None and self.preview_thread.is_alive()):
                self.frames.take()
                self._init_canvas(self.canvas)
        elif kind == "disk_usage":
            _, n_bytes, n_runs = item
            self.disk_var.set(f"models/ {runs.format_size(n_bytes)} · {n_runs} run(s)")
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
        elif kind == "play_stats":
            for key, val in item[1].items():
                if key in self.play_stat_vars:
                    self.play_stat_vars[key].set(val)
        elif kind == "play_done":
            # Keep the last episode's report ("Episode 2: … died in 1-2") visible.
            last = self.play_status_var.get()
            self.play_status_var.set(f"done · {last}" if last.startswith("Episode") else "done")
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
            _, frac, label, eta = item
            self.tune_progress_var.set(100.0 * frac)
            self.tune_progress_text.set(label + (tuning.format_eta(eta) if frac < 1.0 else ""))
        elif kind == "tune_result":
            r = item[1]
            self._tune_results[r.index] = r
            self.render_tune_results()
            self.append_log(f"[tune] config {r.index} ({r.label}) → "
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
            self.on_experience_changed()    # the trials were recorded; refresh notes and estimates

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
        # Rate over recent rollouts only: start-up is excluded automatically
        # because the first sample is the first report, not the launch.
        self._train_rate.add(time.time(), new_steps)
        eta = self._train_rate.eta(self._train_target_steps - new_steps)
        self.train_progress_text.set(tuning.progress_text(new_steps, self._train_target_steps, eta))

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
        self.on_experience_changed()    # the trainer recorded the run
        best = runs.best_model_for_run(self.game, self._train_run_name)
        if best is not None and self.select_model(best):
            self.flash(f"Run '{self._train_run_name}' {status} — its best model is selected on "
                       f"the screen panel, click ▶ Play to watch it.", seconds=12)
        if self.wizard is not None:
            self.wizard.on_train_done(rc, self._train_stop_requested)

    def _play_finished(self) -> None:
        self.btn_play_stop.config(state="disabled")
        self._update_start_buttons()

    def set_inputs_disabled(self, disabled: bool) -> None:
        """Lock every input on every tab while training or tuning runs, so a
        config cannot change mid-run and a second CPU-hungry session cannot
        be started. Start / Stop buttons are managed by their own callers."""
        widgets: list[tk.Widget] = [
            self.game_combo, self.btn_rescan,
            self.preset_combo, self.btn_preset_save, self.run_name_combo,
            self.resume_check, self.preview_check, self.btn_run_delete, self.btn_run_compact,
            *self._play_widgets, self.btn_screen_clear,
            *self._tune_config_widgets,
            self.btn_tune_save_preset, self.btn_tune_to_train, self.btn_tune_load,
            self.btn_tune_delete,
        ]
        self._input_lock.apply(widgets, disabled)
        self.form.set_enabled(not disabled)
        if self.wizard is not None:
            self.wizard.set_inputs_disabled(disabled)
        if self.presets_tab is not None:
            self.presets_tab.set_inputs_disabled(disabled)
        if self.experience_tab is not None:
            self.experience_tab.set_inputs_disabled(disabled)

    def open_tensorboard(self) -> None:
        """Serve models/<game>/ (every run, incl. tune trials) and open a browser tab."""
        url = f"http://localhost:{TENSORBOARD_PORT}"
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            webbrowser.open(url)
            return
        logdir = PROJECT_DIR / "models" / self.game
        exe = shutil.which("tensorboard")
        if exe is None and FROZEN:
            messagebox.showinfo("TensorBoard",
                                "TensorBoard is not installed on this computer. Install it with "
                                "`pip install tensorboard` and click again, or read the training "
                                f"curves later from {logdir}.")
            return
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


ERROR_LOG = PROJECT_DIR / "gui_errors.log"


def _report_callback_exception(exc_type, exc, tb) -> None:
    """Unhandled error in a Tk callback: keep a trace on disk and tell the
    user, instead of printing to a terminal nobody is watching."""
    import traceback
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        with ERROR_LOG.open("a") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}")
    except OSError:
        pass
    sys.stderr.write(text)
    try:
        messagebox.showerror("Unexpected error",
                             f"{exc_type.__name__}: {exc}\n\nDetails were written to {ERROR_LOG.name}. "
                             f"Running training or tuning continues.")
    except tk.TclError:
        pass


def run() -> None:
    os.chdir(PROJECT_DIR)
    root = tk.Tk()
    root.report_callback_exception = _report_callback_exception
    AIboyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
