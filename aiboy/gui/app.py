"""The AIboy window.

Tabs, in workflow order:
  Wizard      guided Search -> Save -> Train -> Watch flow (see wizard.py)
  Train       a single headless training run (subprocess) with live stats
  Tune        hyperparameter sweeps over short training trials
  Presets     browse / edit / organise presets (see presets_tab.py)
  Experience  everything AIboy has tried so far (see experience_tab.py)
then the Preview (a photo of a Game Boy with the emulator on its LCD and
its buttons lit per action, see gameboy.py) and the Tracking panel to its
right: the live game, the training run's progress, the controls to play any
saved model, and "Play yourself" (the plain game with the keyboard, the
mouse or a game controller, see controls.py).

Training and tuning run `main.py train` as subprocesses so the window stays
responsive; playback runs on a background thread (player.py). All
cross-thread traffic goes through two queues drained by `_pump()`. Every
finished trainer process appends to the experience file, which the app
re-reads to reuse known results, refresh its estimates and fill the
Experience tab.
"""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image

from aiboy import APP_NAME, presets, runs, settings, tuning
from aiboy.experience import Experience, Record
from aiboy.games import (GAMES, OBS_TYPES, SML_ALL_LEVELS, RomInfo, check_start_level,
                         discover_roms, display_name, level_choices, probe_rom)
from aiboy.gui.agent_view import AgentView, legend as agent_view_legend
from aiboy.gui.controls import ControlMap, Gamepad, HeldButtons, KeyboardInput
from aiboy.gui.controls_dialog import ControlsDialog
from aiboy.gui.experience_tab import ExperienceTab
from aiboy.gui.gameboy import LCD_SCALES, GameBoyView, dmg_tint, photo_size
from aiboy.gui.player import GAME_H, GAME_W, SPEED_CHOICES, EmbeddedPlayer, IntroVideo, LatestFrame
from aiboy.gui.presets_tab import PresetsTab
from aiboy.paths import DATA_DIR, app_command
from aiboy.gui.widgets import (MONO, MONO_BOLD, THEME, ConfigForm, WidgetLock, info_icon, make_table,
                               setup_styles, tooltip)

PROJECT_DIR = DATA_DIR      # ROMs/, models/, presets and the error log live here
DEFAULT_GAME = "mario"
TENSORBOARD_PORT = 6006
TENSORBOARD_LOG = PROJECT_DIR / "tensorboard.log"    # what the TensorBoard process printed
TENSORBOARD_START_S = 90     # how long the built app's TensorBoard may take to answer

# SB3's verbose=1 table: "|    ep_rew_mean    | 123     |"
STAT_LINE = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([\S]+)\s+\|")
TRACKED_STATS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "fps", "time_elapsed")
# SB3's EvalCallback, three lines per evaluation (the third only on a record):
#   Eval num_timesteps=100000, episode_reward=123.45 +/- 6.78
#   Episode length: 1234.00 +/- 0.00
#   New best mean reward!
EVAL_LINE = re.compile(r"Eval num_timesteps=(\d+), episode_reward=(-?[\d.]+) \+/- ([\d.]+)")
EVAL_LEN_LINE = re.compile(r"Episode length: (-?[\d.]+) \+/-")
EVAL_BEST_LINE = "New best mean reward!"
# Live-game cells, in display order. A person playing has no round counter
# (see set_live_mode).
LIVE_KEYS = ("episode", "world", "power", "lives", "coins", "reward", "x", "steps", "action")
HUMAN_HIDDEN_KEYS = ("episode",)
# What the Tracking panel follows: nothing, a training run, a search
# (tune sweep or wizard search), an agent playing, or the person playing.
ACTIVITIES = ("none", "train", "tune", "play", "human")


def is_noise_line(line: str) -> bool:
    """True for trainer output that the Messages box does not need: the
    stats tables (shown as numbers under Tracking) and the evaluation
    lines (shown in the Evaluations table)."""
    s = line.strip()
    if not s or s.startswith(("|", "-")):
        return True
    return bool(EVAL_LINE.search(s) or EVAL_LEN_LINE.search(s)) or s == EVAL_BEST_LINE


def format_stat(val: str) -> str:
    """A trainer number for a label: SB3 prints 1.74e+03, people read 1740."""
    try:
        x = float(val)
    except ValueError:
        return val
    if x != x:                                   # nan
        return "—"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}"


def format_elapsed(seconds: float) -> str:
    """'45 s', '12 min 05 s', '1 h 03 min' for the training clock."""
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"

LEVEL_CHOICES = level_choices()

# Window: tabs (grow) | Preview (the Game Boy photo) | Tracking (TRACK_W).
# The tabs need TABS_W, the Tracking panel TRACK_PANEL_W; the Preview's
# width and the window's height follow the Game Boy's size, which is the
# largest LCD scale the display has room for (see choose_lcd_scale).
TABS_W = 760
TRACK_W = 330               # width of the Tracking panel's contents
TRACK_PANEL_W = 360
CHROME_W, CHROME_H = 60, 155    # paddings, panel headers, top and status bars
SCREEN_MARGIN_W, SCREEN_MARGIN_H = 20, 80       # room for the menu bar, dock, window title

STOP_GRACE_SECONDS = 20     # SIGINT -> trainer saves final.zip; SIGKILL after this
NO_MODEL = "(no trained model yet)"     # shown in the model box until a run has produced one
FLASH_SECONDS = 6           # transient status-bar messages
PUMP_MS = 16                # the GUI's frame timer: paints the newest emulator frame at ~60 Hz


IDLE_SCREEN_TEXT = "No video"
HUMAN_FROM_START = "from the start"     # the "Start:" choice that boots the game normally
NO_GAMEPAD = "none — keyboard"
GAMEPAD_OFF = "unavailable"           # SDL could not start; the hover text says why
INTRO_DISABLED = os.environ.get("AIBOY_NO_INTRO") == "1"      # tests: silent, no video


def choose_lcd_scale(screen_w: int, screen_h: int) -> float:
    """The largest emulator pixel scale whose window fits the display."""
    for scale in LCD_SCALES:
        w, h = window_size(scale)
        if w <= screen_w - SCREEN_MARGIN_W and h <= screen_h - SCREEN_MARGIN_H:
            return scale
    return LCD_SCALES[-1]


def window_size(lcd_scale: float) -> tuple[int, int]:
    pw, ph = photo_size(lcd_scale)
    return TABS_W + pw + TRACK_PANEL_W + CHROME_W, ph + CHROME_H


def idle_screen(intro: IntroVideo | None = None,
                size: tuple[int, int] = (240, 216)) -> Image.Image:
    """Screen for the LCD while no video is active: the intro video's last
    frame with the logo, with a pixel-font "No video" under it. Falls back
    to a drawn screen when the intro assets are missing."""
    from PIL import ImageDraw
    if intro is not None:
        base = Image.fromarray(intro.idle_frame())                        # 160x144
        draw = ImageDraw.Draw(base)
        w = draw.textlength(IDLE_SCREEN_TEXT)
        draw.text(((GAME_W - w) / 2, 92), IDLE_SCREEN_TEXT, fill=(2, 10, 7))
        return base.resize(size, Image.NEAREST)
    img = Image.new("RGB", (GAME_W, GAME_H), (155, 188, 15))
    draw = ImageDraw.Draw(img)
    dark = (15, 56, 15)
    draw.rectangle([8, 8, GAME_W - 9, GAME_H - 9], outline=dark, width=2)
    for i, line in enumerate((APP_NAME, IDLE_SCREEN_TEXT)):
        w = draw.textlength(line)
        draw.text(((GAME_W - w) / 2, 56 + i * 20), line, fill=dark)
    return img.resize(size, Image.NEAREST)


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
        root.title(APP_NAME)
        self.lcd_scale = choose_lcd_scale(root.winfo_screenwidth(), root.winfo_screenheight())
        self.window_w, self.window_h = window_size(self.lcd_scale)
        root.geometry(f"{self.window_w}x{self.window_h}")
        root.minsize(self.window_w - 90, self.window_h - 80)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Cross-thread channels
        self.stats_queue: queue.Queue = queue.Queue()
        self.frames = LatestFrame()
        self.player = EmbeddedPlayer(self.frames, self.stats_queue)
        # What the person is holding (keyboard, mouse, controller) while
        # playing, mapped by the remembered control scheme; the controller
        # thread watches for a pad all the time.
        self.held = HeldButtons()
        self.controls = ControlMap.load()
        self.gamepad = Gamepad(self.held, self.controls,
                               on_change=lambda name: self.stats_queue.put(("gamepad", name)))
        self._controls_dialog: ControlsDialog | None = None

        # Subprocess / thread state
        self.train_proc: subprocess.Popen | None = None
        self._train_stop_requested = False
        self.play_stop = threading.Event()
        self.play_thread: threading.Thread | None = None
        self.play_mode = "agent"                # "agent" (a model plays) or "human"
        self.play_rounds: list[dict] = []       # finished rounds of the current playback
        # A playback is in progress as far as the window is concerned: from
        # `_begin_playback` until its end event (`play_done` / `play_error`)
        # has been handled. Its thread may outlive this by a moment while it
        # closes its emulator; `claim_screen` waits for that.
        self._play_open = False
        self.preview_stop = threading.Event()
        self.preview_thread: threading.Thread | None = None
        self.tune_stop = threading.Event()
        self.tune_thread: threading.Thread | None = None
        self.tune_proc: subprocess.Popen | None = None
        self.tb_proc: subprocess.Popen | None = None
        self._tb_started_at = 0.0
        self._train_target_steps = 1
        self._train_rollout_size = 1
        self._train_baseline_steps: int | None = None
        self._train_rate = tuning.RateEstimator()

        self._model_paths: dict[str, Path] = {}
        self._all_presets: dict[str, dict] = {}
        self._tune_results: dict[int, tuning.ConfigResult] = {}
        self.frames_painted = 0                 # emulator frames put on the LCD so far
        self._pending_stats: dict[str, str] = {}   # live-panel values, applied once per pump
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.train_progress_text = tk.StringVar(value="—")
        self._closing = False
        self._train_run_name = ""
        self._train_preview_on = False          # the running/last run shows its best model live
        self._evals: list[dict] = []            # the run's evaluations, from the trainer's output
        self._activity = "none"                 # what Tracking follows (ACTIVITIES)
        self._track_shown: dict[str, bool] = {}
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
        self.gamepad.start()
        self.rescan_roms()
        self.refresh_run_names()
        self.refresh_models()
        self.refresh_presets()
        self._pump()
        if self.intro is not None:
            # Boot video with its jingle once the window is up (always, like
            # a real Game Boy; the Sound checkbox is for the games); ends on
            # the idle frame ("No video") unless something else takes the screen.
            self.gameboy.set_power(True)
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
        return False, f"'{self.game}' cannot be trained here: {text}"

    def _update_game_status(self) -> None:
        info = self.current_rom()
        if info is None:
            level, text = "unsupported", "no ROM selected"
        else:
            level, text = info.status()
        self.game_status_var.set(text)
        colors = {"ok": THEME.ok, "playable": THEME.warn,
                  "unsupported": THEME.err, "checking": THEME.muted}
        self.game_status_label.config(foreground=colors[level])
        self.root.title(f"{APP_NAME} — {display_name(self.game)}" if info else APP_NAME)
        self._update_start_buttons()

    def _update_start_buttons(self) -> None:
        """Start buttons are only enabled for a runnable game (and no run active)."""
        ok, _ = self.game_runnable()
        busy = self.busy()
        state = "normal" if ok and not busy else "disabled"
        self.btn_train_start.config(state=state)
        self.btn_tune_start.config(state=state)
        if not self.playing_active():
            self.btn_play_start.config(state=state)
            # Anyone can play a ROM that boots, supported for training or not.
            info = self.current_rom()
            can_play = info is not None and info.playable and not busy
            self.btn_human.config(state="normal" if can_play else "disabled")
            # Locked with the other inputs while training or tuning runs
            # (a ROM probe finishing mid-run must not unlock it).
            self.human_start_combo.config(
                state="readonly" if self.game == DEFAULT_GAME and not busy else "disabled")
        if self.wizard is not None:
            self.wizard.on_game_changed(ok)

    def _on_game_changed(self) -> None:
        self._update_game_status()
        self.refresh_run_names()
        self.refresh_models()
        self.refresh_presets()
        self.set_live_mode(self.play_mode)      # the live panel's words follow the game

    # ---------- state queries ----------

    def training_active(self) -> bool:
        return self.train_proc is not None and self.train_proc.poll() is None

    def tuning_active(self) -> bool:
        return self.tune_thread is not None and self.tune_thread.is_alive()

    def playing_active(self) -> bool:
        """An agent or a person is playing on the screen (see `_play_open`)."""
        return self._play_open

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

        # Main area, three sections: workflow tabs | Preview | Tracking.
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.nb = ttk.Notebook(main)
        self.nb.grid(row=0, column=0, sticky="nsew")
        # The Preview's header carries the Sound checkbox: the game's own
        # sound while something plays at real speed (see player.SoundGate).
        preview_head = ttk.Frame(main)
        ttk.Label(preview_head, text="Preview").pack(side="left")
        self.sound_var = tk.BooleanVar(value=bool(settings.get("sound")))
        self.sound_check = ttk.Checkbutton(preview_head, text="Sound", variable=self.sound_var,
                                           command=self._on_sound_toggled)
        self.sound_check.pack(side="left", padx=(14, 0))
        tooltip(self.sound_check, "Play the game's sound while you play yourself or an agent "
                                  "plays at real speed (1×). Faster or slower playback and the "
                                  "live preview of a training run stay silent: the sound cannot "
                                  "follow them. Off until you tick it; remembered for next time.")
        self.player.set_sound(bool(self.sound_var.get()))
        self.preview_frame = ttk.LabelFrame(main, labelwidget=preview_head, padding=8)
        self.preview_frame.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.tracking_frame = ttk.LabelFrame(main, text="Tracking", padding=8)
        self.tracking_frame.grid(row=0, column=2, sticky="ns", padx=(10, 0))

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
        self._build_screen(self.preview_frame)
        self._build_tracking(self.tracking_frame)
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
        """Re-read the experience file after a trainer finished, act on what
        it now says (improve presets), and tell the tabs that show or use it."""
        self.experience.reload()
        if settings.get("auto_improve_presets") and not self.busy():
            self.improve_presets(quiet=True)
        if self.experience_tab is not None:
            self.experience_tab.refresh()
        if self.wizard is not None:
            self.wizard.on_experience_changed()
        self.tune_update_summary()

    def improve_presets(self, *, quiet: bool = False) -> list:
        """Change every preset whose own settings AIboy has measured to be
        clearly beaten (see experience.Experience.improvements). Built-ins
        get a resettable override; each change is logged, recorded in the
        experience file and shown in the status bar. Returns the changes."""
        game_presets = {n: c for n, c in presets.load_all().items()
                        if c.get("game", DEFAULT_GAME) == self.game}
        improved = self.experience.improvements(game_presets)
        for imp in improved:
            try:
                presets.apply_improvement(imp.preset, imp.overrides)
            except (ValueError, OSError) as e:
                self.append_log(f"[aiboy] could not improve preset '{imp.preset}': {e}\n")
                continue
            self.experience.record_improvement(imp, game_presets[imp.preset])
            self.append_log(f"[aiboy] preset '{imp.preset}' improved: {imp.summary()}\n")
        if improved:
            self.refresh_presets()
            if self.preset_var.get() in {imp.preset for imp in improved} and not self.busy():
                self.apply_preset()           # the Train tab shows the improved values
            names = ", ".join(f"'{imp.preset}'" for imp in improved)
            self.flash(f"AIboy improved {len(improved)} preset(s) from what it learned: {names} "
                       f"(details on the Experience tab).", seconds=12)
        elif not quiet:
            self.flash("No preset can be improved right now: AIboy has found nothing clearly "
                       "better than the presets' own settings.")
        return improved

    def show_tab(self, tab: ttk.Frame) -> None:
        self.nb.select(tab)

    def _init_canvas(self) -> None:
        """Idle screen ("No video") when nothing else is on the LCD; LED off."""
        self.gameboy.set_screen(idle_screen(self.intro, self.gameboy.screen_size))
        self.gameboy.set_power(False)

    def _on_sound_toggled(self) -> None:
        """The Sound checkbox: remembered, and a game playing at real speed
        follows it at once (the emulator thread mutes or unmutes)."""
        on = bool(self.sound_var.get())
        settings.put("sound", on)
        self.player.set_sound(on)

    # ---------- Train tab ----------

    def _build_train(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        # Resume is "if one exists": with the box ticked a run that already
        # has checkpoints is continued, a new run simply starts (see
        # `_resume_from`). Ticked by default so a re-used run name never
        # overwrites a trained model by accident.
        self.resume_var = tk.BooleanVar(value=True)
        self.resume_var.trace_add("write", lambda *_a: self._update_run_hint())
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
        self.resume_check = ttk.Checkbutton(run_row, text="Resume if possible",
                                            variable=self.resume_var)
        self.resume_check.pack(side="left", padx=(18, 0))
        tooltip(self.resume_check, "Continue from the run's newest checkpoint if it has one. A "
                                   "run without a checkpoint simply starts fresh. Untick it to "
                                   "start over and replace the run's models.")
        self.run_hint_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.run_hint_var, foreground=THEME.muted, font=MONO,
                  wraplength=600).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.preview_check = ttk.Checkbutton(controls, text="Live preview while training",
                                             variable=self.preview_var)
        self.preview_check.grid(row=2, column=0, sticky="w", pady=(4, 0))
        tooltip(self.preview_check, "Play the run's newest best model on the Preview screen "
                                    "while it trains, reloading it whenever it improves. Costs "
                                    "some training speed.")

        btns = ttk.Frame(controls)
        btns.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.btn_train_start = ttk.Button(btns, text="Start training", command=self.start_training)
        self.btn_train_start.pack(side="left", padx=(0, 4))
        self.btn_train_stop = ttk.Button(btns, text="Stop", command=self.stop_training,
                                         state="disabled")
        self.btn_train_stop.pack(side="left", padx=4)
        btn_tb = ttk.Button(btns, text="TensorBoard", command=self.open_tensorboard)
        btn_tb.pack(side="left", padx=(16, 4))
        tooltip(btn_tb, "Open the training curves of every run of this game in the browser.")
        btn_log = ttk.Button(btns, text="Full log…", command=self.open_log)
        btn_log.pack(side="left", padx=4)
        tooltip(btn_log, "Everything the trainer printed, word for word, in its own window. "
                         "Only needed when something went wrong.")
        # Housekeeping for the named run, on its own line so nothing is clipped.
        keep = ttk.Frame(controls)
        keep.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(keep, text="This run:", foreground=THEME.muted).pack(side="left", padx=(0, 4))
        self.btn_run_open = ttk.Button(keep, text="Open folder", command=self._open_run_folder)
        self.btn_run_open.pack(side="left", padx=2)
        self.btn_run_compact = ttk.Button(keep, text="Compact…", command=self._compact_run)
        self.btn_run_compact.pack(side="left", padx=2)
        self.btn_run_delete = ttk.Button(keep, text="Delete…", command=self._delete_run)
        self.btn_run_delete.pack(side="left", padx=2)

        # ---- This run: its evaluations and the trainer's messages, read
        # out of the trainer's output. Progress, ETA and the live numbers
        # are under Tracking; the raw output stays behind "Full log…". ----
        parent.rowconfigure(3, weight=1)
        details = ttk.Frame(parent)
        details.grid(row=3, column=0, sticky="nsew")
        details.columnconfigure(1, weight=1)
        details.rowconfigure(0, weight=1)
        eval_frame = ttk.LabelFrame(details, text="Evaluations", padding=4)
        eval_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.eval_tree = make_table(eval_frame, [
            ("n", "#", 30, "e", False), ("steps", "steps", 70, "e", False),
            ("score", "score", 80, "e", False), ("length", "length", 60, "e", False),
            ("note", "", 70, "w", True),
        ], height=6)
        tooltip(self.eval_tree, "Every evaluation of this run: after how many steps it was "
                                "made, the score of the evaluation round, its length, and "
                                "whether it set a new record (the best model is saved then).")
        msg_frame = ttk.LabelFrame(details, text="Messages", padding=4)
        msg_frame.grid(row=0, column=1, sticky="nsew")
        msg_frame.columnconfigure(0, weight=1)
        msg_frame.rowconfigure(0, weight=1)
        self.messages_text = tk.Text(msg_frame, height=6, width=30, wrap="word", font=MONO,
                                     relief="flat", highlightthickness=0,
                                     background=THEME.panel[0], foreground=THEME.panel[1],
                                     insertbackground=THEME.panel[1], state="disabled")
        self.messages_text.grid(row=0, column=0, sticky="nsew")
        msg_scroll = ttk.Scrollbar(msg_frame, command=self.messages_text.yview)
        msg_scroll.grid(row=0, column=1, sticky="ns")
        self.messages_text.config(yscrollcommand=msg_scroll.set)
        tooltip(self.messages_text, "What the trainer, the search and the app report: how a "
                                    "run was started, warnings, errors, finished rounds, "
                                    "improved presets. The numbers it prints every few "
                                    "seconds are left out; they are under Tracking.")

        # The raw output, in a window of its own (see open_log).
        self._log_win = tk.Toplevel(self.root)
        self._log_win.title(f"{APP_NAME} — full log")
        self._log_win.withdraw()
        self._log_win.protocol("WM_DELETE_WINDOW", self._log_win.withdraw)
        log_frame = ttk.Frame(self._log_win, padding=4)
        log_frame.pack(fill="both", expand=True)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=30, width=110, wrap="none", font=MONO,
                                background="#111", foreground="#ddd", insertbackground="#ddd")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=yscroll.set)

    def open_log(self) -> None:
        """Show the raw trainer output (kept for troubleshooting)."""
        win = self._log_win
        if not getattr(self, "_log_placed", False):
            win.geometry(f"+{self.root.winfo_rootx() + 40}+{self.root.winfo_rooty() + 60}")
            self._log_placed = True
        win.deiconify()
        win.lift()

    def note(self, text: str) -> None:
        """Add one line to the Messages box on the Train tab."""
        box = self.messages_text
        box.config(state="normal")
        box.insert("end", text if text.endswith("\n") else text + "\n")
        line_count = int(box.index("end-1c").split(".")[0])
        if line_count > 500:
            box.delete("1.0", "100.0")
        box.see("end")
        box.config(state="disabled")

    def _clear_run_details(self) -> None:
        """A new training run starts: forget the previous one's evaluations."""
        self._evals.clear()
        for row in self.eval_tree.get_children():
            self.eval_tree.delete(row)
        for key in ("count", "last", "best"):
            self.eval_vars[key].set("—")

    def _add_eval(self, step: int, reward: float, std: float) -> None:
        ev = {"n": len(self._evals) + 1, "step": step, "reward": reward, "std": std,
              "length": None, "best": False}
        self._evals.append(ev)
        self.eval_tree.insert("", "end", iid=str(ev["n"]),
                              values=(ev["n"], tuning.format_steps(step), f"{reward:.1f}", "—", ""))
        self.eval_tree.see(str(ev["n"]))
        self.eval_vars["count"].set(str(ev["n"]))
        self.eval_vars["last"].set(f"{reward:.1f}")

    def _update_last_eval(self, *, length: float | None = None, best: bool = False) -> None:
        if not self._evals:
            return
        ev = self._evals[-1]
        if length is not None:
            ev["length"] = length
        if best:
            ev["best"] = True
            self.eval_vars["best"].set(f"{ev['reward']:.1f} at {tuning.format_steps(ev['step'])}")
        self.eval_tree.item(str(ev["n"]), values=(
            ev["n"], tuning.format_steps(ev["step"]), f"{ev['reward']:.1f}",
            f"{ev['length']:.0f}" if ev["length"] is not None else "—",
            "new best" if ev["best"] else ""), tags=("best",) if ev["best"] else ())

    # ---------- Preview and Tracking panels (always visible) ----------

    def _build_screen(self, parent: ttk.Frame) -> None:
        """The Game Boy: the emulator view on its LCD, its buttons lighting
        up as the agent presses them, and a status line under it. Playback,
        the live preview during training and the wizard's Watch step all
        render here, so nothing has to switch tabs to be seen."""
        self.gameboy = GameBoyView(parent, self.lcd_scale)
        self.gameboy.pack()
        self._init_canvas()
        self.keyboard = KeyboardInput(self.gameboy, self.held, self.controls)
        self.gameboy.on_buttons(lambda pressed: self.held.set("mouse", pressed))
        self.gameboy.set_key_hints(self.controls.key_hints())
        self.play_status_var = tk.StringVar(value="idle")
        status = ttk.Label(parent, textvariable=self.play_status_var, font=MONO,
                           wraplength=self.gameboy.geo.width, anchor="center", justify="center")
        status.pack(fill="x", pady=(4, 0))
        tooltip(status, "What the screen is showing right now: the model being played, the "
                        "live preview of a training run, your own game, or how the last "
                        "round ended.")

    def _build_tracking(self, parent: ttk.Frame) -> None:
        """The panel follows what the app is doing (see `_layout_tracking`).
        One header line names the activity; under it only the boxes that
        matter for it: the live game while something plays, the finished
        rounds of an agent, the training run's numbers, the search's
        numbers, and the two ways to start something (watch an agent, play
        yourself) whenever nothing is running."""
        self.activity_var = tk.StringVar(value="Nothing running")
        head = ttk.Label(parent, textvariable=self.activity_var, font=MONO_BOLD,
                         wraplength=TRACK_W, anchor="w", justify="left")
        head.pack(fill="x", pady=(0, 6))
        tooltip(head, "What this panel is following right now. It changes by itself when a "
                      "training, a search or a game starts, and keeps the last result until "
                      "something else starts (Clear empties it).")
        self._track_boxes = {
            "live": self._build_live_box(parent),
            "view": self._build_view_box(parent),
            "rounds": self._build_rounds_box(parent),
            "train": self._build_train_box(parent),
            "tune": self._build_search_box(parent),
            "watch": self._build_watch_box(parent),
            "you": self._build_you_box(parent),
        }
        # Advanced options live in a small dialog (see _open_play_advanced). The
        # observation setup is normally filled in from the model's run.json.
        self.play_max_steps_var = tk.IntVar(value=0)
        self.play_stochastic_var = tk.BooleanVar(value=False)
        self.play_action_repeat_var = tk.IntVar(value=4)
        self.play_frame_stack_var = tk.IntVar(value=4)
        self.play_obs_type_var = tk.StringVar(value="tiles")
        self.play_level_var = tk.StringVar(value="default")
        # Attempt limits follow the model's run.json (see sync_play_options).
        self.play_time_budget = presets.PRESET_DEFAULTS["time_budget"]
        self.play_stall_steps = presets.PRESET_DEFAULTS["stall_steps"]
        self._play_adv_win: tk.Toplevel | None = None
        self._layout_tracking()

    def _build_live_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """Live game: two columns of label / value pairs. The wording and the
        cells follow who is playing (see set_live_mode)."""
        stats = ttk.LabelFrame(parent, text="Live game", padding=(6, 2))
        self.play_stat_vars = {k: tk.StringVar(value="—") for k in LIVE_KEYS}
        self._live_labels: dict[str, ttk.Label] = {}
        self._live_values: dict[str, ttk.Label] = {}
        self._live_help: dict[str, str] = {}
        for key in LIVE_KEYS:
            lbl = ttk.Label(stats, foreground=THEME.muted)
            val = ttk.Label(stats, textvariable=self.play_stat_vars[key], font=MONO_BOLD,
                            anchor="w")
            self._live_labels[key], self._live_values[key] = lbl, val
            tooltip(lbl, lambda k=key: self._live_help.get(k, ""), val)
        # The agent's view: the table of numbers it plays from (see
        # _build_view_box); for a person, what an agent would get for their game.
        self.view_var = tk.BooleanVar(value=bool(settings.get("show_agent_view")))
        self.view_title_var = tk.StringVar(value="What the agent sees")   # the table's header
        self.view_check = ttk.Checkbutton(stats, text="Show what the agent sees",
                                          variable=self.view_var, command=self._on_view_toggled)
        tooltip(self.view_check, lambda: (
            "Show, under this box, the numbers an agent gets instead of the picture, for the "
            "game you are playing: one per tile of the screen, tinted by what they mean. A way "
            "to check what an agent can and cannot tell apart at any spot. Hover the table for "
            "how to read it." if self.play_mode == "human" else
            "Show, under this box, the numbers the agent gets instead of the picture: one per "
            "tile of the screen, tinted by what they mean. Hover the table for how to read it."))
        self.set_live_mode("agent")
        return stats

    def _build_view_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """The agent's observation as a 16 x 20 table (see agent_view.py),
        shown while an agent plays, or a person does, and the checkbox is on."""
        head = ttk.Frame(parent)
        ttk.Label(head, textvariable=self.view_title_var).pack(side="left")
        info_icon(head, lambda: agent_view_legend(self.game)).pack(side="left", padx=(4, 0))
        box = ttk.LabelFrame(parent, labelwidget=head, padding=(4, 2))
        self.agent_view = AgentView(box, self.game)
        self.agent_view.pack()
        tooltip(self.agent_view, lambda: agent_view_legend(self.game))
        self.view_note_var = tk.StringVar(value="")
        self.view_note = ttk.Label(box, textvariable=self.view_note_var, foreground=THEME.muted,
                                   wraplength=TRACK_W - 20, justify="left")
        self._pending_view = None       # newest grid from the player, drawn once per pump
        if self.view_var.get():
            self.player.view_wanted.set()
        return box

    def _on_view_toggled(self) -> None:
        on = bool(self.view_var.get())
        settings.put("show_agent_view", on)
        if on:
            self.player.view_wanted.set()
        else:
            self.player.view_wanted.clear()
            self._pending_view = None
        self._layout_tracking()

    def _show_agent_view(self, grid) -> None:
        """Draw the newest grid (None: a pixel model, which sees the picture)."""
        if grid is None:
            self.agent_view.pack_forget()
            self.view_note_var.set("This agent sees the screen's pixels, the picture itself, "
                                   "so there is no table of numbers.")
            self.view_note.pack(fill="x")
            return
        if not self.agent_view.winfo_ismapped():
            self.view_note.pack_forget()
            self.agent_view.pack()
        self.agent_view.set_game(self.game)
        self.agent_view.show(grid)

    def _build_rounds_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """Finished rounds of the agent being watched (or previewed)."""
        box = ttk.LabelFrame(parent, text="Rounds", padding=(4, 2))
        self.rounds_tree = make_table(box, [
            ("round", "#", 34, "e", False), ("score", "score", 58, "e", False),
            ("steps", "steps", 54, "e", False), ("end", "how it ended", 150, "w", True),
        ], height=4)
        tooltip(self.rounds_tree, "Every finished round of this playback: its score, how many "
                                  "decisions the agent made, and how it ended.")
        return box

    def _build_train_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """The training run: name, state, progress, the key numbers from the
        trainer's output, and its evaluations."""
        box = ttk.LabelFrame(parent, text="Training", padding=(6, 2))
        for c in (1, 3):
            box.columnconfigure(c, weight=1)
        self.stat_vars: dict[str, tk.StringVar] = {
            k: tk.StringVar(value=v) for k, v in
            [("status", "idle"), ("total_timesteps", "—"), ("ep_rew_mean", "—"),
             ("ep_len_mean", "—"), ("fps", "—"), ("time_elapsed", "—")]}
        self.train_run_var = tk.StringVar(value="—")
        self.train_elapsed_var = tk.StringVar(value="—")
        self.eval_vars = {k: tk.StringVar(value="—") for k in ("count", "last", "best")}

        def cell(row: int, col: int, label: str, var: tk.StringVar, help_text: str,
                 span: int = 1, width: int | None = 8, wrap: int = 0) -> None:
            lbl = ttk.Label(box, text=f"{label}:", foreground=THEME.muted)
            lbl.grid(row=row, column=col * 2, sticky="nw", padx=(0 if col == 0 else 12, 4))
            val = ttk.Label(box, textvariable=var, font=MONO_BOLD, anchor="w", width=width,
                            wraplength=wrap or 0, justify="left")
            val.grid(row=row, column=col * 2 + 1, columnspan=span * 2 - 1, sticky="w")
            tooltip(lbl, help_text, val)

        cell(0, 0, "run", self.train_run_var, "Name of the run being trained; its models are "
             "saved under models/<game>/<run>.", span=2, width=None, wrap=TRACK_W - 60)
        cell(1, 0, "status", self.stat_vars["status"], "State of the run: running, stopping, "
             "finished, stopped or failed.", span=2, width=None, wrap=TRACK_W - 60)
        ttk.Progressbar(box, mode="determinate", maximum=100,
                        variable=self.train_progress_var).grid(row=2, column=0, columnspan=4,
                                                               sticky="ew", pady=(4, 0))
        prog = ttk.Label(box, textvariable=self.train_progress_text, font=MONO, anchor="w",
                         wraplength=TRACK_W - 10)
        prog.grid(row=3, column=0, columnspan=4, sticky="w", pady=(1, 4))
        tooltip(prog, "Steps trained of the run's target, and the time left estimated from "
                      "the speed of the last few updates.")
        cell(4, 0, "avg score", self.stat_vars["ep_rew_mean"],
             "Average score of the last 100 training rounds. Rising is good.")
        cell(4, 1, "round length", self.stat_vars["ep_len_mean"],
             "Average length (decisions) of the last 100 training rounds.")
        cell(5, 0, "speed", self.stat_vars["fps"],
             "Game steps per second across all parallel games.")
        cell(5, 1, "elapsed", self.train_elapsed_var, "Time since the run started.")
        cell(6, 0, "evaluations", self.eval_vars["count"],
             "How many times the agent has been evaluated so far: every evaluation plays a "
             "test round with the current model (the table on the Train tab lists them).")
        cell(6, 1, "last score", self.eval_vars["last"],
             "Score of the most recent evaluation round.")
        cell(7, 0, "best score", self.eval_vars["best"],
             "Best evaluation score so far and after how many steps it was reached; the "
             "model that scored it is the run's best model.", span=2, width=None,
             wrap=TRACK_W - 90)
        self.btn_track_train_stop = ttk.Button(box, text="■ Stop training",
                                               command=self.stop_training)
        self.btn_track_train_stop.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 2))
        tooltip(self.btn_track_train_stop, "End the run early. Its best model so far is kept "
                                           "and the last state is saved.")
        return box

    def _build_search_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """A search for better settings (the wizard's search or a Tune sweep):
        which variation is being tried, how far it is, and the best so far."""
        box = ttk.LabelFrame(parent, text="Search", padding=(6, 2))
        for c in (1, 3):
            box.columnconfigure(c, weight=1)
        self.tune_vars = {k: tk.StringVar(value="—") for k in
                          ("trial", "trying", "steps", "reward", "fps", "ep_len", "best")}

        def cell(row: int, col: int, label: str, key: str, help_text: str,
                 span: int = 1, width: int | None = 8, wrap: int = 0) -> None:
            lbl = ttk.Label(box, text=f"{label}:", foreground=THEME.muted)
            lbl.grid(row=row, column=col * 2, sticky="nw", padx=(0 if col == 0 else 12, 4))
            val = ttk.Label(box, textvariable=self.tune_vars[key], font=MONO_BOLD, anchor="w",
                            width=width, wraplength=wrap or 0, justify="left")
            val.grid(row=row, column=col * 2 + 1, columnspan=span * 2 - 1, sticky="w")
            tooltip(lbl, help_text, val)

        cell(0, 0, "trial", "trial", "Which short training run of the search this is, and how "
             "many are already known from earlier searches (those are scored from memory).",
             span=2, width=None, wrap=TRACK_W - 60)
        cell(1, 0, "trying", "trying", "The settings of the variation being trained right now.",
             span=2, width=None, wrap=TRACK_W - 70)
        ttk.Progressbar(box, mode="determinate", maximum=100,
                        variable=self.tune_progress_var).grid(row=2, column=0, columnspan=4,
                                                              sticky="ew", pady=(4, 0))
        prog = ttk.Label(box, textvariable=self.tune_progress_text, font=MONO, anchor="w",
                         wraplength=TRACK_W - 10)
        prog.grid(row=3, column=0, columnspan=4, sticky="w", pady=(1, 4))
        tooltip(prog, "Trials finished of the whole search, and the estimated time left.")
        cell(4, 0, "steps", "steps", "Steps trained of this trial's length.", span=2, width=None)
        cell(5, 0, "avg score", "reward", "Average score of the trial's last 100 training rounds.")
        cell(5, 1, "round length", "ep_len", "Average length of the trial's last 100 rounds.")
        cell(6, 0, "speed", "fps", "Game steps per second across all parallel games.")
        cell(7, 0, "best so far", "best", "The variation with the best score of this search, "
             "and its score.", span=2, width=None, wrap=TRACK_W - 90)
        self.btn_track_tune_stop = ttk.Button(box, text="■ Stop search", command=self.stop_tuning)
        self.btn_track_tune_stop.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 2))
        tooltip(self.btn_track_tune_stop, "End the search after the current trial. The results "
                                          "so far are kept.")
        return box

    def _build_watch_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """Pick a saved model and watch it play on the Game Boy."""
        ctl = ttk.LabelFrame(parent, text="Watch an agent", padding=6)
        ctl.columnconfigure(1, weight=1)
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(ctl, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_selected())
        # Re-scan the model files whenever the list is opened; no Refresh button needed.
        self.model_combo.bind("<Button-1>", lambda e: self.refresh_models(), add="+")
        self.model_note_var = tk.StringVar(value="")
        tooltip(self.model_combo, lambda: "Every trained agent of this game: the best and the "
                                          "final model of each run, and its snapshots. " + (
                                          f"Selected: {self.model_note_var.get()}"
                                          if self.model_note_var.get() else ""))
        self.play_episodes_var = tk.IntVar(value=3)
        self.play_speed_label_var = tk.StringVar(value=SPEED_CHOICES[1][0])
        ep_lbl = ttk.Label(ctl, text="Rounds:")
        ep_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ep_spin = ttk.Spinbox(ctl, from_=1, to=100, textvariable=self.play_episodes_var, width=5)
        ep_spin.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(6, 0))
        tooltip(ep_lbl, "How many rounds to play. A round ends when the agent dies, "
                        "completes the game, or runs out of time or progress.", ep_spin)
        sp_lbl = ttk.Label(ctl, text="Speed:")
        sp_lbl.grid(row=2, column=0, sticky="w", pady=(4, 0))
        speed_combo = ttk.Combobox(ctl, textvariable=self.play_speed_label_var, state="readonly",
                                   values=[c[0] for c in SPEED_CHOICES], width=13)
        speed_combo.grid(row=2, column=1, sticky="w", padx=(4, 0), pady=(4, 0))
        tooltip(sp_lbl, "Playback speed relative to a real Game Boy.", speed_combo)
        self.btn_play_adv = ttk.Button(ctl, text="Advanced…", command=self._open_play_advanced)
        self.btn_play_adv.grid(row=2, column=2, sticky="e", pady=(4, 0))
        tooltip(self.btn_play_adv, "Step cap, random actions, and the observation setup "
                                   "(normally read from the model's run).")

        btns = ttk.Frame(ctl)
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.btn_play_start = ttk.Button(btns, text="▶ Play", command=self.start_playing)
        self.btn_play_start.pack(side="left")
        self.btn_play_stop = ttk.Button(btns, text="■ Stop", command=self.stop_playing,
                                        state="disabled")
        self.btn_play_stop.pack(side="left", padx=4)
        self.btn_screen_clear = ttk.Button(btns, text="Clear", command=self.clear_screen)
        self.btn_screen_clear.pack(side="left", padx=4)
        tooltip(self.btn_play_start, "Let the selected agent play on the Game Boy.")
        tooltip(self.btn_play_stop, "Stop the playback after the current step.")
        tooltip(self.btn_screen_clear, "Stop the playback, blank the screen and empty this panel.")
        self._play_widgets: list[tk.Widget] = [self.model_combo, ep_spin, speed_combo,
                                               self.btn_play_start, self.btn_play_adv]
        return ctl

    def _build_you_box(self, parent: ttk.Frame) -> ttk.LabelFrame:
        """Play yourself: the plain game on the Game Boy, with the keyboard,
        the mouse on the picture, or a game controller."""
        head = ttk.Frame(parent)
        ttk.Label(head, text="Play yourself").pack(side="left")
        info_icon(head, lambda: self.controls.help_text(self.gamepad.pad_kind,
                                                        self.gamepad.pad_name)).pack(
            side="left", padx=(4, 0))
        you = ttk.LabelFrame(parent, labelwidget=head, padding=6)
        you.columnconfigure(1, weight=1)
        start_lbl = ttk.Label(you, text="Start:")
        start_lbl.grid(row=0, column=0, sticky="w")
        self.human_start_var = tk.StringVar(value=HUMAN_FROM_START)
        self.human_start_combo = ttk.Combobox(
            you, textvariable=self.human_start_var, state="readonly", width=13,
            values=[HUMAN_FROM_START] + [f"{w}-{l}" for (w, l) in SML_ALL_LEVELS])
        self.human_start_combo.grid(row=0, column=1, sticky="w", padx=(4, 0))
        tooltip(start_lbl, "Boot the game normally (title screen, press START), or jump "
                           "straight into a level of Super Mario Land.", self.human_start_combo)
        pad_lbl = ttk.Label(you, text="Controller:")
        pad_lbl.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.gamepad_var = tk.StringVar(value=NO_GAMEPAD)
        self.gamepad_label = ttk.Label(you, textvariable=self.gamepad_var, font=MONO,
                                       foreground=THEME.muted, wraplength=TRACK_W - 90)
        self.gamepad_label.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))
        tooltip(pad_lbl, self._gamepad_help, self.gamepad_label)
        self.btn_human = ttk.Button(you, text="🎮 Play yourself", command=self.toggle_human_play)
        self.btn_human.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        tooltip(self.btn_human, lambda: (
            "Stop your game." if self.play_mode == "human" and self.playing_active() else
            "Play the game yourself on the Game Boy, at real speed: keyboard, controller, or "
            "clicks on the picture's buttons. Hover the ⓘ for the keys."))
        self.btn_controls = ttk.Button(you, text="Controls…", command=self.open_controls)
        self.btn_controls.grid(row=2, column=1, sticky="e", pady=(8, 0))
        tooltip(self.btn_controls, "Choose which keys and which controller buttons press each "
                                   "Game Boy button. Remembered for next time.")
        self._play_widgets.extend([self.human_start_combo, self.btn_human])
        return you

    # ---------- Tracking: what to show ----------

    def tracking_layout(self) -> dict[str, bool]:
        """Which Tracking boxes the current activity needs (see ACTIVITIES).
        The last activity's boxes stay until another one starts; the two
        start boxes come back as soon as nothing runs."""
        act, running = self._activity, self.busy() or self.playing_active()
        # The live preview matters while the run trains; afterwards its best
        # model is selected under "Watch an agent".
        preview = act == "train" and self._train_preview_on and self.training_active()
        view = (act in ("play", "human") or preview) and bool(self.view_var.get())
        return {
            "live": act in ("play", "human") or preview,
            "view": view,
            "rounds": act == "play" or preview,
            "train": act == "train",
            "tune": act == "tune",
            "watch": act == "play" or not running,
            # The agent's table takes the room of "Play yourself" while an
            # agent's game is on the panel (Clear brings it back); a person's
            # game has no Rounds table, so the table fits next to their box.
            "you": act == "human" or (not running and not (act == "play" and view)),
        }

    def _layout_tracking(self) -> None:
        """Pack the boxes the activity needs, in panel order; called every
        pump, but only repacks when the set changes (a repack redraws the
        whole window on macOS)."""
        show = self.tracking_layout()
        if show == self._track_shown:
            return
        self._track_shown = show
        for name, box in self._track_boxes.items():
            box.pack_forget()
        self.rounds_tree.configure(height=3 if show["view"] else 4)   # room for the table
        for name, box in self._track_boxes.items():
            if show[name]:
                box.pack(fill="x", pady=(0, 6))
        # A Stop button only while there is something to stop.
        for btn, active in ((self.btn_track_train_stop, self.training_active()),
                            (self.btn_track_tune_stop, self.tuning_active())):
            if active:
                btn.grid()
            else:
                btn.grid_remove()

    def set_activity(self, activity: str, header: str) -> None:
        """Tracking now follows `activity` (one of ACTIVITIES); `header` is
        its one-line title."""
        assert activity in ACTIVITIES, activity
        self._activity = activity
        self.activity_var.set(header)
        self._track_shown = {}          # force a repack (a Stop button may have changed)
        self._layout_tracking()

    def _gamepad_help(self) -> str:
        """Hover text of the Controller line: which pads are in use and what
        is pressed on them right now, or why controllers are off."""
        pad = self.gamepad
        if pad.error:
            return (f"Game controllers are off: {pad.error}. The keyboard and the mouse still "
                    f"work. `python main.py controller-test` shows what SDL sees.")
        if not pad.pad_name:
            return ("No game controller found. Plug one in by USB or pair it by Bluetooth; it "
                    "is picked up while the app runs.")
        pressed = pad.raw_text()
        return (f"Game controller in use: {pad.pad_name}. "
                + (f"Pressed on it right now: {pressed}. " if pressed else
                   "Nothing is pressed on it right now. ")
                + "Every connected controller plays; Controls… shows what each button does.")

    def open_controls(self) -> None:
        """The key / controller mapping window (one at a time)."""
        if self._controls_dialog is not None and self._controls_dialog.win.winfo_exists():
            self._controls_dialog.win.lift()
            return

        def closed():
            self._controls_dialog = None
            if self.playing_active() and self.play_mode == "human":
                self.gamepad.active = True
                self.gameboy.focus_set()

        self._controls_dialog = ControlsDialog(self.gameboy, self.controls, self.gamepad,
                                               on_change=self.on_controls_changed,
                                               on_close=closed)

    def on_controls_changed(self) -> None:
        """A key or controller button was re-bound (already saved)."""
        self.gameboy.set_key_hints(self.controls.key_hints())
        self.keyboard.release_all()
        self.gamepad.remap()

    def set_live_mode(self, mode: str) -> None:
        """Word the live panel for who is playing ("agent" or "human") and
        show only the cells that mean something for them: a person has no
        round counter."""
        human = mode == "human"
        names = {"episode": "round", "reward": "score", "x": "position", "action": "pressing",
                 "steps": "time" if human else "steps"}
        spec = GAMES.get(self.game)
        if spec is not None:
            names.update(spec.stat_labels)      # e.g. Kirby's "power" cell is his health
        self._live_help = {
            "episode": "The round the agent is on, of how many.",
            "world": "Level Mario is in (world-level).",
            "power": ("Kirby's health, out of 6." if names.get("power") == "health" else
                      "Mario's size: small, super (mushroom) or superball (flower)."),
            "lives": "Lives left.", "coins": "Coins collected in this game.",
            "reward": ("The game's score." if human else
                       "Score the agent has earned in this round so far (its reward)."),
            "x": ("How far right Mario is in the level (the number the agent's table shows "
                  "in its fourth HUD cell, divided by 4096)." if human else
                  "How far right Mario is in the level, and the furthest he got."),
            "steps": ("Time left on the game's clock, and how long you have been playing."
                      if human else "Decisions the agent has made in this round."),
            "action": ("What you are pressing right now (also lit on the buttons)." if human
                       else "What the agent is pressing right now (also lit on the buttons)."),
        }
        shown = [k for k in LIVE_KEYS if not (human and k in HUMAN_HIDDEN_KEYS)]
        for key in LIVE_KEYS:
            self._live_labels[key].config(text=f"{names.get(key, key)}:")
            self._live_labels[key].grid_forget()
            self._live_values[key].grid_forget()
        left = shown[:(len(shown) + 1) // 2]
        for col, keys in enumerate((left, shown[len(left):])):
            for row, key in enumerate(keys):
                self._live_labels[key].grid(row=row, column=col * 2, sticky="w",
                                            padx=(0 if col == 0 else 12, 4))
                self._live_values[key].config(width=7 if col == 0 else 14)
                self._live_values[key].grid(row=row, column=col * 2 + 1, sticky="w")
        # The table of numbers: the agent's own while one plays, what an
        # agent would get while a person plays (a supported game only).
        self.view_check.config(text="Show what an agent would see" if human else
                               "Show what the agent sees")
        self.view_title_var.set("What an agent would see" if human else "What the agent sees")
        self.view_check.grid(row=len(left), column=0, columnspan=4, sticky="w", pady=(2, 0))

    def clear_rounds(self) -> None:
        self.play_rounds.clear()
        for row in self.rounds_tree.get_children():
            self.rounds_tree.delete(row)

    def add_round(self, r: dict) -> None:
        """A round of the agent finished: keep it and list it under Tracking."""
        self.play_rounds.append(r)
        iid = self.rounds_tree.insert("", "end", values=(
            r["episode"], f"{r['reward']:.0f}", r["steps"], r["end"]))
        self.rounds_tree.see(iid)

    def clear_screen(self) -> None:
        """Blank the emulator view and empty the Tracking panel (stops
        playback first). Training and searching are not touched."""
        if self.playing_active():
            self.play_stop.set()
        if self.preview_thread is not None and self.preview_thread.is_alive():
            self.preview_stop.set()
        self.intro_stop.set()
        self.frames.take()
        self._init_canvas()
        self.gameboy.show(None)
        self.clear_rounds()
        self._pending_stats.clear()
        self._pending_view = None
        self.agent_view.clear()
        for v in self.play_stat_vars.values():
            v.set("—")
        self.play_status_var.set("idle")
        if not self.busy():
            self.set_activity("none", "Nothing running")

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
        ttk.Separator(body).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        head = ttk.Frame(body)
        head.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(head, text="Observation setup", foreground=THEME.muted).pack(side="left")
        info_icon(head, "Filled in automatically from the model's run.json. Only change it "
                        "for models from older runs, and make it match how the model was "
                        "trained.").pack(side="left", padx=(6, 0))
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
        win.geometry(f"+{self.gameboy.winfo_rootx()}+{self.root.winfo_rooty() + 80}")

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
        _field(2, 1, "Metric:", metric_combo)
        self.tune_keep_best_var = tk.IntVar(value=9)
        keep_spin = ttk.Spinbox(cfg_frame, from_=0, to=100, width=6,
                                textvariable=self.tune_keep_best_var)
        _field(3, 0, "Keep best N trial runs:", keep_spin)
        tooltip(keep_spin, "Only the run folders of the best N configs are kept on disk; the "
                           "others are deleted as the sweep goes. 0 keeps all. Scores of "
                           "deleted runs stay in the results and in AIboy's experience.")
        # The metric's explanation follows the variable, whoever sets it (the
        # combobox, a loaded results file, the wizard), and shows on hover.
        self.tune_metric_tip = tooltip(
            metric_combo, lambda: tuning.METRIC_NOTES.get(self.tune_metric_var.get(), ""))
        self.tune_metric_var.trace_add("write", lambda *_a: self._on_metric_changed())
        tooltip(self.tune_preset_combo, "The preset every candidate starts from; the sweep "
                                        "changes only the listed knobs.")
        self.tune_skip_done_var = tk.BooleanVar(value=True)
        skip_cb = ttk.Checkbutton(
            cfg_frame, variable=self.tune_skip_done_var, command=self.tune_update_summary,
            text="Reuse known results")
        skip_cb.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        tooltip(skip_cb, "A candidate AIboy has already trained with the same settings and "
                         "trial length is scored from memory instead of trained again. This "
                         "also resumes an interrupted sweep.")
        self._tune_config_widgets.append(skip_cb)
        self.tune_baseline_var = tk.BooleanVar(value=True)
        base_cb = ttk.Checkbutton(
            cfg_frame, variable=self.tune_baseline_var, command=self.tune_update_summary,
            text="Include the base preset as candidate 1")
        base_cb.grid(row=4, column=2, columnspan=2, sticky="w", pady=(4, 0), padx=12)
        tooltip(base_cb, "The base preset itself is trained unchanged as the first candidate, "
                         "so a winner has to beat it, not just the other variations.")
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
                                            wraplength=600)
        self.tune_template_note.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 4))
        tooltip(self.btn_tune_suggest, "Fill the sweep with untested variations around the "
                                       "best settings AIboy knows for this preset.")

        self.tune_sweep_text = tk.Text(sweep_frame, height=5, wrap="word", font=MONO, undo=True)
        self.tune_sweep_text.grid(row=2, column=0, columnspan=4, sticky="ew")
        self.tune_sweep_text.bind("<<Modified>>", self._on_sweep_modified)
        self._tune_config_widgets.append(self.tune_sweep_text)
        tooltip(self.tune_sweep_text, "The sweep as JSON: either {knob: [values, …]} for a grid "
                                      "/ random sample, or a list of {knob: value} candidates. "
                                      "Edit it freely; the line below reports what it expands to.")

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
                                            font=MONO, wraplength=600)
        self.tune_summary_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self._tune_config_widgets.extend([rb_grid, rb_random, n_random_spin])

        # Progress, ETA and the trial's live numbers are under Tracking.
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.btn_tune_start = ttk.Button(ctrl, text="Start tuning", command=self.start_tuning)
        self.btn_tune_start.grid(row=0, column=0, padx=(0, 4))
        self.btn_tune_stop = ttk.Button(ctrl, text="Stop", command=self.stop_tuning, state="disabled")
        self.btn_tune_stop.grid(row=0, column=1, padx=4)
        self.btn_tune_load = ttk.Button(ctrl, text="Load results…", command=self._tune_load_results)
        self.btn_tune_load.grid(row=0, column=2, padx=4)
        tooltip(self.btn_tune_load, "Show the results of an earlier sweep from its file under "
                                    "models/<game>/_tune/.")
        self.tune_progress_var = tk.DoubleVar(value=0.0)
        self.tune_progress_text = tk.StringVar(value="—")

        res_frame = ttk.LabelFrame(parent, text="Results (best first)", padding=4)
        res_frame.grid(row=4, column=0, sticky="nsew")
        self.tune_tree = make_table(res_frame, [
            ("rank", "#", 30, "e", False), ("config", "config (overrides)", 190, "w", True),
            ("score", "score", 90, "e", False), ("ep_len", "ep len", 52, "e", False),
            ("steps", "steps", 66, "e", False), ("time", "time", 52, "e", False),
            ("runs", "runs", 110, "w", True),
        ], height=6)
        self.tune_tree.bind("<Double-1>", lambda e: self._tune_load_into_train())

        actions = ttk.Frame(parent)
        actions.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.btn_tune_save_preset = ttk.Button(actions, text="Save as preset…",
                                               command=self._tune_save_as_preset)
        self.btn_tune_save_preset.pack(side="left", padx=(0, 4))
        self.btn_tune_to_train = ttk.Button(actions, text="Load into Train tab",
                                            command=self._tune_load_into_train)
        self.btn_tune_to_train.pack(side="left", padx=4)
        tooltip(self.btn_tune_save_preset, "Keep the selected row's settings as a preset.")
        tooltip(self.btn_tune_to_train, "Put the selected row's settings on the Train tab "
                                        "(double-clicking a row does the same).")
        self.btn_tune_delete = ttk.Button(actions, text="Delete trial runs…",
                                          command=lambda: self.delete_tune_data(
                                              self.tune_run_prefix_var.get().strip() or "tune"))
        self.btn_tune_delete.pack(side="right")

        self.apply_tune_template()

    def _on_metric_changed(self) -> None:
        """The metric variable changed: explain it and re-rank the current
        results under it (a trace; the table may not exist yet at build time)."""
        metric = self.tune_metric_var.get()
        if hasattr(self, "tune_tree"):
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
            self.flash(f"{stale} result(s) come from an older file and keep their original metric")
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
        know = self.experience.knowledge_for(tuning.trial_config(base, {}, trial_steps))
        combos = self.experience.suggest(base, trial_steps, n_seeds, 6, tuning.METRIC_LATE, know)
        if not combos:
            self.flash("Nothing left to suggest: every nearby variation is already known.")
            return
        if know is not None and know.borrowed_from:
            around = tuning.describe_overrides(
                tuning.config_diff(know.config, tuning.full_config(base, {}))) or "the preset itself"
            note = (f"Nothing is known about this task yet; {len(combos)} untested variations "
                    f"around what worked best for the {know.borrowed_from} ({around}, "
                    f"{know.n_trials} trials, score {know.score:.0f}).")
        elif know is not None:
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
        why = check_start_level(plan["game"], plan["base"].get("start_level"))
        if why:
            messagebox.showerror("Tune", why)
            return False
        plan["source"] = source
        n_trials = len(plan["combos"]) * plan["n_seeds"]
        to_train = n_trials - self.known_trials(plan)
        if to_train > 40:
            eta = tuning.estimate_seconds(to_train, plan["trial_steps"], plan["base"],
                                          self.fps_for(plan["base"]))
            if not messagebox.askyesno(
                    "Tune", f"This sweep trains {to_train} trials "
                            f"(≈ {tuning.format_duration(eta)}). Start anyway?"):
                return False
        if self.playing_active():
            self.play_stop.set()

        self._tune_results.clear()
        self.render_tune_results()
        self.tune_tree.heading("score", text=plan["metric"])
        self.tune_progress_var.set(0)
        self.tune_progress_text.set(f"0/{n_trials}")
        for v in self.tune_vars.values():
            v.set("—")
        self.tune_vars["trial"].set(f"starting {n_trials} trials…")
        self.set_activity("tune", "Searching for better settings" if source == "wizard"
                          else f"Tuning: sweep '{plan['prefix']}'")
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
            reward, ep_len, fps = (format_stat(stats.get(k, "—"))
                                   for k in ("ep_rew_mean", "ep_len_mean", "fps"))
            self.stats_queue.put(("tune_live",
                f"{label} · {tuning.format_steps(steps)} / {tuning.format_steps(trial_steps)} "
                f"steps · average score {reward} · {fps} steps/s",
                {"steps": f"{tuning.format_steps(steps)} / {tuning.format_steps(trial_steps)}",
                 "reward": reward, "ep_len": ep_len, "fps": fps}))
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
        def _known(i: int, s: int) -> Record | None:
            if not plan["skip_done"]:
                return None
            return exp.find(tuning.trial_config(base, combos[i - 1], trial_steps, s))

        to_train = sum(0 if _known(i, s) else 1
                       for i in range(1, len(combos) + 1) for s in range(n_seeds))
        eta = tuning.SweepEta(to_train, trial_steps)
        self.stats_queue.put(("tune_known", total - to_train, total))
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
                    self.stats_queue.put(("tune_trial", label, tuning.plain_overrides(overrides)))
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
        if meta.get("metric") in tuning.METRICS:
            self.tune_metric_var.set(meta["metric"])     # note and heading follow (trace)
        self._tune_results = {r.index: r for r in results}
        if meta.get("sweep"):
            self.tune_sweep_text.delete("1.0", "end")
            self.tune_sweep_text.insert("1.0", json.dumps(meta["sweep"], indent=2))
        self.render_tune_results()
        self.flash(f"Loaded {len(results)} results from {Path(path).name}.")

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
        """Populate the Train tab from a preset-style dict. Optional fields
        missing from `cfg` fall back to PRESET_DEFAULTS so applying a preset
        is deterministic. Playback keeps following the selected model's own
        run.json; only a model without one (an older run) takes its
        observation setup from the Train tab."""
        self.form.set_config(cfg)
        self.refresh_models()
        if self.selected_model_config() is None:
            self.sync_play_options(cfg)
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

    def _resume_from(self, run_name: str) -> Path | None:
        """The checkpoint a start would continue from: the run's newest one
        if Resume is ticked and the run has any, else None (a fresh start)."""
        if not self.resume_var.get():
            return None
        return runs.latest_checkpoint(runs.run_paths(self.game, run_name)["checkpoints"])

    def _update_run_hint(self) -> None:
        if not hasattr(self, "run_hint_var"):
            return                                   # variable trace during construction
        name = self.run_name_var.get().strip() or "default"
        if not runs.is_run_name(name):
            self.run_hint_var.set("⚠ run names may not start with '_'")
            return
        paths = runs.run_paths(self.game, name)
        checkpoint = runs.latest_checkpoint(paths["checkpoints"])
        if checkpoint is None:
            exists = paths["base"].exists()
            self.run_hint_var.set(f"{paths['base']}/ " + ("(exists, no checkpoint yet: starts fresh)"
                                                          if exists else "(new run)"))
        elif self.resume_var.get():
            self.run_hint_var.set(f"{paths['base']}/ exists: continues from "
                                  f"{checkpoint.parent.name}/{checkpoint.name} and trains the "
                                  f"Timesteps above on top of what it has")
        else:
            self.run_hint_var.set(f"{paths['base']}/ exists: Resume is off, so training starts "
                                  f"over and replaces its models — choose a new name to keep them")

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
            self.model_var.set(labels[0] if labels else NO_MODEL)
            self._on_model_selected()
        if hasattr(self, "run_hint_var"):
            self._update_run_hint()

    def selected_model_config(self) -> dict | None:
        """The training settings recorded (run.json) for the model selected
        under "Watch an agent"; None when no model is selected or its run
        has no manifest."""
        if not hasattr(self, "model_var"):
            return None
        path = self._model_paths.get(self.model_var.get())
        if path is None:
            return None
        game_run = runs.run_of_model(path)
        return runs.read_run_config(*game_run) if game_run else None

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
        why = check_start_level(self.game, cfg.get("start_level"))
        if why:
            messagebox.showerror("Training", why)
            return False
        # Playback would compete for CPU and for the shared canvas.
        if self.playing_active():
            self.play_stop.set()
        run_name = self.run_name_var.get().strip() or "default"
        resume_from = self._resume_from(run_name)
        cmd = runs.build_train_cmd(cfg, run_name, resume=resume_from is not None,
                                   source="train")

        if resume_from is not None:
            self.append_log(f"[gui] resuming '{run_name}' from {resume_from}\n")
        elif self.resume_var.get():
            self.append_log(f"[gui] '{run_name}' has no checkpoint yet: starting fresh\n")
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
        self.train_elapsed_var.set("—")
        self.train_run_var.set(run_name)
        self._clear_run_details()
        self._train_preview_on = bool(self.preview_var.get())
        self.set_activity("train", f"Training '{run_name}'")
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
                continue
            m = EVAL_LINE.search(line)
            if m:
                self.stats_queue.put(("eval", int(m.group(1)), float(m.group(2)),
                                      float(m.group(3))))
                continue
            m = EVAL_LEN_LINE.search(line)
            if m:
                self.stats_queue.put(("eval_len", float(m.group(1))))
            elif line.strip() == EVAL_BEST_LINE:
                self.stats_queue.put(("eval_best",))
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
        self.gameboy.set_power(True)
        self.clear_rounds()
        self.set_live_mode("agent")
        self._pending_stats.clear()
        self._pending_view = None
        self.agent_view.clear()
        for v in self.play_stat_vars.values():
            v.set("—")
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

    def claim_screen(self) -> bool:
        """Make sure no emulator thread still owns the screen before a new
        playback starts: the thread of a playback that has already ended,
        or the live preview of a finished training run, gets a moment to
        close its emulator. False while a playback is still in progress."""
        if self.playing_active():
            return False
        for thread, stop in ((self.play_thread, None), (self.preview_thread, self.preview_stop)):
            if thread is not None and thread.is_alive():
                if stop is not None:
                    stop.set()
                thread.join(timeout=5)
                if thread.is_alive():
                    return False
        return True

    def start_playing(self) -> bool:
        """Play the selected model in the embedded canvas. True if started."""
        if self.busy():
            messagebox.showwarning(
                "Play", "Training or tuning is running. Stop it first, or wait for it to finish.")
            return False
        if not self.claim_screen():
            messagebox.showwarning("Play", "Playback already running.")
            return False
        ok, why = self.game_runnable()
        if not ok:
            messagebox.showerror("Play", why)
            return False
        label = self.model_var.get()
        if label not in self._model_paths:
            self.refresh_models()
            label = self.model_var.get()
        if label not in self._model_paths:
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
        why = check_start_level(self.game, raw_level)
        if why:
            messagebox.showerror("Play", why)
            return False

        self._begin_playback("agent")
        self.set_activity("play", f"Agent playing: {label}")
        self.play_status_var.set(f"loading {model_path.name}…")
        self.play_thread = self.player.play(
            model_path=model_path, game=self.game, obs_type=self.play_obs_type_var.get(),
            action_repeat=action_repeat, frame_stack=frame_stack,
            start_level=None if raw_level == "default" else raw_level,
            episodes=episodes, max_steps=max_steps,
            deterministic=not self.play_stochastic_var.get(),
            speed_mult=dict(SPEED_CHOICES)[self.play_speed_label_var.get()],
            stop=self.play_stop,
            time_budget=self.play_time_budget, stall_steps=self.play_stall_steps,
        )
        return True

    def stop_playing(self) -> None:
        self.play_stop.set()

    def _begin_playback(self, mode: str) -> None:
        """Take the screen for a playback of `mode` ("agent" / "human")."""
        self.intro_stop.set()
        self.play_stop.clear()
        self._play_open = True
        self.play_mode = mode
        self.set_live_mode(mode)
        self.gameboy.set_power(True)
        self.btn_play_start.config(state="disabled")
        self.btn_play_stop.config(state="normal")
        self.btn_human.config(state="disabled" if mode == "agent" else "normal",
                              text="■ Stop playing" if mode == "human" else "🎮 Play yourself")
        self.clear_rounds()
        self._pending_stats.clear()
        self._pending_view = None
        self.agent_view.clear()
        for v in self.play_stat_vars.values():
            v.set("—")

    # ---------- Playing yourself ----------

    def toggle_human_play(self) -> None:
        if self.playing_active() and self.play_mode == "human":
            self.stop_playing()
        else:
            self.start_human_play()

    def start_human_play(self) -> bool:
        """Run the plain game on the Game Boy with the person's own input
        (keyboard on the picture, clicks on its buttons, a game controller).
        True if started."""
        if self.busy():
            messagebox.showwarning(
                "Play", "Training or tuning is running. Stop it first, or wait for it to finish.")
            return False
        if not self.claim_screen():
            messagebox.showwarning("Play", "Playback already running. Stop it first.")
            return False
        info = self.current_rom()
        if info is None:
            messagebox.showerror("Play", f"No ROM for '{self.game}' in ROMs/.")
            return False
        if not info.playable:
            messagebox.showerror("Play", f"'{self.game}' cannot boot: {info.error}")
            return False
        start_level = None
        choice = self.human_start_var.get()
        if self.game == DEFAULT_GAME and choice != HUMAN_FROM_START:
            w, l = choice.split("-")
            start_level = (int(w), int(l))
        self._begin_playback("human")
        self.set_activity("human", f"You play {display_name(self.game)}")
        self.play_status_var.set("starting…")
        self.held.clear()
        self.gamepad.remap()          # a button already held on the pad counts from the start
        self.keyboard.set_enabled(True)
        self.gamepad.active = True
        self.gameboy.focus_set()
        self.play_thread = self.player.play_human(
            game=self.game, rom_path=info.path, held=self.held, stop=self.play_stop,
            start_level=start_level)
        return True

    # ---------- Event loop ----------

    def _pump(self) -> None:
        if self._closing:
            return
        try:
            while True:
                self._handle_event(self.stats_queue.get_nowait())
        except queue.Empty:
            pass
        if self._pending_stats:
            # Every label change redraws the window (all of it, on macOS), so
            # the live numbers are applied once per pump, newest values only.
            for key, val in self._pending_stats.items():
                if self.play_stat_vars[key].get() != val:
                    self.play_stat_vars[key].set(val)
            self._pending_stats.clear()
        if self._pending_view is not None:
            grid, self._pending_view = self._pending_view, None
            if self._track_shown.get("view"):
                self._show_agent_view(grid[0])

        latest = self.frames.take()
        if latest is not None:
            self.gameboy.set_screen(
                Image.fromarray(dmg_tint(latest)).resize(self.gameboy.screen_size, Image.NEAREST))
            self.frames_painted += 1
        if self._play_open and self.stats_queue.empty() and not (
                self.play_thread is not None and self.play_thread.is_alive()):
            # The playback thread is gone without its end event (it reports
            # before it returns, so this is a safety net): close the playback
            # here, or the Play buttons would stay locked.
            self.play_status_var.set("ended")
            self._play_finished()

        self._update_status_bar()
        self._layout_tracking()
        try:
            self.root.after(PUMP_MS, self._pump)
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
                    f"average score {self.stat_vars['ep_rew_mean'].get()} · "
                    f"{self.stat_vars['fps'].get()} steps/s")
        elif self.tuning_active():
            text = (f"Searching · {self.tune_progress_text.get()} trials · "
                    f"{self.tune_vars['trial'].get()}: {self.tune_vars['trying'].get()}")
        elif self.playing_active():
            who = "You play" if self.play_mode == "human" else "Agent playing"
            text = f"{who} · {self.play_status_var.get()}"
        else:
            text = "Nothing running"
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
                self._init_canvas()
        elif kind == "disk_usage":
            _, n_bytes, n_runs = item
            self.disk_var.set(f"models/ {runs.format_size(n_bytes)} · {n_runs} run(s)")
        elif kind == "log":
            self.append_log(item[1])
        elif kind == "stat":
            _, key, val = item
            if key in self.stat_vars:
                self.stat_vars[key].set(val if key == "total_timesteps" else format_stat(val))
            if key == "total_timesteps":
                self._update_train_progress(val)
            elif key == "time_elapsed":
                try:
                    self.train_elapsed_var.set(format_elapsed(float(val)))
                except ValueError:
                    self.train_elapsed_var.set(val)
        elif kind == "eval":
            self._add_eval(item[1], item[2], item[3])
        elif kind == "eval_len":
            self._update_last_eval(length=item[1])
        elif kind == "eval_best":
            self._update_last_eval(best=True)
        elif kind == "train_done":
            self._on_train_done(item[1])
        elif kind == "play_status":
            self.play_status_var.set(item[1])
        elif kind == "preview_done":
            # The live preview ended; a playback that took over the screen
            # meanwhile (the wizard's Watch step) keeps its status and LED.
            if not self.playing_active():
                self.play_status_var.set("idle")
                self.gameboy.show(None)
                self.gameboy.set_power(False)
        elif kind == "play_stat":
            _, key, val = item
            if key in self.play_stat_vars:
                self._pending_stats[key] = val
            if key == "action":
                self.gameboy.show(val)
        elif kind == "agent_view":
            self._pending_view = (item[1],)     # a tuple: None is a valid grid (pixel model)
        elif kind == "play_stats":
            for key, val in item[1].items():
                if key in self.play_stat_vars:
                    self._pending_stats[key] = val
            if "action" in item[1]:
                self.gameboy.show(item[1]["action"])
        elif kind == "play_episode":
            r = item[1]
            self.add_round(r)
            self.append_log(f"[play] round {r['episode']}: score {r['reward']:.0f}, "
                            f"{r['steps']} steps, {r['end']}\n")
        elif kind == "play_done":
            # Keep the last report ("Round 2: … died in 1-2", "you played…") visible.
            last = self.play_status_var.get()
            if last.startswith("Round"):
                self.play_status_var.set(f"done · {last}")
            elif not last.startswith("you played"):
                self.play_status_var.set("done")
            if self.play_mode == "human" and item[1]:
                self.append_log(f"[play] you: {item[1][-1]['end']}\n")
            head = self.activity_var.get()
            self.activity_var.set(head.replace("Agent playing:", "Agent played:", 1)
                                  .replace("You play ", "You played ", 1))
            self._play_finished()
            if self.wizard is not None and self.play_mode == "agent":
                self.wizard.on_play_done(item[1])
        elif kind == "gamepad":
            name = item[1]
            if self.gamepad.error:
                self.gamepad_var.set(GAMEPAD_OFF)
                self.gamepad_label.config(foreground=THEME.err)
                self.flash(f"Game controllers are off: {self.gamepad.error}", seconds=12)
            else:
                self.gamepad_var.set(name or NO_GAMEPAD)
                self.gamepad_label.config(foreground=THEME.ok if name else THEME.muted)
                self.flash(f"Game controller connected: {name}" if name
                           else "Game controller disconnected.")
            if self._controls_dialog is not None:
                self._controls_dialog.refresh()
        elif kind == "play_error":
            self.play_status_var.set("error")
            self._play_finished()
            if self.wizard is not None and self.play_mode == "agent":
                self.wizard.on_play_error(item[1])
            messagebox.showerror("Play error", item[1])
        elif kind == "tune_live":
            for key, val in item[2].items():
                self.tune_vars[key].set(val)
        elif kind == "tune_known":
            _, known, total = item
            self._tune_known = (known, total)
            self.tune_vars["trial"].set(f"{total} trials" + (
                f", {known} already known" if known else ""))
        elif kind == "tune_trial":
            _, label, settings_text = item
            known = getattr(self, "_tune_known", (0, 0))[0]
            self.tune_vars["trial"].set(label + (f" ({known} known)" if known else ""))
            self.tune_vars["trying"].set(settings_text)
            for key in ("steps", "reward", "ep_len", "fps"):
                self.tune_vars[key].set("—")
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
            best = tuning.best_result(self.tune_results())
            if best is not None:
                self.tune_vars["best"].set(
                    f"{tuning.plain_overrides(best.overrides)} (score {best.score_text()})")
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
            self.tune_vars["trying"].set("—")
            self.tune_vars["trial"].set("stopped" if cancelled else "all done")
            head = self.activity_var.get()
            self.activity_var.set(
                ("Search stopped" if cancelled else "Search finished") if head.startswith("Search")
                else head + (" — stopped" if cancelled else " — finished"))
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
            status = f"failed (rc={rc}) — see Messages on the Train tab"
            self.train_progress_text.set("failed")
        self.stat_vars["status"].set(status)
        self.activity_var.set(f"Training '{self._train_run_name}' — {status.split(' (')[0]}")
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
            self.flash(f"Run '{self._train_run_name}' {status} — its best model is selected "
                       f"under Tracking, click ▶ Play to watch it.", seconds=12)
        if self.wizard is not None:
            self.wizard.on_train_done(rc, self._train_stop_requested)

    def _play_finished(self) -> None:
        """The playback reported its end (`play_done` / `play_error`, sent
        once its emulator is closed): put the screen and the inputs to rest
        and free the start buttons."""
        self._play_open = False
        self.gameboy.show(None)
        self.gameboy.set_power(self.preview_thread is not None and self.preview_thread.is_alive())
        self.btn_play_stop.config(state="disabled")
        self.keyboard.set_enabled(False)
        self.gamepad.active = False
        self.held.clear()
        self.btn_human.config(text="🎮 Play yourself")
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
        """Serve models/<game>/ (every run, incl. tune trials) and open a browser tab.

        TensorBoard runs as `main.py tensorboard` (the built app: the app
        itself with that sub-command), so it is the copy this program was
        installed or built with; nothing on the computer's PATH is needed.
        Its output goes to TENSORBOARD_LOG; the browser opens once the port
        answers (`_tensorboard_poll`), and a start-up failure is shown."""
        url = f"http://localhost:{TENSORBOARD_PORT}"
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            webbrowser.open(url)
            return
        logdir = PROJECT_DIR / "models" / self.game
        if importlib.util.find_spec("tensorboard") is None:
            messagebox.showinfo("TensorBoard",
                                "TensorBoard is not installed on this computer. Install it with "
                                "`pip install tensorboard` and click again, or read the training "
                                f"curves later from {logdir}.")
            return
        if self._port_answers(TENSORBOARD_PORT):
            # Somebody else's server (an earlier TensorBoard that was never
            # closed?); starting ours would fail, and the readiness poll
            # could not tell the two apart.
            messagebox.showerror("TensorBoard",
                                 f"Port {TENSORBOARD_PORT} is already in use by another program "
                                 f"(an earlier TensorBoard?). Close it and click again.")
            return
        cmd = app_command("tensorboard", "--logdir", str(logdir), "--port", str(TENSORBOARD_PORT))
        try:
            with TENSORBOARD_LOG.open("w") as log:      # a file, not a pipe: nobody would drain it
                self.tb_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                cwd=str(PROJECT_DIR))
        except OSError as e:
            messagebox.showerror("TensorBoard", f"Could not start TensorBoard:\n{e}")
            return
        self.append_log(f"[gui] {' '.join(cmd)} → {url}\n")
        self._tb_started_at = time.monotonic()
        self.root.after(500, lambda: self._tensorboard_poll(url))

    def _tensorboard_poll(self, url: str) -> None:
        """Open the browser as soon as TensorBoard answers on its port. If the
        process quit first (port taken, package broken) or nothing answers
        within TENSORBOARD_START_S, say why instead of opening a tab on
        nothing."""
        proc = self.tb_proc
        if self._closing or proc is None:
            return
        if proc.poll() is not None:
            why = self._tensorboard_output()
            self.append_log(f"[gui] tensorboard exited with {proc.returncode}: {why}\n")
            messagebox.showerror("TensorBoard",
                                 f"TensorBoard stopped right after starting:\n{why[-1500:]}\n\n"
                                 f"(the full output is in {TENSORBOARD_LOG.name})")
            return
        if not self._port_answers(TENSORBOARD_PORT):
            if time.monotonic() - self._tb_started_at > TENSORBOARD_START_S:
                messagebox.showerror("TensorBoard",
                                     f"TensorBoard did not answer on port {TENSORBOARD_PORT} within "
                                     f"{TENSORBOARD_START_S} s; see {TENSORBOARD_LOG.name}.")
                return
            self.root.after(500, lambda: self._tensorboard_poll(url))
            return
        self.append_log(f"[gui] tensorboard is up: {url}\n")
        webbrowser.open(url)

    @staticmethod
    def _port_answers(port: int) -> bool:
        try:
            with socket.create_connection(("localhost", port), timeout=0.2):
                return True
        except OSError:
            return False

    @staticmethod
    def _tensorboard_output() -> str:
        try:
            return TENSORBOARD_LOG.read_text(errors="replace").strip()
        except OSError:
            return ""

    def append_log(self, text: str) -> None:
        """Keep `text` in the raw log; everything but the trainer's stats
        tables and evaluation lines also goes to the Messages box."""
        self.log_text.insert("end", text)
        self.log_text.see("end")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 2000:
            self.log_text.delete("1.0", "500.0")
        if not is_noise_line(text) and not text.startswith("$ "):     # the command line: log only
            self.note(text)

    def _on_close(self) -> None:
        self._closing = True
        self.play_stop.set()
        self.preview_stop.set()
        self.tune_stop.set()
        # The emulator thread closes its PyBoy (and its SDL audio device)
        # before the controller thread shuts SDL down.
        for thread in (self.play_thread, self.preview_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=3)
        self.gamepad.stop()
        for proc in (self.tune_proc, self.train_proc):
            if proc is not None and proc.poll() is None:
                interrupt(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    kill_if_alive(proc)
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            self.tb_proc.terminate()
        self.gamepad.join(timeout=2)    # SDL shuts down on its own thread before the process ends
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
