"""Tkinter GUI for gameboyEnv: train (subprocess) or play (embedded canvas).

- Training launches `python main.py train ...` as a subprocess and streams stats.
- Play runs the emulator with rendering enabled but no SDL2 window, grabbing
  frames via `screen_ndarray()` and painting them to an embedded Canvas.
  This avoids the macOS "hidden window" problem entirely.
"""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from env import GAMES, MarioEnv

SCALE = 3
GAME_W, GAME_H = 160, 144
CANVAS_W, CANVAS_H = GAME_W * SCALE, GAME_H * SCALE

# Stat lines in SB3's dashboard output: "|    ep_rew_mean     | 850.3    |"
STAT_LINE = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([\S]+)\s+\|")
TRACKED_STATS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "fps", "time_elapsed")


class GameBoyAIGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Game Boy AI — Train & Play")
        root.geometry("980x680")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.project_dir = Path(__file__).resolve().parent

        # Cross-thread channels
        self.stats_queue: queue.Queue = queue.Queue()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Training subprocess state
        self.train_proc: subprocess.Popen | None = None

        # Play thread state
        self.play_stop = threading.Event()
        self.play_thread: threading.Thread | None = None

        self._closing = False
        self._build_ui()
        self._pump()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Game:").pack(side="left")
        self.game_var = tk.StringVar(value="mario")
        ttk.Combobox(top, textvariable=self.game_var, values=sorted(GAMES),
                     state="readonly", width=10).pack(side="left", padx=6)
        ttk.Label(top, text="Run name (blank = same as game):").pack(side="left", padx=(20, 4))
        self.run_name_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.run_name_var, width=16).pack(side="left")

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

        self.tsteps_var = tk.IntVar(value=200_000)
        self.n_envs_var = tk.IntVar(value=4)
        self.ent_coef_var = tk.DoubleVar(value=0.05)
        self.resume_var = tk.BooleanVar(value=False)

        row = 0
        def add(label, widget):
            nonlocal row
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3, padx=6)
            row += 1

        add("Total timesteps:", ttk.Spinbox(parent, from_=1000, to=20_000_000,
                                             increment=10_000, textvariable=self.tsteps_var, width=18))
        add("Parallel envs:", ttk.Spinbox(parent, from_=1, to=16,
                                           textvariable=self.n_envs_var, width=18))
        add("Entropy coef:", ttk.Spinbox(parent, from_=0.0, to=0.5,
                                          increment=0.01, textvariable=self.ent_coef_var, width=18))
        ttk.Checkbutton(parent, text="Resume from newest checkpoint",
                        variable=self.resume_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1

        btns = ttk.Frame(parent)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        self.btn_train_start = ttk.Button(btns, text="Start training", command=self.start_training)
        self.btn_train_start.pack(side="left", padx=4)
        self.btn_train_stop = ttk.Button(btns, text="Stop", command=self.stop_training, state="disabled")
        self.btn_train_stop.pack(side="left", padx=4)
        row += 1

        # Live stats
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

        # Log
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
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        self.play_episodes_var = tk.IntVar(value=3)
        self.play_max_steps_var = tk.IntVar(value=2000)
        self.play_stochastic_var = tk.BooleanVar(value=False)
        self.play_speed_var = tk.IntVar(value=0)  # 0 = as fast as GUI can pump

        row = 0
        def add(label, widget):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3, padx=6)
            row += 1

        add("Episodes:", ttk.Spinbox(left, from_=1, to=100, textvariable=self.play_episodes_var, width=12))
        add("Max steps (0=∞):", ttk.Spinbox(left, from_=0, to=1_000_000, increment=100,
                                              textvariable=self.play_max_steps_var, width=12))
        ttk.Checkbutton(left, text="Stochastic (sample actions)",
                        variable=self.play_stochastic_var).grid(row=row, column=0, columnspan=2,
                                                                sticky="w", pady=6)
        row += 1

        btns = ttk.Frame(left)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        self.btn_play_start = ttk.Button(btns, text="Start playing", command=self.start_playing)
        self.btn_play_start.pack(fill="x", pady=2)
        self.btn_play_stop = ttk.Button(btns, text="Stop", command=self.stop_playing, state="disabled")
        self.btn_play_stop.pack(fill="x", pady=2)
        row += 1

        self.play_status_var = tk.StringVar(value="idle")
        ttk.Label(left, textvariable=self.play_status_var, font=("Menlo", 10),
                  wraplength=200).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Game canvas
        canvas_frame = ttk.LabelFrame(right, text=f"Game Boy ({SCALE}x)", padding=6)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H,
                                bg="#222", highlightthickness=0)
        self.canvas.pack()
        placeholder = np.full((GAME_H, GAME_W, 3), 34, dtype=np.uint8)
        self._tk_img = ImageTk.PhotoImage(
            Image.fromarray(placeholder).resize((CANVAS_W, CANVAS_H), Image.NEAREST)
        )
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

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
            "--run-name", run_name,
            "--device", "auto",
        ]
        if self.resume_var.get():
            cmd.append("--resume")

        self._append_log(f"$ {' '.join(cmd)}\n")
        self.train_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=str(self.project_dir),
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
        run_name = self.run_name_var.get().strip() or game
        model_path = self._resolve_model(run_name)
        if model_path is None:
            messagebox.showerror(
                "Play",
                f"No trained model found for run '{run_name}'.\n"
                f"Train first, or place a .zip at logs/{run_name}/best_model.zip"
            )
            return

        self.play_stop.clear()
        self.btn_play_start.config(state="disabled")
        self.btn_play_stop.config(state="normal")
        self.play_status_var.set(f"loading {model_path.name}...")
        self.play_thread = threading.Thread(
            target=self._play_loop, args=(game, model_path), daemon=True
        )
        self.play_thread.start()

    def stop_playing(self) -> None:
        self.play_stop.set()

    def _resolve_model(self, run_name: str) -> Path | None:
        for c in (Path("logs") / run_name / "best_model.zip",
                  Path("checkpoints") / run_name / "final.zip"):
            if c.exists():
                return c
        ckpt_dir = Path("checkpoints") / run_name
        if ckpt_dir.exists():
            zips = sorted(ckpt_dir.glob("ppo_*.zip"))
            if zips:
                return zips[-1]
        return None

    def _play_loop(self, game: str, model_path: Path) -> None:
        try:
            spec = GAMES[game]
            rom_path = self.project_dir / "ROMs" / spec.rom_file
            pyboy = PyBoy(str(rom_path), window_type="null",
                          game_wrapper=True, disable_renderer=False)
            pyboy.set_emulation_speed(0)

            if game == "mario":
                base = MarioEnv(pyboy, frame_skip=4)
            else:
                base = pyboy.openai_gym(observation_type="raw", action_type="press")

            monitored = Monitor(base)
            vec = DummyVecEnv([lambda env=monitored: env])
            vec = VecTransposeImage(vec)
            vec = VecFrameStack(vec, n_stack=4, channels_order="first")

            model = PPO.load(str(model_path), env=vec, device="cpu")
            deterministic = not self.play_stochastic_var.get()
            episodes = int(self.play_episodes_var.get())
            max_steps = int(self.play_max_steps_var.get())

            self.stats_queue.put(("play_status", f"playing — model={model_path.name}"))
            for ep in range(episodes):
                if self.play_stop.is_set():
                    break
                obs = vec.reset()
                self._push_frame(pyboy)
                total = 0.0
                steps = 0
                done = [False]
                while not done[0] and not self.play_stop.is_set():
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, _info = vec.step(action)
                    total += float(reward[0])
                    steps += 1
                    self._push_frame(pyboy)
                    if max_steps and steps >= max_steps:
                        break
                self.stats_queue.put((
                    "play_status",
                    f"Episode {ep + 1}/{episodes}: reward={total:.1f} steps={steps}",
                ))

            vec.close()
            self.stats_queue.put(("play_done", None))
        except Exception:
            import traceback
            self.stats_queue.put(("play_error", traceback.format_exc()))

    def _push_frame(self, pyboy: PyBoy) -> None:
        arr = pyboy.botsupport_manager().screen().screen_ndarray()
        try:
            self.frame_queue.put_nowait(np.asarray(arr, dtype=np.uint8))
        except queue.Full:
            pass  # drop frame; GUI will pull the next one

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
                    key, val = item[1], item[2]
                    if key in self.stat_labels:
                        self.stat_labels[key].config(text=val)
                elif kind == "train_done":
                    rc = item[1]
                    self.stat_labels["status"].config(text=f"finished (rc={rc})")
                    self.btn_train_start.config(state="normal")
                    self.btn_train_stop.config(state="disabled")
                    self.train_proc = None
                elif kind == "play_status":
                    self.play_status_var.set(item[1])
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
            self.root.after(33, self._pump)  # ~30 fps GUI refresh
        except tk.TclError:
            pass  # root was destroyed between scheduling and this call

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")
        # Trim very old lines to keep GUI snappy
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


def run() -> None:
    root = tk.Tk()
    GameBoyAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
