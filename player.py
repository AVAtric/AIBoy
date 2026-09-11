"""Embedded playback: run a trained model in a windowless PyBoy and stream
frames + stats to the GUI.

The emulator renders into memory (`window_type="null"`, renderer on); every
emulator tick pushes the screen into `frame_queue`, and per-step numbers go
into `event_queue` using the GUI's `_pump` protocol:

    ("log", text)                 training-log line
    ("stat", key, value)          Train-tab live stat (preview only)
    ("play_status", text)         Play-tab status line
    ("play_stat", key, value)     Play-tab live-episode panel
    ("play_done", summary)        playback finished; summary = list of
                                  {"reward", "steps"} per completed episode
    ("play_error", traceback)

Pacing is done with `time.sleep()` because PyBoy's own speed setting only
paces when it owns an SDL2 window.

Two entry points share one engine:
  - `play()`     plays a fixed model for N episodes (Play tab, wizard step 4)
  - `preview()`  follows a training run, reloading `best_model.zip` whenever
                 it changes, until training ends (Train tab live preview)
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import GAMES, ROM_DIR, MarioEnv, prepare_level_states, wrap_vec_env

GB_FPS = 60.0
PREVIEW_STEP_SLEEP = 0.02   # cap the preview at ~50 env-steps/s so training keeps the CPU

# Canvas geometry and speed presets shared by the Play tab and the wizard.
SCALE = 3
GAME_W, GAME_H = 160, 144
CANVAS_W, CANVAS_H = GAME_W * SCALE, GAME_H * SCALE
SPEED_CHOICES = [("0.5×", 0.5), ("1× (real time)", 1.0), ("2×", 2.0),
                 ("4×", 4.0), ("Unlimited", 0.0)]


class _Session:
    """One emulator + wrapped vec-env for a given observation setup."""

    def __init__(self, game: str, obs_type: str, action_repeat: int, frame_stack: int,
                 start_level, frame_queue: queue.Queue):
        spec = GAMES[game]
        rom_path = ROM_DIR / spec.rom_file
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}")
        self.game = game
        self.action_repeat = action_repeat
        # Rendering stays on even for tile obs: the canvas shows pixels.
        self.pyboy = PyBoy(str(rom_path), window_type="null", game_wrapper=True,
                           disable_renderer=False)
        self.pyboy.set_emulation_speed(0)
        self._frame_queue = frame_queue

        if game == "mario":
            base = MarioEnv(self.pyboy, frame_skip=action_repeat, obs_type=obs_type,
                            tick_callback=self.grab_frame, start_level=start_level)
        else:
            if self.pyboy.game_wrapper() is None:
                self.pyboy.stop(save=False)
                raise RuntimeError(f"Game '{game}' has no PyBoy game_wrapper and cannot be played.")
            if obs_type != "pixels":
                self.pyboy.stop(save=False)
                raise RuntimeError(f"obs_type={obs_type!r} is only supported for mario")
            base = self.pyboy.openai_gym(observation_type="raw", action_type="press")
        self.vec = wrap_vec_env(DummyVecEnv([lambda env=Monitor(base): env]),
                                obs_type, frame_stack)

    def grab_frame(self) -> None:
        arr = self.pyboy.botsupport_manager().screen().screen_ndarray()
        try:
            self._frame_queue.put_nowait(np.asarray(arr, dtype=np.uint8))
        except queue.Full:
            pass

    def close(self) -> None:
        try:
            self.vec.close()   # closes MarioEnv -> pyboy.stop()
        except Exception:
            pass


class EmbeddedPlayer:
    def __init__(self, frame_queue: queue.Queue, event_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.events = event_queue

    # ---------- public entry points ----------

    def play(self, *, model_path: Path, game: str, obs_type: str, action_repeat: int,
             frame_stack: int, start_level, episodes: int, max_steps: int,
             deterministic: bool, speed_mult: float, stop: threading.Event) -> threading.Thread:
        t = threading.Thread(
            target=self._play_loop, daemon=True,
            args=(model_path, game, obs_type, action_repeat, frame_stack, start_level,
                  episodes, max_steps, deterministic, speed_mult, stop))
        t.start()
        return t

    def preview(self, *, model_path: Path, game: str, obs_type: str, action_repeat: int,
                frame_stack: int, start_level, stop: threading.Event,
                training_active: Callable[[], bool]) -> threading.Thread:
        t = threading.Thread(
            target=self._preview_loop, daemon=True,
            args=(model_path, game, obs_type, action_repeat, frame_stack, start_level,
                  stop, training_active))
        t.start()
        return t

    # ---------- shared helpers ----------

    def _emit(self, *item) -> None:
        self.events.put(item)

    def _report_step(self, game: str, action, info, ep_reward: float, ep_steps: int,
                     mirror_train_stats: bool) -> None:
        act_id = int(np.asarray(action).flat[0])
        names = MarioEnv.ACTION_NAMES
        action_name = names[act_id] if game == "mario" and act_id < len(names) else str(act_id)
        self._emit("play_stat", "action", action_name)
        self._emit("play_stat", "reward", f"{ep_reward:.1f}")
        self._emit("play_stat", "steps", str(ep_steps))
        i0 = info[0] if info and isinstance(info[0], dict) else {}
        if "x" in i0:
            self._emit("play_stat", "x", f"{i0['x']} (max {i0.get('max_x', '?')})")
        if "world" in i0:
            w = i0["world"]
            world = f"{w[0]}-{w[1]}"
            self._emit("play_stat", "world", world)
            if mirror_train_stats:
                self._emit("stat", "world", world)
        for key in ("lives", "coins"):
            if key in i0:
                self._emit("play_stat", key, str(i0[key]))
                if mirror_train_stats:
                    self._emit("stat", key, str(i0[key]))
        if mirror_train_stats and "max_x" in i0:
            self._emit("stat", "max_x", str(i0["max_x"]))

    # ---------- play ----------

    def _play_loop(self, model_path, game, obs_type, action_repeat, frame_stack, start_level,
                   episodes, max_steps, deterministic, speed_mult, stop) -> None:
        session = None
        summary: list[dict] = []
        try:
            if game == "mario":
                # Subprocess with timeout, so a level that cannot boot cannot
                # wedge the GUI thread.
                prepare_level_states(start_level)
            session = _Session(game, obs_type, action_repeat, frame_stack, start_level,
                               self.frame_queue)
            self._emit("play_status", f"loaded {model_path.name}")
            model = PPO.load(str(model_path), env=session.vec, device="cpu")
            # Target wall-clock time per env step; 0 = no throttle.
            step_period = (action_repeat / GB_FPS) / speed_mult if speed_mult > 0 else 0.0

            for ep in range(episodes):
                if stop.is_set():
                    break
                obs = session.vec.reset()
                session.grab_frame()
                total, steps, done = 0.0, 0, [False]
                self._emit("play_stat", "episode", f"{ep + 1}/{episodes}")
                while not done[0] and not stop.is_set():
                    t0 = time.time()
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = session.vec.step(action)
                    total += float(reward[0])
                    steps += 1
                    self._report_step(game, action, info, total, steps, mirror_train_stats=False)
                    if max_steps and steps >= max_steps:
                        break
                    if step_period > 0:
                        remaining = step_period - (time.time() - t0)
                        if remaining > 0:
                            time.sleep(remaining)
                if stop.is_set():
                    break
                summary.append({"reward": total, "steps": steps})
                self._emit("play_status", f"Episode {ep + 1} done: reward={total:.1f} steps={steps}")
            self._emit("play_done", summary)
        except Exception:
            self._emit("play_error", traceback.format_exc())
        finally:
            if session is not None:
                session.close()

    # ---------- preview ----------

    def _preview_loop(self, model_path, game, obs_type, action_repeat, frame_stack, start_level,
                      stop, training_active) -> None:
        session = None
        model = None
        model_mtime = 0.0
        ep_num, ep_reward, ep_steps = 1, 0.0, 0
        obs = None
        self._emit("log", f"[preview] waiting for first best_model at {model_path}\n")
        try:
            while not stop.is_set() and training_active():
                if not model_path.exists():
                    time.sleep(2)
                    continue
                mtime = model_path.stat().st_mtime
                if model is None or mtime > model_mtime:
                    try:
                        if session is not None:
                            session.close()
                        session = _Session(game, obs_type, action_repeat, frame_stack,
                                           start_level, self.frame_queue)
                        model = PPO.load(str(model_path), env=session.vec, device="cpu")
                        model_mtime = mtime
                        obs = session.vec.reset()
                        ep_num, ep_reward, ep_steps = 1, 0.0, 0
                        self._emit("log", f"[preview] loaded {model_path.name}\n")
                        self._emit("play_status", f"preview — {model_path.name}")
                        self._emit("play_stat", "episode", f"{ep_num} (preview)")
                    except Exception as e:
                        # The trainer may still be writing the zip; retry shortly.
                        self._emit("log", f"[preview] load error: {e}\n")
                        model = None
                        time.sleep(3)
                        continue

                action, _ = model.predict(obs, deterministic=False)
                obs, r, done, info = session.vec.step(action)
                ep_reward += float(r[0])
                ep_steps += 1
                self._report_step(game, action, info, ep_reward, ep_steps, mirror_train_stats=True)
                if done[0]:
                    obs = session.vec.reset()
                    ep_num += 1
                    ep_reward, ep_steps = 0.0, 0
                    self._emit("play_stat", "episode", f"{ep_num} (preview)")
                time.sleep(PREVIEW_STEP_SLEEP)
        except Exception:
            self._emit("log", f"[preview] error:\n{traceback.format_exc()}\n")
        finally:
            if session is not None:
                session.close()
            self._emit("log", "[preview] stopped\n")
            self._emit("play_status", "idle")
