"""Embedded playback: run a trained model in a windowless PyBoy and stream
frames + stats to the GUI.

The emulator renders into memory (`window_type="null"`, renderer on); every
emulator tick stores the screen in a `LatestFrame` slot (the GUI paints the
newest frame it finds there), and per-step numbers go into `event_queue`
using the GUI's `_pump` protocol:

    ("log", text)                 training-log line
    ("play_status", text)         screen-panel status line
    ("play_stat", key, value)     screen-panel live-episode stats
    ("play_done", summary)        playback finished; summary = list of
                                  {"reward", "steps"} per completed episode
    ("play_error", traceback)

Pacing is done per emulator frame with `time.sleep()` because PyBoy's own
speed setting only paces when it owns an SDL2 window. Per-frame (not
per-step) pacing matters: a jump step holds the button for 10 frames while
a walk step takes 4, so pacing per step would make jumps run 2.5x too fast
and the canvas would only ever show the last frame of each step.

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

from games import GAMES, ROM_DIR, prepare_level_states

GB_FPS = 60.0
PREVIEW_STEP_SLEEP = 0.02   # cap the preview at ~50 env-steps/s so training keeps the CPU
RESYNC_AFTER = 0.25         # if pacing falls this far behind, drop the backlog instead of racing


class LatestFrame:
    """Single-slot, thread-safe hand-off of the newest emulator frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def take(self) -> np.ndarray | None:
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

# Canvas geometry and speed presets shared by the Play tab and the wizard.
SCALE = 3
GAME_W, GAME_H = 160, 144
CANVAS_W, CANVAS_H = GAME_W * SCALE, GAME_H * SCALE
SPEED_CHOICES = [("0.5×", 0.5), ("1× (real time)", 1.0), ("2×", 2.0),
                 ("4×", 4.0), ("Unlimited", 0.0)]


class _Session:
    """One emulator + wrapped vec-env for a given observation setup.

    `tick_period` > 0 paces every emulator frame to that many seconds
    (1/60 for real time); 0 runs unthrottled.
    """

    def __init__(self, game: str, obs_type: str, action_repeat: int, frame_stack: int,
                 start_level, frames: LatestFrame, tick_period: float = 0.0):
        # Heavy imports (PyBoy, SB3 / torch) happen here, on the playback
        # thread, so the GUI window opens without loading them.
        from pyboy import PyBoy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from env import MarioEnv, wrap_vec_env

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
        self._frames = frames
        self.tick_period = tick_period
        self._deadline = time.perf_counter()

        if game == "mario":
            base = MarioEnv(self.pyboy, frame_skip=action_repeat, obs_type=obs_type,
                            tick_callback=self.on_tick, start_level=start_level)
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

    def push_frame(self) -> None:
        arr = self.pyboy.botsupport_manager().screen().screen_ndarray()
        self._frames.set(np.array(arr, dtype=np.uint8, copy=True))

    def start_pacing(self) -> None:
        """Reset the frame clock (call after a reset or a model load so the
        time spent there is not 'caught up' by racing ahead)."""
        self._deadline = time.perf_counter()

    def on_tick(self) -> None:
        """Called by the env after every emulator frame: publish it, then hold
        the frame until its wall-clock slot so playback runs at game speed."""
        self.push_frame()
        if self.tick_period <= 0:
            return
        self._deadline += self.tick_period
        now = time.perf_counter()
        if self._deadline > now:
            time.sleep(self._deadline - now)
        elif now - self._deadline > RESYNC_AFTER:
            self._deadline = now

    def close(self) -> None:
        try:
            self.vec.close()   # closes MarioEnv -> pyboy.stop()
        except Exception:
            pass


class EmbeddedPlayer:
    def __init__(self, frames: LatestFrame, event_queue: queue.Queue):
        self.frames = frames
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

    def _report_step(self, game: str, action, info, ep_reward: float, ep_steps: int) -> None:
        from env import MarioEnv
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
            self._emit("play_stat", "world", f"{w[0]}-{w[1]}")
        for key in ("lives", "coins"):
            if key in i0:
                self._emit("play_stat", key, str(i0[key]))

    # ---------- play ----------

    def _play_loop(self, model_path, game, obs_type, action_repeat, frame_stack, start_level,
                   episodes, max_steps, deterministic, speed_mult, stop) -> None:
        from stable_baselines3 import PPO
        session = None
        summary: list[dict] = []
        try:
            if game == "mario":
                # Subprocess with timeout, so a level that cannot boot cannot
                # wedge the GUI thread.
                prepare_level_states(start_level)
            tick_period = (1.0 / GB_FPS) / speed_mult if speed_mult > 0 else 0.0
            session = _Session(game, obs_type, action_repeat, frame_stack, start_level,
                               self.frames, tick_period)
            self._emit("play_status", f"loaded {model_path.name}")
            model = PPO.load(str(model_path), env=session.vec, device="cpu")

            for ep in range(episodes):
                if stop.is_set():
                    break
                obs = session.vec.reset()
                session.push_frame()
                session.start_pacing()
                total, steps, done = 0.0, 0, [False]
                self._emit("play_stat", "episode", f"{ep + 1}/{episodes}")
                while not done[0] and not stop.is_set():
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = session.vec.step(action)
                    total += float(reward[0])
                    steps += 1
                    self._report_step(game, action, info, total, steps)
                    if max_steps and steps >= max_steps:
                        break
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
        from stable_baselines3 import PPO
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
                                           start_level, self.frames)
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
                self._report_step(game, action, info, ep_reward, ep_steps)
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
