"""Embedded playback: run a trained model in a windowless PyBoy and stream
frames + stats to the GUI.

The emulator renders into memory (`window_type="null"`, renderer on); every
emulator tick stores the screen in a `LatestFrame` slot (the GUI paints the
newest frame it finds there), and per-step numbers go into `event_queue`
using the GUI's `_pump` protocol:

    ("log", text)                 training-log line
    ("play_status", text)         screen-panel status line
    ("play_stat", key, value)     one live-game stat
    ("play_stats", {key: value})  the per-step stats, one event per step
    ("play_episode", {...})       a round finished: {"episode", "reward",
                                  "steps", "end"} (playback and preview)
    ("play_done", summary)        playback finished; summary = list of
                                  {"reward", "steps"} per completed episode
    ("play_error", traceback)

Pacing is done per emulator frame with `time.sleep()` because PyBoy's own
speed setting only paces when it owns an SDL2 window. Per-frame (not
per-step) pacing matters: a jump step holds the button for 10 frames while
a walk step takes 4, so pacing per step would make jumps run 2.5x too fast
and the canvas would only ever show the last frame of each step.

Three entry points share one engine:
  - `play()`        plays a fixed model for N episodes (Play tab, wizard step 4)
  - `preview()`     follows a training run at training speed (unthrottled),
                    reloading `best_model.zip` whenever it changes, until
                    training ends (Train tab live preview)
  - `play_human()`  runs the plain game at real speed with the buttons the
                    person holds (keyboard / mouse / controller, see
                    controls.py) — no model, no environment, any ROM
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aiboy.games import (DEFAULT_STALL_STEPS, DEFAULT_TIME_BUDGET, GAMES, POWER_NAMES, ROM_DIR,
                         _level_state_path, display_name, power_state, prepare_level_states)
from aiboy.gui.controls import BUTTONS, HeldButtons, action_name, diff_presses
from aiboy.paths import ASSET_DIR, LOCAL_ASSET_DIR, local_or_shipped

GB_FPS = 60.0
RESYNC_AFTER = 0.25         # if pacing falls this far behind, drop the backlog instead of racing


def _single_threaded_torch() -> None:
    """One thread for the policy's inference on the playback thread: the
    models are tiny, and torch's default thread pool spins on every core,
    which competes with the GUI's drawing and makes playback stutter."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


class FramePacer:
    """Holds every emulator frame to its wall-clock slot: `period` seconds
    per frame (1/60 for real time), 0 = unthrottled. `reset()` after a
    pause (a reset, a model load) so the time spent there is not caught up
    by racing ahead."""

    def __init__(self, period: float):
        self.period = period
        self.reset()

    @classmethod
    def for_speed(cls, speed_mult: float) -> FramePacer:
        """A pacer at `speed_mult` times a real Game Boy (0 = unlimited)."""
        return cls((1.0 / GB_FPS) / speed_mult if speed_mult > 0 else 0.0)

    def reset(self) -> None:
        self._deadline = time.perf_counter()

    def wait(self) -> None:
        if self.period <= 0:
            return
        self._deadline += self.period
        now = time.perf_counter()
        if self._deadline > now:
            time.sleep(self._deadline - now)
        elif now - self._deadline > RESYNC_AFTER:
            self._deadline = now


def screen_frame(pyboy) -> np.ndarray:
    """The emulator's screen as an owned (144, 160, 3) uint8 array."""
    arr = pyboy.botsupport_manager().screen().screen_ndarray()
    return np.array(arr, dtype=np.uint8, copy=True)


def intro_asset(name: str, assets: Path = ASSET_DIR, local: Path = LOCAL_ASSET_DIR) -> Path:
    """Path of a boot-video asset: the shipped `assets/<name>` (the "AIboy"
    clip made by tools/make_intro.py), or a local `orig_<name>` when there
    is one, next to the app or next to the shipped file (see
    aiboy.paths.local_or_shipped)."""
    return local_or_shipped(name, assets, local)


INTRO_FRAMES = intro_asset("gb_intro.npz")
INTRO_SOUND = intro_asset("gb_intro.wav")


def play_sound(path: Path):
    """Start playing a WAV without blocking; returns something with .poll()/
    .terminate() or None if no player is available. Uses what the OS ships:
    afplay (macOS), winsound (Windows), paplay / aplay / ffplay (Linux)."""
    import shutil
    import subprocess
    import sys as _sys
    path = Path(path)
    if not path.exists():
        return None
    if _sys.platform == "darwin" and shutil.which("afplay"):
        cmd = ["afplay", str(path)]
    elif _sys.platform.startswith("win"):
        try:
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            return None

        class _Done:                       # winsound plays in-process; nothing to manage
            def poll(self): return 0
            def terminate(self): winsound.PlaySound(None, 0)
        return _Done()
    else:
        for player, extra in (("paplay", []), ("aplay", ["-q"]), ("ffplay", ["-nodisp", "-autoexit", "-v", "quiet"])):
            if shutil.which(player):
                cmd = [player, *extra, str(path)]
                break
        else:
            return None
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


class IntroVideo:
    """The start-up boot video: frames from assets/gb_intro.npz (or a local
    orig_gb_intro.npz, see `intro_asset`) shown on the canvas in step
    with the wall clock while the WAV plays. `idle_frame()` is the last frame
    with the logo visible, used as the "No video" screen."""

    def __init__(self, frames_path: Path = INTRO_FRAMES, sound_path: Path = INTRO_SOUND):
        data = np.load(frames_path)
        self.frames: np.ndarray = data["frames"]
        self.fps = float(data["fps"])
        self.idle_index = int(data["idle_index"])
        self.sound_path = sound_path

    @classmethod
    def available(cls, frames_path: Path = INTRO_FRAMES) -> bool:
        return Path(frames_path).exists()

    def idle_frame(self) -> np.ndarray:
        return self.frames[self.idle_index]

    def play(self, frames: LatestFrame, stop: threading.Event, on_done) -> threading.Thread:
        """Push frames at `fps` until the end or `stop`; then call `on_done()`
        from this thread (the GUI hands it to the Tk thread via its queue)."""
        def _run():
            sound = play_sound(self.sound_path)
            t0 = time.perf_counter()
            for i, frame in enumerate(self.frames):
                if stop.is_set():
                    break
                due = t0 + i / self.fps
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                frames.set(frame)
            if stop.is_set() and sound is not None:
                try:
                    sound.terminate()
                except Exception:
                    pass
            on_done()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t


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

# Emulator frame size and the speed presets of the Play controls.
GAME_W, GAME_H = 160, 144
SPEED_CHOICES = [("0.5×", 0.5), ("1× (real time)", 1.0), ("2×", 2.0),
                 ("4×", 4.0), ("Unlimited", 0.0)]

HUMAN_STATS_EVERY = 6           # frames between two live-panel updates while a person plays (10 Hz)
PREVIEW_SPEED = 0.0             # the live preview during training runs unthrottled, like the trainer


def _button_events():
    """Game Boy button -> (press, release) PyBoy WindowEvents."""
    from pyboy import WindowEvent as W
    return {
        "up": (W.PRESS_ARROW_UP, W.RELEASE_ARROW_UP),
        "down": (W.PRESS_ARROW_DOWN, W.RELEASE_ARROW_DOWN),
        "left": (W.PRESS_ARROW_LEFT, W.RELEASE_ARROW_LEFT),
        "right": (W.PRESS_ARROW_RIGHT, W.RELEASE_ARROW_RIGHT),
        "a": (W.PRESS_BUTTON_A, W.RELEASE_BUTTON_A),
        "b": (W.PRESS_BUTTON_B, W.RELEASE_BUTTON_B),
        "select": (W.PRESS_BUTTON_SELECT, W.RELEASE_BUTTON_SELECT),
        "start": (W.PRESS_BUTTON_START, W.RELEASE_BUTTON_START),
    }


def episode_end_reason(info, steps: int, max_steps: int) -> str:
    """Human-readable reason an episode ended, from the env's last info dict."""
    i0 = info[0] if info and isinstance(info[0], dict) else {}
    w = i0.get("world")
    where = f" in {w[0]}-{w[1]}" if w else ""
    if i0.get("marathon_done"):
        return "completed every level"
    if i0.get("died"):
        return f"died{where}"
    if i0.get("game_over"):
        return f"game over{where}"
    if i0.get("level_cleared"):
        return f"cleared{where}"
    if max_steps and steps >= max_steps:
        return f"step cap reached{where}"
    if i0.get("time_budget_exceeded"):
        return f"time budget used up{where}"
    if i0.get("stalled"):
        return f"no progress for {i0.get('stuck', 0)} steps{where}"
    return f"ended{where}"


class _Session:
    """One emulator + wrapped vec-env for a given observation setup.

    `speed_mult` paces every emulator frame to that multiple of a real
    Game Boy's speed; 0 runs unthrottled.
    """

    def __init__(self, game: str, obs_type: str, action_repeat: int, frame_stack: int,
                 start_level, frames: LatestFrame, speed_mult: float = 0.0,
                 time_budget: int = DEFAULT_TIME_BUDGET, stall_steps: int = DEFAULT_STALL_STEPS):
        # Heavy imports (PyBoy, SB3 / torch) happen here, on the playback
        # thread, so the GUI window opens without loading them.
        from pyboy import PyBoy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from aiboy.env import make_env, wrap_vec_env

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
        self.pacer = FramePacer.for_speed(speed_mult)

        try:
            base = make_env(game, self.pyboy, frame_skip=action_repeat, obs_type=obs_type,
                            tick_callback=self.on_tick, start_level=start_level,
                            time_budget=time_budget, stall_steps=stall_steps)
        except Exception:
            self.pyboy.stop(save=False)
            raise
        self.vec = wrap_vec_env(DummyVecEnv([lambda env=Monitor(base): env]),
                                obs_type, frame_stack)

    def push_frame(self) -> None:
        self._frames.set(screen_frame(self.pyboy))

    def on_tick(self) -> None:
        """Called by the env after every emulator frame: publish it, then hold
        the frame until its wall-clock slot so playback runs at game speed."""
        self.push_frame()
        self.pacer.wait()

    def close(self) -> None:
        try:
            self.vec.close()   # closes the env -> pyboy.stop()
        except Exception:
            pass
        try:
            self.pyboy.stop(save=False)   # the generic wrapper's env does not stop it
        except Exception:
            pass


class EmbeddedPlayer:
    def __init__(self, frames: LatestFrame, event_queue: queue.Queue):
        self.frames = frames
        self.events = event_queue

    # ---------- public entry points ----------

    def play(self, *, model_path: Path, game: str, obs_type: str, action_repeat: int,
             frame_stack: int, start_level, episodes: int, max_steps: int,
             deterministic: bool, speed_mult: float, stop: threading.Event,
             time_budget: int = DEFAULT_TIME_BUDGET,
             stall_steps: int = DEFAULT_STALL_STEPS) -> threading.Thread:
        t = threading.Thread(
            target=self._play_loop, daemon=True,
            args=(model_path, game, obs_type, action_repeat, frame_stack, start_level,
                  episodes, max_steps, deterministic, speed_mult, stop, time_budget, stall_steps))
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

    def play_human(self, *, game: str, rom_path: Path, held: HeldButtons,
                   stop: threading.Event, start_level: tuple[int, int] | None = None,
                   speed_mult: float = 1.0) -> threading.Thread:
        t = threading.Thread(target=self._human_loop, daemon=True,
                             args=(game, rom_path, held, stop, start_level, speed_mult))
        t.start()
        return t

    # ---------- shared helpers ----------

    def _emit(self, *item) -> None:
        self.events.put(item)

    def _report_step(self, game: str, action, info, ep_reward: float, ep_steps: int) -> None:
        from aiboy.env import ENV_CLASSES
        act_id = int(np.asarray(action).flat[0])
        cls = ENV_CLASSES.get(game)
        names = cls.ACTION_NAMES if cls is not None else ()
        action_name = names[act_id] if act_id < len(names) else str(act_id)
        stats = {"action": action_name, "reward": f"{ep_reward:.1f}", "steps": str(ep_steps)}
        i0 = info[0] if info and isinstance(info[0], dict) else {}
        if "x" in i0:
            stats["x"] = f"{i0['x']} (max {i0.get('max_x', '?')})"
        if "world" in i0:
            w = i0["world"]
            stats["world"] = f"{w[0]}-{w[1]}"
        for key in ("lives", "coins", "power"):
            if key in i0:
                stats[key] = str(i0[key])
        if "health" in i0:                       # Kirby: the "power" cell shows health
            stats["power"] = str(i0["health"])
        self._emit("play_stats", stats)

    # ---------- play ----------

    def _play_loop(self, model_path, game, obs_type, action_repeat, frame_stack, start_level,
                   episodes, max_steps, deterministic, speed_mult, stop,
                   time_budget=DEFAULT_TIME_BUDGET, stall_steps=DEFAULT_STALL_STEPS) -> None:
        from stable_baselines3 import PPO
        _single_threaded_torch()
        session = None
        summary: list[dict] = []
        try:
            if game == "mario":
                # Subprocess with timeout, so a level that cannot boot cannot
                # wedge the GUI thread.
                prepare_level_states(start_level)
            session = _Session(game, obs_type, action_repeat, frame_stack, start_level,
                               self.frames, speed_mult, time_budget, stall_steps)
            self._emit("play_status", f"loaded {model_path.name}")
            model = PPO.load(str(model_path), env=session.vec, device="cpu")

            for ep in range(episodes):
                if stop.is_set():
                    break
                obs = session.vec.reset()
                session.push_frame()
                session.pacer.reset()
                total, steps, done, clears = 0.0, 0, [False], 0
                self._emit("play_stat", "episode", f"{ep + 1}/{episodes}")
                while not done[0] and not stop.is_set():
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = session.vec.step(action)
                    total += float(reward[0])
                    steps += 1
                    i0 = info[0] if info and isinstance(info[0], dict) else {}
                    if i0.get("marathon_next_level"):
                        # A marathon clear: the next level was loaded in the same life.
                        w = i0["marathon_next_level"]
                        clears += 1
                        self._emit("play_status", f"level cleared — continuing in {w[0]}-{w[1]} "
                                                  f"({clears} cleared so far)")
                    self._report_step(game, action, info, total, steps)
                    if max_steps and steps >= max_steps:
                        break
                if stop.is_set():
                    break
                reason = episode_end_reason(info, steps, max_steps)
                if clears:
                    reason += f" — {clears} level(s) cleared in one life"
                summary.append({"reward": total, "steps": steps, "end": reason})
                self._emit("play_episode", {"episode": ep + 1, "reward": total, "steps": steps,
                                            "end": reason})
                self._emit("play_status",
                           f"Round {ep + 1}: reward {total:.0f}, {steps} steps, {reason}")
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
        _single_threaded_torch()
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
                        # Training speed (unthrottled), the way the agent is
                        # actually trained; the LCD shows whichever frame is
                        # newest at each paint. Real time is for playing.
                        session = _Session(game, obs_type, action_repeat, frame_stack,
                                           start_level, self.frames, speed_mult=PREVIEW_SPEED)
                        model = PPO.load(str(model_path), env=session.vec, device="cpu")
                        model_mtime = mtime
                        obs = session.vec.reset()
                        session.pacer.reset()
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
                    self._emit("play_episode", {"episode": ep_num, "reward": ep_reward,
                                                "steps": ep_steps,
                                                "end": episode_end_reason(info, ep_steps, 0)})
                    obs = session.vec.reset()
                    session.pacer.reset()
                    ep_num += 1
                    ep_reward, ep_steps = 0.0, 0
                    self._emit("play_stat", "episode", f"{ep_num} (preview)")
        except Exception:
            self._emit("log", f"[preview] error:\n{traceback.format_exc()}\n")
        finally:
            if session is not None:
                session.close()
            self._emit("log", "[preview] stopped\n")
            self._emit("play_status", "idle")


    # ---------- a person plays ----------

    def _human_loop(self, game, rom_path, held: HeldButtons, stop, start_level,
                    speed_mult) -> None:
        """The plain game, paced per frame like `play()`, pressing whatever
        `held` says. The HUD numbers of a supported game go to the live
        panel; other ROMs only show the buttons. Ends when `stop` is set."""
        from pyboy import PyBoy
        pyboy = None
        events = _button_events()
        pressed: frozenset[str] = frozenset()
        pacer = FramePacer.for_speed(speed_mult)
        summary: list[dict] = []
        try:
            rom_path = Path(rom_path)
            if not rom_path.exists():
                raise FileNotFoundError(f"ROM not found: {rom_path}")
            title = display_name(game)
            pyboy = PyBoy(str(rom_path), window_type="null", game_wrapper=True,
                          disable_renderer=False)
            pyboy.set_emulation_speed(0)
            gw = pyboy.game_wrapper() if GAMES.get(game) and GAMES[game].supported else None
            if game == "mario" and gw is not None and start_level is not None:
                # The same save-states the agent trains from (made once, in
                # a subprocess with a timeout).
                missing = prepare_level_states(start_level, rom_path)
                if missing:
                    raise RuntimeError(f"Level {start_level[0]}-{start_level[1]} cannot be loaded.")
                gw.start_game()
                with open(_level_state_path(*start_level), "rb") as f:
                    pyboy.load_state(f)
            self._emit("play_status", f"you play {title}")
            self._emit("play_stat", "episode", "you")

            frame = 0
            best = {"world": None, "score": 0, "coins": 0}
            stats_of = self._human_stats if game == "mario" else self._kirby_human_stats
            game_over_seen = False
            pacer.reset()
            while not stop.is_set():
                now_held = held.held()
                if now_held != pressed:
                    press, release = diff_presses(pressed, now_held)
                    for b in release:
                        pyboy.send_input(events[b][1])
                    for b in press:
                        pyboy.send_input(events[b][0])
                    pressed = now_held
                    self._emit("play_stat", "action", action_name(pressed))
                pyboy.tick()
                self.frames.set(screen_frame(pyboy))
                frame += 1
                if gw is not None and frame % HUMAN_STATS_EVERY == 0:
                    self._emit("play_stats", stats_of(pyboy, gw, frame, best))
                    over = bool(gw.game_over())
                    if over and not game_over_seen:
                        self._emit("play_status", "game over — press START to play again")
                    elif not over and game_over_seen:
                        self._emit("play_status", f"you play {title}")
                    game_over_seen = over
                pacer.wait()
            if gw is not None and best["world"] is not None:
                w = best["world"]
                summary.append({"reward": float(best["score"]), "steps": frame,
                                "end": f"reached {w[0]}-{w[1]}, score {best['score']}"})
                self._emit("play_status", f"you played {frame / GB_FPS:.0f} s: reached "
                                          f"{w[0]}-{w[1]}, score {best['score']}, "
                                          f"{best['coins']} coins")
            elif gw is not None and best["score"]:
                summary.append({"reward": float(best["score"]), "steps": frame,
                                "end": f"score {best['score']}"})
                self._emit("play_status", f"you played {frame / GB_FPS:.0f} s: "
                                          f"score {best['score']}")
            else:
                self._emit("play_status", f"you played {frame / GB_FPS:.0f} s")
            self._emit("play_done", summary)
        except Exception:
            self._emit("play_error", traceback.format_exc())
        finally:
            if pyboy is not None:
                for b in BUTTONS:
                    try:
                        pyboy.send_input(events[b][1])
                    except Exception:
                        pass
                try:
                    pyboy.stop(save=False)
                except Exception:
                    pass

    @staticmethod
    def _kirby_human_stats(pyboy, gw, frame: int, best: dict) -> dict:
        """Live-panel numbers for Kirby's Dream Land while a person plays
        (health in the "power" cell; nothing before the game has started)."""
        score, health, lives = int(gw.score), int(gw.health), int(gw.lives_left)
        if health <= 0 and score <= 0:
            return {"steps": f"{frame / GB_FPS:.0f} s"}
        best["score"] = max(best["score"], score)
        return {"power": str(health), "lives": str(max(lives, 0)), "reward": str(score),
                "steps": f"{frame / GB_FPS:.0f} s"}

    @staticmethod
    def _human_stats(pyboy, gw, frame: int, best: dict) -> dict:
        """Live-panel numbers for Super Mario Land while a person plays.
        Also keeps `best` (furthest world, highest score / coins) for the
        closing line."""
        world = tuple(int(v) for v in gw.world)
        score, coins = int(gw.score), int(gw.coins)
        if world != (0, 0) and (best["world"] is None or world > best["world"]):
            best["world"] = world
        best["score"] = max(best["score"], score)
        best["coins"] = max(best["coins"], coins)
        return {
            "world": f"{world[0]}-{world[1]}" if world != (0, 0) else "—",
            "power": POWER_NAMES[power_state(pyboy)],
            "lives": str(int(gw.lives_left)),
            "coins": str(coins),
            "reward": str(score),
            "x": str(int(gw.level_progress)),
            "steps": f"{int(gw.time_left)} left · {frame / GB_FPS:.0f} s",
        }
