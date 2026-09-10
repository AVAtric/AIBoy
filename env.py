"""PyBoy gym environment factory for Game Boy AI training."""
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy, WindowEvent
from stable_baselines3.common.monitor import Monitor


@dataclass(frozen=True)
class GameSpec:
    rom_file: str
    cartridge_title: str


GAMES = {
    "mario": GameSpec("mario.gb", "SUPER MARIOLAN"),
    "kirby": GameSpec("kirby.gb", "KIRBY DREAM LA"),
    "tetris": GameSpec("tetris.gb", "TETRIS"),
}


class ActionRepeat(gym.Wrapper):
    """Repeat each action for k emulator steps; sum reward, return last frame."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        if k < 1:
            raise ValueError("action_repeat must be >= 1")
        self.k = k

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info: dict = {}
        for _ in range(self.k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class MarioEnv(gym.Env):
    """Super Mario Land env: hold-button actions + progress-based reward shaping.

    Problem with PyBoy's openai_gym(action_type='press'): the button is released
    every frame, so RIGHT taps don't actually walk Mario forward. This env holds
    each action's buttons for `frame_skip` frames, mirroring how a human plays.

    Reward shaping:
      + progress_weight * dx   (dx = change in level_progress this step)
      + score_weight * dscore  (small bonus for coins / enemies)
      - time_penalty           (per-step cost -> encourages fast play)
      - death_penalty          (on losing a life, episode terminates)
      + completion_bonus       (on world change, episode terminates)

    Truncation:
      - stuck_steps consecutive steps without a new max_x
      - game_wrapper.game_over() returns True
    """

    metadata = {"render_modes": []}

    # Actions: buttons to hold during this step (7 discrete actions)
    ACTIONS = (
        (),                                                              # 0 NOOP
        (WindowEvent.PRESS_ARROW_RIGHT,),                                # 1 RIGHT
        (WindowEvent.PRESS_ARROW_LEFT,),                                 # 2 LEFT
        (WindowEvent.PRESS_BUTTON_A,),                                   # 3 JUMP
        (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A),     # 4 RIGHT+JUMP
        (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B),     # 5 RIGHT+RUN
        (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A,
         WindowEvent.PRESS_BUTTON_B),                                    # 6 RIGHT+RUN+JUMP
    )
    RELEASES = (
        (),
        (WindowEvent.RELEASE_ARROW_RIGHT,),
        (WindowEvent.RELEASE_ARROW_LEFT,),
        (WindowEvent.RELEASE_BUTTON_A,),
        (WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A),
        (WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_B),
        (WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A,
         WindowEvent.RELEASE_BUTTON_B),
    )

    def __init__(
        self,
        pyboy: PyBoy,
        frame_skip: int = 4,
        stuck_steps: int = 250,
        progress_weight: float = 1.0,
        score_weight: float = 0.02,
        time_penalty: float = 0.1,
        death_penalty: float = 25.0,
        completion_bonus: float = 500.0,
    ):
        super().__init__()
        self.pyboy = pyboy
        self.gw = pyboy.game_wrapper()
        self.frame_skip = frame_skip
        self.stuck_steps = stuck_steps
        self.progress_weight = progress_weight
        self.score_weight = score_weight
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.completion_bonus = completion_bonus

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(0, 255, (144, 160, 3), dtype=np.uint8)

        self._started = False
        self._last_x = 0
        self._max_x = 0
        self._last_lives = 0
        self._last_world = (0, 0)
        self._last_score = 0
        self._stuck = 0

    def _obs(self) -> np.ndarray:
        return np.asarray(
            self.pyboy.botsupport_manager().screen().screen_ndarray(), dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self._started:
            self.gw.start_game()
            self._started = True
        else:
            self.gw.reset_game()
        self._last_x = self.gw.level_progress
        self._max_x = self._last_x
        self._last_lives = self.gw.lives_left
        self._last_world = tuple(self.gw.world)
        self._last_score = self.gw.score
        self._stuck = 0
        return self._obs(), {}

    def step(self, action: int):
        for evt in self.ACTIONS[action]:
            self.pyboy.send_input(evt)
        for _ in range(self.frame_skip):
            self.pyboy.tick()
        for evt in self.RELEASES[action]:
            self.pyboy.send_input(evt)

        x = self.gw.level_progress
        lives = self.gw.lives_left
        world = tuple(self.gw.world)
        score = self.gw.score

        terminated = False
        reward = 0.0
        if world != self._last_world:
            reward += self.completion_bonus
            terminated = True
        elif lives < self._last_lives:
            reward -= self.death_penalty
            terminated = True
        else:
            reward += self.progress_weight * (x - self._last_x)
            reward += self.score_weight * max(0, score - self._last_score)
            reward -= self.time_penalty

        if x > self._max_x:
            self._max_x = x
            self._stuck = 0
        else:
            self._stuck += 1

        truncated = self._stuck >= self.stuck_steps or self.gw.game_over()

        info = {
            "x": x, "max_x": self._max_x, "lives": lives,
            "world": world, "stuck": self._stuck,
        }
        self._last_x = x
        self._last_lives = lives
        self._last_world = world
        self._last_score = score
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.pyboy.stop(save=False)
        except Exception:
            pass


def make_pyboy_env(
    game: str = "mario",
    window_type: str = "null",
    action_repeat: int = 4,
    seed: int | None = None,
    rom_dir: str | Path = "ROMs",
    emulation_speed: int | None = None,
) -> gym.Env:
    """Build a PyBoy gym environment for the given game.

    For 'mario': uses MarioEnv (hold-button actions + progress reward).
    For other games: uses PyBoy's default openai_gym + ActionRepeat wrapper.
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game '{game}'. Available: {sorted(GAMES)}")
    spec = GAMES[game]
    rom_path = Path(rom_dir) / spec.rom_file
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    disable_renderer = window_type == "null"
    pyboy = PyBoy(
        str(rom_path),
        window_type=window_type,
        game_wrapper=True,
        disable_renderer=disable_renderer,
    )
    if emulation_speed is None:
        emulation_speed = 0 if window_type == "null" else 1
    pyboy.set_emulation_speed(emulation_speed)

    actual_title = pyboy.cartridge_title()
    if actual_title != spec.cartridge_title:
        print(f"[env] warning: expected '{spec.cartridge_title}', got '{actual_title}'")

    if game == "mario":
        env: gym.Env = MarioEnv(pyboy, frame_skip=action_repeat)
    else:
        env = pyboy.openai_gym(observation_type="raw", action_type="press")
        if action_repeat > 1:
            env = ActionRepeat(env, k=action_repeat)

    if seed is not None:
        env.reset(seed=seed)
    return env


def env_factory(
    game: str,
    seed: int = 0,
    window_type: str = "null",
    action_repeat: int = 4,
) -> gym.Env:
    """Picklable factory for SubprocVecEnv workers."""
    env = make_pyboy_env(
        game=game,
        window_type=window_type,
        action_repeat=action_repeat,
        seed=seed,
    )
    return Monitor(env)
