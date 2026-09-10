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
    "wario": GameSpec("wario.gb", "WARIO"),
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
    """Super Mario Land env: hold-button actions + shaped reward.

    Buttons are held for the whole `frame_skip` window (not released every
    frame like PyBoy's default openai_gym). Without this, tapping RIGHT
    barely moves Mario and the agent has no gradient toward forward progress.

    Discrete(10) action space, symmetric so Mario can jump in either direction
    (essential for maneuvers like "back up, jump onto a brick, jump forward
    over an obstacle with enemies on top"):
        0 NOOP           1 RIGHT          2 LEFT          3 JUMP
        4 RIGHT+JUMP     5 RIGHT+RUN      6 RIGHT+RUN+JUMP
        7 LEFT+JUMP      8 LEFT+RUN       9 LEFT+RUN+JUMP

    Two observation modes (`obs_type`):
      - `"pixels"`: (144, 160, 3) uint8 raw RGB screen — use with CnnPolicy
      - `"tiles"`:  (16, 20, 1) float32 semantic tile categories from PyBoy's
        `custom_minimal_enemy()` — values in {-1.0=Mario, 0.0=empty,
        0.5=ground/ledge, 0.6=enemy, 1.0=pipe/wall}. Mario is distinguished
        from enemies (unlike custom_minimal_policy() where both are 0.6).
        216× smaller than pixels, trains ~10× faster on CPU.

    Per-step reward:
        + progress_weight * new_max_x_delta   (ONLY reward for new territory;
                                                re-covering ground gives 0,
                                                so back-and-forth cannot farm)
        + score_weight    * dscore            (coins / enemies)
        - time_penalty                        (small per-step cost — pressure
                                                against noop / oscillation)

    Terminal: life lost → -death_penalty; world change → +completion_bonus.
    Truncation: no new max_x for stuck_steps, or `game_over()` returns True.
    """

    metadata = {"render_modes": []}

    ACTIONS = (
        (),                                                                # 0 NOOP
        (WindowEvent.PRESS_ARROW_RIGHT,),                                  # 1 RIGHT
        (WindowEvent.PRESS_ARROW_LEFT,),                                   # 2 LEFT
        (WindowEvent.PRESS_BUTTON_A,),                                     # 3 JUMP
        (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A),       # 4 RIGHT+JUMP
        (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B),       # 5 RIGHT+RUN
        (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A,
         WindowEvent.PRESS_BUTTON_B),                                      # 6 RIGHT+RUN+JUMP
        (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A),        # 7 LEFT+JUMP
        (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B),        # 8 LEFT+RUN
        (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A,
         WindowEvent.PRESS_BUTTON_B),                                      # 9 LEFT+RUN+JUMP
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
        (WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A),
        (WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B),
        (WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A,
         WindowEvent.RELEASE_BUTTON_B),
    )

    # Divisor for normalizing tile IDs to [0, 1]-ish
    TILE_NORM = 400.0

    def __init__(
        self,
        pyboy: PyBoy,
        frame_skip: int = 4,
        stuck_steps: int = 200,
        progress_weight: float = 3.0,
        coin_weight: float = 5.0,
        score_weight: float = 0.05,
        time_penalty: float = 0.03,
        death_penalty: float = 500.0,
        completion_bonus: float = 1000.0,
        obs_type: str = "pixels",
        tick_callback=None,
        jump_hold_bonus: int = 6,
    ):
        super().__init__()
        if obs_type not in ("pixels", "tiles"):
            raise ValueError(f"obs_type must be 'pixels' or 'tiles', got {obs_type!r}")
        self.pyboy = pyboy
        self.gw = pyboy.game_wrapper()
        self.frame_skip = frame_skip
        self.stuck_steps = stuck_steps
        self.progress_weight = progress_weight
        self.coin_weight = coin_weight
        self.score_weight = score_weight
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.completion_bonus = completion_bonus
        self.obs_type = obs_type
        self.tick_callback = tick_callback  # called after each pyboy.tick() during step()
        # When the action includes JUMP, hold A for `frame_skip + jump_hold_bonus`
        # additional ticks so Mario gets a real jump (Super Mario Land jump
        # height scales with A-hold duration up to ~12 frames).
        self.jump_hold_bonus = jump_hold_bonus
        self._jump_actions = frozenset(i for i, evts in enumerate(self.ACTIONS)
                                        if WindowEvent.PRESS_BUTTON_A in evts)

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        if obs_type == "pixels":
            self.observation_space = spaces.Box(0, 255, (144, 160, 3), dtype=np.uint8)
        else:
            # custom_minimal_enemy() values in {-1.0, 0.0, 0.5, 0.6, 1.0}
            self.observation_space = spaces.Box(-1.0, 1.0, (16, 20, 1), dtype=np.float32)

        self._started = False
        self._last_x = 0
        self._max_x = 0
        self._last_lives = 0
        self._last_world = (0, 0)
        self._last_score = 0
        self._last_coins = 0
        self._stuck = 0

    def _obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return np.asarray(
                self.pyboy.botsupport_manager().screen().screen_ndarray(), dtype=np.uint8
            )
        # Semantic tile grid from PyBoy's SML wrapper:
        # -1.0 = Mario, 0.0 = empty, 0.5 = ground/ledge, 0.6 = enemy, 1.0 = pipe/wall.
        # Mario is distinct from enemies (unlike custom_minimal_policy).
        arr = np.asarray(self.gw.custom_minimal_enemy(), dtype=np.float32)
        return arr[..., np.newaxis]

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
        self._last_coins = self.gw.coins
        self._stuck = 0
        return self._obs(), {}

    def step(self, action: int):
        for evt in self.ACTIONS[action]:
            self.pyboy.send_input(evt)
        # Hold ALL buttons longer when the action includes JUMP so Mario
        # actually clears tall obstacles (SML jump height scales with A-hold
        # up to ~12 frames; 4 gives only ~4 tiles which won't clear a pipe+goomba).
        is_jump = action in self._jump_actions
        n_ticks = self.frame_skip + (self.jump_hold_bonus if is_jump else 0)
        for _ in range(n_ticks):
            self.pyboy.tick()
            if self.tick_callback is not None:
                self.tick_callback()
        for evt in self.RELEASES[action]:
            self.pyboy.send_input(evt)

        x = self.gw.level_progress
        lives = self.gw.lives_left
        world = tuple(self.gw.world)
        score = self.gw.score
        coins = self.gw.coins

        terminated = False
        reward = 0.0
        if world != self._last_world:
            reward += self.completion_bonus
            terminated = True
        elif lives < self._last_lives:
            reward -= self.death_penalty
            terminated = True
        else:
            # Progress reward: only NEW forward territory counts. Re-covering
            # or backtracking gives 0 (not extra negative) so the agent is
            # free to reposition (e.g. back up to jump on a brick) without
            # extra punishment beyond the constant per-step time_penalty.
            if x > self._max_x:
                reward += self.progress_weight * (x - self._max_x)
            # Explicit reward for collecting coins
            dcoins = coins - self._last_coins
            if dcoins > 0:
                reward += self.coin_weight * dcoins
            # Broader score reward — enemy kills, pipe-entry bonuses, etc.
            dscore = score - self._last_score
            if dscore > 0:
                reward += self.score_weight * dscore
            # Constant time penalty: every step costs a tiny bit. Standing
            # still or backtracking still incurs this cost (like an in-game
            # timer counting down), so the only way to net positive reward
            # is via forward progress, coins, kills, or level completion.
            reward -= self.time_penalty

        if x > self._max_x:
            self._max_x = x
            self._stuck = 0
        else:
            self._stuck += 1

        truncated = self._stuck >= self.stuck_steps or self.gw.game_over()

        info = {
            "x": x, "max_x": self._max_x, "lives": lives,
            "world": world, "coins": coins, "score": score,
            "stuck": self._stuck,
        }
        self._last_x = x
        self._last_lives = lives
        self._last_world = world
        self._last_score = score
        self._last_coins = coins
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
    obs_type: str = "pixels",
) -> gym.Env:
    """Build a PyBoy gym env.
    - 'mario' → MarioEnv (hold-button actions + shaped reward, pixels or tiles)
    - other games → PyBoy's default openai_gym + ActionRepeat (pixels only)
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game '{game}'. Available: {sorted(GAMES)}")
    spec = GAMES[game]
    rom_path = Path(rom_dir) / spec.rom_file
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    disable_renderer = window_type == "null" and obs_type != "pixels"
    # Note: for tile obs we can safely disable rendering (we read game_area).
    # For pixel obs we need rendering even in null mode.
    if obs_type == "pixels":
        disable_renderer = False if window_type == "SDL2" else False
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
        env: gym.Env = MarioEnv(pyboy, frame_skip=action_repeat, obs_type=obs_type)
    else:
        if pyboy.game_wrapper() is None:
            pyboy.stop(save=False)
            raise ValueError(
                f"Game '{game}' has no PyBoy game_wrapper; cannot build a gym env. "
                f"Only games with a built-in wrapper are supported."
            )
        if obs_type != "pixels":
            pyboy.stop(save=False)
            raise ValueError(f"obs_type={obs_type!r} only supported for mario")
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
    obs_type: str = "pixels",
) -> gym.Env:
    """Picklable factory for SubprocVecEnv workers."""
    env = make_pyboy_env(
        game=game,
        window_type=window_type,
        action_repeat=action_repeat,
        seed=seed,
        obs_type=obs_type,
    )
    return Monitor(env)
