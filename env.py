"""PyBoy gym environment factory for Game Boy AI training."""
import random as _random_mod
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy, WindowEvent
from stable_baselines3.common.monitor import Monitor


# Super Mario Land has 4 worlds × 3 levels = 12 total levels. PyBoy's
# `set_world_level(w, l)` docstring is wrong — it says args are 0-indexed
# (0-3 world, 0-2 level) but the actual SML memory encoding is 1-INDEXED:
# byte 0x11 = 1-1, 0x21 = 2-1, 0x43 = 4-3. Writing invalid values (0x00,
# 0x10, 0x20, 0x30, 0x40, 0x01…) silently loads a default fallback level
# while `game_wrapper.world` still reports the (wrong) patched tuple. So
# we always pass w, l as 1-indexed to `start_game(world_level=…)`.
#
# Even with 1-indexed values, levels 2-3 and 4-3 don't load correctly —
# `start_game(world_level=(2,3))` also falls through to the default level.
# Those are the truly-broken level values in the wrapper.
SML_BROKEN_LEVELS = frozenset({(2, 3), (4, 3)})
SML_ALL_LEVELS = tuple(
    (w, l) for w in range(1, 5) for l in range(1, 4)
    if (w, l) not in SML_BROKEN_LEVELS
)

# Where per-level save-state files live. Gitignored via models/.
LEVEL_STATES_DIR = Path("models") / "mario" / "_level_states"


def _level_state_path(world: int, level: int) -> Path:
    return LEVEL_STATES_DIR / f"{world}-{level}.state"


def _bootstrap_level_state(rom_path: Path, world: int, level: int,
                            timeout_sec: int = 20) -> bool:
    """Create a save-state file for SML level `world-level` (1-indexed) via a
    subprocess (isolated so hangs can be killed). Returns True on success.
    Idempotent: skips work if the state file already exists.
    """
    state_file = _level_state_path(world, level)
    if state_file.exists() and state_file.stat().st_size > 1000:
        return True
    if (world, level) in SML_BROKEN_LEVELS:
        return False
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # PyBoy's set_world_level encoding is 1-indexed (memory byte 0x11 = 1-1),
    # despite what its docstring claims. Pass w, l as-is (1-indexed).
    code = (
        "from pyboy import PyBoy;"
        f"p = PyBoy({str(rom_path.resolve())!r}, window_type='null', "
        "game_wrapper=True, disable_renderer=True);"
        f"p.game_wrapper().start_game(world_level=({world},{level}));"
        # Let the level fully load before saving state so the tile map is
        # settled — avoids a state that boots into the "level intro" screen.
        "[p.tick() for _ in range(60)];"
        f"p.save_state(open({str(state_file.resolve())!r}, 'wb'));"
        "p.stop(save=False)"
    )
    import subprocess
    import sys as _sys
    try:
        subprocess.run(
            [_sys.executable, "-c", code],
            timeout=timeout_sec, capture_output=True, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        state_file.unlink(missing_ok=True)
        return False
    return state_file.exists() and state_file.stat().st_size > 1000


def ensure_level_states(rom_path: Path, targets=None) -> list[tuple[int, int]]:
    """Bootstrap save-state files for the given levels. If targets is None,
    bootstraps every usable level. Returns list of levels that succeeded.
    """
    if targets is None:
        targets = list(SML_ALL_LEVELS)
    ok: list[tuple[int, int]] = []
    for w, l in targets:
        if _bootstrap_level_state(Path(rom_path), w, l):
            ok.append((w, l))
    return ok


def parse_start_level(spec):
    """Parse a start_level spec into None, "random", or a (world, level) tuple.

    Accepts: None, "default", "random", "W-L" string (e.g. "2-3"),
    or a (world, level) tuple.
    """
    if spec is None or spec == "default":
        return None
    if spec == "random":
        return "random"
    if isinstance(spec, tuple) and len(spec) == 2:
        w, l = int(spec[0]), int(spec[1])
    elif isinstance(spec, str) and "-" in spec:
        parts = spec.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid start_level {spec!r}, expected 'W-L' (e.g. '2-3')")
        w, l = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"invalid start_level {spec!r}")
    if (w, l) in SML_BROKEN_LEVELS:
        raise ValueError(
            f"level {w}-{l} cannot be started via PyBoy's SML wrapper "
            f"(known upstream bug — start_game hangs). Pick a different level."
        )
    if (w, l) not in SML_ALL_LEVELS:
        raise ValueError(f"level {w}-{l} not in Super Mario Land (worlds 1-4, levels 1-3)")
    return (w, l)


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

    Discrete(11) action space, symmetric so Mario can jump in either direction
    (essential for maneuvers like "back up, jump onto a brick, jump forward
    over an obstacle with enemies on top") plus DOWN so Mario can enter
    downward pipes:
        0 NOOP           1 RIGHT          2 LEFT          3 JUMP
        4 RIGHT+JUMP     5 RIGHT+RUN      6 RIGHT+RUN+JUMP
        7 LEFT+JUMP      8 LEFT+RUN       9 LEFT+RUN+JUMP
        10 DOWN  (crouch / enter pipe when standing on one)

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
        + coin_weight    * dcoins             (each coin picked up)
        + score_weight    * dscore            (coins / enemies / items)
        - time_penalty                        (small per-step cost — pressure
                                                against noop / oscillation)

    Three modes, decided by `start_level`:
      - None (campaign): play through the game. Death does NOT end the
        episode — Mario respawns at level start, life count decremented,
        max_x tracker resets. Level clear also doesn't end the episode —
        Mario transitions naturally to the next level and max_x resets.
        Episode ends only on `game_over()` (all lives exhausted) or the
        stuck timeout.
      - "W-L" (fixed): each reset loads the save-state for that level. Any
        death or level clear ends the episode; next reset replays the same
        level (so lives are effectively infinite for training).
      - "random": each reset picks a random usable level from
        SML_ALL_LEVELS and loads its state. Death or level clear ends the
        episode so the next reset picks a new level — Mario is always
        training on varied content.

    Rewards fire in every mode: +completion_bonus on level clear,
    -death_penalty on death (whether or not the episode ends).
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
        (WindowEvent.PRESS_ARROW_DOWN,),                                   # 10 DOWN
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
        (WindowEvent.RELEASE_ARROW_DOWN,),                                 # 10 DOWN
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
        start_level=None,
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
        # Starting level:
        #   None       → SML's default (world 1-1) → CAMPAIGN mode
        #   (w, l)     → always start there (1-indexed, e.g. (2, 3) = level 2-3)
        #   "random"   → pick a random level from SML_ALL_LEVELS each reset
        self.start_level = parse_start_level(start_level)
        self._level_rng = _random_mod.Random()
        # Campaign mode = play through the game respecting lives and level
        # progression (deaths respawn, level-clears advance). Only triggered
        # when no start_level is set.
        self._campaign_mode = self.start_level is None
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
            # Pixel mode: the raw RGB screen already includes the HUD (lives
            # counter, coin count, timer, score) so the agent can in principle
            # read them from pixels.
            return np.asarray(
                self.pyboy.botsupport_manager().screen().screen_ndarray(), dtype=np.uint8
            )
        # Semantic tile grid from PyBoy's SML wrapper:
        #   -1.0 = Mario, 0.0 = empty, 0.5 = ground/ledge, 0.6 = enemy, 1.0 = pipe/wall
        # Mario is distinct from enemies (unlike custom_minimal_policy).
        arr = np.asarray(self.gw.custom_minimal_enemy(), dtype=np.float32)
        # The HUD area at the top of the screen collapses to 0.0 in this
        # representation, so the agent otherwise has NO awareness of lives,
        # coins, timer, its position in the level, OR which level it's
        # playing. Overlay 6 normalised scalar features into the top-left
        # cells so the agent knows how careful to be (low lives), how
        # urgent the level is (low time), where in the level it is
        # (current_x), and which world/level it's in (crucial in campaign
        # / random mode — each level has different obstacles, enemies,
        # physics). max_x is intentionally NOT included: it's env-internal
        # bookkeeping (used by the reward function), the agent doesn't
        # need it to pick optimal actions, and current_x is strictly more
        # informative game-state.
        # This doesn't change the obs shape or the policy architecture.
        w, l = self.gw.world
        arr[0, 0] = min(max(self.gw.lives_left, 0), 9) / 9.0        # lives:  0 .. ~1
        arr[0, 1] = min(self.gw.coins, 99) / 99.0                    # coins:  0 .. ~1
        arr[0, 2] = min(max(self.gw.time_left, 0), 400) / 400.0      # timer:  0 .. 1
        arr[0, 3] = min(self.gw.level_progress, 4096) / 4096.0       # current x in level
        arr[0, 4] = min(max(int(w) - 1, 0), 3) / 3.0                 # world:  0, 1/3, 2/3, 1
        arr[0, 5] = min(max(int(l) - 1, 0), 2) / 2.0                 # level:  0, 1/2, 1
        return arr[..., np.newaxis]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._level_rng.seed(seed)

        target = None
        if self.start_level == "random":
            target = self._level_rng.choice(SML_ALL_LEVELS)
        elif isinstance(self.start_level, tuple):
            target = self.start_level

        if target is not None:
            # Preferred path: load a pre-saved emulator state for this level.
            # This is what PyBoy's docs recommend for level switching — instant,
            # race-free, and avoids the start_game()-hang for some levels.
            state_file = _level_state_path(*target)
            if state_file.exists() and state_file.stat().st_size > 1000:
                # PyBoy needs the game_wrapper "started" flag set for helpers
                # like level_progress / lives_left to read from the right memory
                # after load_state, so make sure start_game() has run at least
                # once first.
                if not self._started:
                    self.gw.start_game()
                    self._started = True
                with open(state_file, "rb") as f:
                    self.pyboy.load_state(f)
                # CRITICAL: after load_state, memory-mapped values like
                # level_progress / lives_left / world stay stale until at
                # least one tick runs and the wrapper refreshes. Without
                # this, our first step() sees `lives_left` drop from stale
                # to real, misfires the death termination, and the env
                # ends up in a reset-die-reset-die infinite loop.
                for _ in range(4):
                    self.pyboy.tick()
            else:
                # Fallback: state file missing → try the old path. May hang
                # for broken levels. `ensure_level_states()` should have been
                # called upfront to avoid this. Note: 1-indexed args.
                self.gw.set_world_level(target[0], target[1])
                self.gw.start_game()
                self._started = True
        elif not self._started:
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
        is_game_over = self.gw.game_over()

        died = lives < self._last_lives
        level_cleared = world != self._last_world

        reward = 0.0
        if level_cleared:
            reward += self.completion_bonus
        if died:
            reward -= self.death_penalty
        if not (level_cleared or died):
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

        # Termination logic depends on mode
        if self._campaign_mode:
            # Play through: only game_over ends the episode. Death and level
            # clear are transitions within the same episode. Reset the max_x
            # tracker after each transition so the agent can earn progress
            # reward on the new attempt / new level.
            terminated = is_game_over
            if died or level_cleared:
                self._max_x = x
                self._stuck = 0
        else:
            # Fixed / random: any death or level clear ends the episode.
            terminated = died or level_cleared

        if x > self._max_x:
            self._max_x = x
            self._stuck = 0
        elif not (died or level_cleared):
            self._stuck += 1

        truncated = self._stuck >= self.stuck_steps or is_game_over

        info = {
            "x": x, "max_x": self._max_x, "lives": lives,
            "world": world, "coins": coins, "score": score,
            "stuck": self._stuck, "game_over": is_game_over,
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
    start_level=None,
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
        env: gym.Env = MarioEnv(pyboy, frame_skip=action_repeat, obs_type=obs_type,
                                 start_level=start_level)
    else:
        if start_level is not None and start_level != "default":
            pyboy.stop(save=False)
            raise ValueError(f"start_level only supported for mario, not {game!r}")
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
    start_level=None,
) -> gym.Env:
    """Picklable factory for SubprocVecEnv workers."""
    env = make_pyboy_env(
        game=game,
        window_type=window_type,
        action_repeat=action_repeat,
        seed=seed,
        obs_type=obs_type,
        start_level=start_level,
    )
    return Monitor(env)
