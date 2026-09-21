"""The Gymnasium environments AIboy trains in (PyBoy underneath).

    GameBoyEnv   the base every game builds on: hold-button actions, pixel or
                 tile observations, the frame loop (with the GUI's per-frame
                 callback), start / reset through PyBoy's game wrapper, and
                 the furthest-point tracker behind the stall rule
    MarioEnv     Super Mario Land: shaped reward and the five level modes
                 (campaign, fixed level, random, sequential, marathon)
    KirbyEnv     Kirby's Dream Land: reward for getting further (the screen
                 scrolling right), score, health and lives
    ENV_CLASSES  game name -> environment class; `make_pyboy_env` builds one

A game adds three things to the base: its action table (`ACTION_NAMES` /
`ACTIONS`, press events; the releases follow), its tile grid (`_tiles`, a
16x20 float array in [-1, 1] for the "tiles" observation) and its reward
(`_evaluate`, called after every action with the emulator advanced). A ROM
without an environment here can be played by hand, not trained. Only the
stock PyBoy 1.6 is needed: the tile meanings live in `mario_tile_lut`.
"""
from __future__ import annotations

import random as _random_mod
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy, WindowEvent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage

# Light, emulator-free facts live in games.py; re-exported here so existing
# `from env import ...` call sites keep working.
from aiboy.games import (  # noqa: F401
    ADDR_GAME_STATE, ADDR_KIRBY_X, ADDR_POWERUP_STATE, ADDR_SUPERBALL, DEFAULT_STALL_STEPS,
    DEFAULT_TIME_BUDGET, GAMES, KIRBY_ATTEMPT_STEPS, KIRBY_MAX_HEALTH, LEVEL_MODES,
    LEVEL_STATES_DIR, TIMER_START, MULTI_LEVEL_MODES, OBS_TYPES, POWER_NAMES, POWER_SMALL,
    POWER_SUPER, POWER_SUPERBALL, ROM_DIR, ROM_SUFFIXES, SML_ALL_LEVELS, SML_BACKGROUND_TILES,
    SML_BROKEN_LEVELS, SML_CLEAR_STATES, SML_DEATH_STATES, SUPPORTED_GAMES, GameSpec, RomInfo,
    _level_state_path,
    check_start_level, discover_roms, ensure_level_states, level_choices, level_targets,
    parse_start_level, power_state, prepare_level_states, probe_rom,
)

GB_W, GB_H = 160, 144
TILE_SHAPE = (16, 20)

# Press event -> its release, so a game only lists what an action presses.
RELEASE_OF = {
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
    WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
}
UP, DOWN, LEFT, RIGHT = (WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN,
                         WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT)
A, B = WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B


def mario_tile_lut() -> np.ndarray:
    """Tile identifier -> the value MarioEnv's tile observation shows for it.

    Built from the tile categories PyBoy's Super Mario Land wrapper defines
    (its module-level lists), in this order, a later category overriding an
    earlier one for a tile listed twice:

        Mario (incl. plane and submarine)        -1.0
        coin                                      0.80
        mushroom, heart, star, lever              0.85
        plain and moving blocks (ground)          0.5
        pushable and question blocks              0.6
        pipes                                     0.5
        enemies, easy and hard, and projectiles   1.0
        everything else (sky, background)         0.0

    with one correction: SML_BACKGROUND_TILES (decoration PyBoy files under
    the blocks; see games.py for how that was measured) read 0.0.

    Otherwise these are exactly the values of the `custom_minimal_enemy()`
    method a locally patched PyBoy used to provide. Changing a value here
    changes what a trained model sees: bump ENV_VERSIONS["mario"].
    """
    from pyboy.plugins import game_wrapper_super_mario_land as sml
    categories = (
        (sml.base_scripts + sml.plane + sml.submarine, -1.0),
        (sml.coin, 0.80),
        (sml.mushroom + sml.heart + sml.star + sml.lever, 0.85),
        (sml.neutral_blocks + sml.moving_blocks, 0.5),
        (sml.pushable_blokcs + sml.question_block, 0.6),
        (sml.pipes, 0.5),
        (sml.goomba + sml.koopa + sml.moth + sml.flying_moth + sml.sphinx, 1.0),
        (sml.big_sphinx + sml.fist + sml.bill + sml.projectiles + sml.shell + sml.explosion
         + sml.spike + sml.plant, 1.0),
    )
    lut = np.zeros(int(sml.TILES), dtype=np.float32)
    for tiles, value in categories:
        lut[list(tiles)] = value
    lut[list(SML_BACKGROUND_TILES)] = 0.0
    return lut


class GameBoyEnv(gym.Env):
    """What every game shares.

    Actions are held for the whole `frame_skip` window (not released every
    frame like PyBoy's default openai_gym): tapping RIGHT barely moves a
    character, holding it does. A game may hold some actions longer
    (`hold_frames`, Mario's jump).

    Observations (`obs_type`):
      - "pixels": (144, 160, 3) uint8 raw RGB screen, for CnnPolicy
      - "tiles":  (16, 20, 1) float32 grid in [-1, 1] the game derives from
        PyBoy's tile map (`_tiles`), 216x smaller than pixels; games write
        a few normalised HUD numbers into its top-left cells

    `reset()` starts the game through PyBoy's wrapper the first time and
    resets to that saved point afterwards (`_load_start`), then calls
    `_begin_attempt()` so reward bookkeeping starts clean. `step()` presses,
    advances the emulator (calling `tick_callback` after every frame: the
    GUI paints and paces there), releases, and asks the game to judge the
    result (`_evaluate`).

    Two generic attempt limits: `stuck_steps` steps without new progress
    (the furthest-point tracker `_progress`; 0 = off) and, for Mario, the
    in-game timer budget `time_budget` (games without a timer ignore it).
    """

    metadata = {"render_modes": []}
    ACTION_NAMES: tuple[str, ...] = ("NOOP",)
    ACTIONS: tuple[tuple, ...] = ((),)
    TILE_NORM = 400.0           # divisor that maps PyBoy tile identifiers into [0, 1]

    def __init__(self, pyboy: PyBoy, frame_skip: int = 4, obs_type: str = "pixels",
                 tick_callback=None, stuck_steps: int = DEFAULT_STALL_STEPS,
                 time_budget: int = DEFAULT_TIME_BUDGET, time_penalty: float = 0.03,
                 death_penalty: float = 500.0):
        super().__init__()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type must be one of {OBS_TYPES}, got {obs_type!r}")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.pyboy = pyboy
        self.gw = pyboy.game_wrapper()
        if self.gw is None:
            raise ValueError("PyBoy has no game wrapper for this cartridge")
        self.frame_skip = frame_skip
        self.obs_type = obs_type
        self.tick_callback = tick_callback
        self.stuck_steps = stuck_steps
        self.time_budget = time_budget
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        if obs_type == "pixels":
            self.observation_space = spaces.Box(0, 255, (GB_H, GB_W, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, (*TILE_SHAPE, 1), dtype=np.float32)
        self._started = False
        self._max_x = 0.0
        self._stuck = 0

    # ----- what a game provides -----

    def _tiles(self) -> np.ndarray:
        """The (16, 20) float32 grid of the "tiles" observation."""
        raise NotImplementedError

    def _evaluate(self, action: int) -> tuple[float, bool, bool, dict]:
        """After the emulator advanced for `action`: (reward, terminated,
        truncated, info)."""
        raise NotImplementedError

    def hold_frames(self, action: int) -> int:
        """Emulator frames the buttons of `action` stay down."""
        return self.frame_skip

    def _load_start(self) -> None:
        """Put the game at the start of an attempt: PyBoy's `start_game`
        once (it saves that point), `reset_game` after that."""
        if not self._started:
            self.gw.start_game()
            self._started = True
        else:
            self.gw.reset_game()

    def _begin_attempt(self) -> None:
        """Reward bookkeeping for a fresh attempt (called after every load)."""
        self._stuck = 0

    # ----- shared machinery -----

    @classmethod
    def releases(cls, action: int) -> tuple:
        return tuple(RELEASE_OF[evt] for evt in cls.ACTIONS[action])

    def _screen(self) -> np.ndarray:
        return np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray(),
                          dtype=np.uint8)

    def _obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            # The raw screen includes the HUD, so the agent can read lives,
            # score and the rest from the pixels.
            return self._screen()
        return self._tiles()[..., np.newaxis]

    def _progress(self, x: float, *, restart: bool = False, count: bool = True) -> float:
        """Feed the furthest-point tracker. Returns the new territory gained
        (0 when none). `restart` begins a fresh attempt at `x`; `count=False`
        leaves the stall counter alone (a step that died or cleared)."""
        if restart:
            self._max_x, self._stuck = x, 0
            return 0.0
        if x > self._max_x:
            gained = x - self._max_x
            self._max_x, self._stuck = x, 0
            return gained
        if count:
            self._stuck += 1
        return 0.0

    def _stalled(self) -> bool:
        return self.stuck_steps > 0 and self._stuck >= self.stuck_steps

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load_start()
        self._begin_attempt()
        return self._obs(), {}

    def step(self, action: int):
        for evt in self.ACTIONS[action]:
            self.pyboy.send_input(evt)
        for _ in range(self.hold_frames(action)):
            self.pyboy.tick()
            if self.tick_callback is not None:
                self.tick_callback()
        for evt in self.releases(action):
            self.pyboy.send_input(evt)
        reward, terminated, truncated, info = self._evaluate(action)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            self.pyboy.stop(save=False)
        except Exception:
            pass


class MarioEnv(GameBoyEnv):
    """Super Mario Land: hold-button actions + shaped reward.

    Discrete(11) action space, symmetric so Mario can jump in either direction
    (essential for maneuvers like "back up, jump onto a brick, jump forward
    over an obstacle with enemies on top") plus DOWN so Mario can enter
    downward pipes:
        0 NOOP           1 RIGHT          2 LEFT          3 JUMP
        4 RIGHT+JUMP     5 RIGHT+RUN      6 RIGHT+RUN+JUMP
        7 LEFT+JUMP      8 LEFT+RUN       9 LEFT+RUN+JUMP
        10 DOWN  (crouch / enter pipe when standing on one)

    Tiles: the 16x20 game area with every tile replaced by its meaning
    (`mario_tile_lut`): -1.0 Mario, 0.0 empty, 0.5 ground and pipes, 0.6
    question / pushable blocks, 0.80 coin, 0.85 power-up, 1.0 enemy. Seven
    HUD scalars (lives, coins, timer, x, world, level, power-up) are written
    into cells (0, 0..6); see `_tiles`.

    Per-step reward:
        + progress_weight * new_max_x_delta   (ONLY reward for new territory;
                                                re-covering ground gives 0,
                                                so back-and-forth cannot farm)
        + coin_weight    * dcoins             (each coin picked up)
        + score_weight    * dscore            (enemies / items / coins — SML
                                                also adds 100 score per coin,
                                                so a coin is worth
                                                coin_weight + 100*score_weight)
        - time_penalty                        (small per-step cost — pressure
                                                against noop / oscillation)
      and, in any mode, +completion_bonus on level clear / -death_penalty on
      death (that step carries no progress / coin / score / time terms).

    Five modes, decided by `start_level`:
      - None / "default" (CAMPAIGN): play through the game. Death does NOT
        end the episode — Mario respawns at level start, life count
        decremented, max_x tracker resets. Level clear also doesn't end
        the episode — Mario transitions naturally to the next level and
        max_x resets. Episode ends only on `game_over()` (all lives
        exhausted) or the stuck timeout.
      - "W-L" (FIXED): each reset loads the save-state for that level.
        Any death or level clear ends the episode; next reset replays the
        same level (so lives are effectively infinite for training).
      - "random" (RANDOM): each reset picks a random usable level from
        SML_ALL_LEVELS and loads its state. Death or level clear ends the
        episode so the next reset picks a new level — Mario is always
        training on varied content.
      - "sequential" (SEQUENTIAL): cycles through SML_ALL_LEVELS. The
        cursor only ADVANCES on a real level clear; death retries the
        same level with fresh lives (via state reload).
      - "marathon" (MARATHON): one life through every usable level in one
        episode, always from 1-1. On level clear we IMMEDIATELY force-load
        the next level's save-state — the in-game level-end cutscene
        (Mario walking off screen, bonus countdown, intro screen) is
        skipped entirely. ANY death ends the episode and the next episode
        starts at 1-1 again (so does an exhausted time budget or a stall).
        Clearing the last usable level ends the episode with a big bonus.
        Training, evaluation and playback all play the same marathon.

    Rewards fire in every mode: +completion_bonus on level clear,
    -death_penalty on death (whether or not the episode ends). Both events
    are detected the step they happen via SML's game-state byte
    (ADDR_GAME_STATE) rather than ~20-70 steps later when PyBoy's life
    counter / world tuple catch up, so episodes that end on a clear or a
    death end immediately and no samples are spent inside cutscenes.
    """

    ACTION_NAMES = (
        "NOOP", "RIGHT", "LEFT", "JUMP",
        "RIGHT+JUMP", "RIGHT+RUN", "RIGHT+RUN+JUMP",
        "LEFT+JUMP", "LEFT+RUN", "LEFT+RUN+JUMP",
        "DOWN",
    )
    ACTIONS = (
        (),                     # 0 NOOP
        (RIGHT,),               # 1 RIGHT
        (LEFT,),                # 2 LEFT
        (A,),                   # 3 JUMP
        (RIGHT, A),             # 4 RIGHT+JUMP
        (RIGHT, B),             # 5 RIGHT+RUN
        (RIGHT, A, B),          # 6 RIGHT+RUN+JUMP
        (LEFT, A),              # 7 LEFT+JUMP
        (LEFT, B),              # 8 LEFT+RUN
        (LEFT, A, B),           # 9 LEFT+RUN+JUMP
        (DOWN,),                # 10 DOWN
    )

    def __init__(
        self,
        pyboy: PyBoy,
        frame_skip: int = 4,
        stuck_steps: int = DEFAULT_STALL_STEPS,
        time_budget: int = DEFAULT_TIME_BUDGET,
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
        super().__init__(pyboy, frame_skip=frame_skip, obs_type=obs_type,
                         tick_callback=tick_callback, stuck_steps=stuck_steps,
                         time_budget=time_budget, time_penalty=time_penalty,
                         death_penalty=death_penalty)
        # Two ways an attempt can be cut short without a death:
        #   time_budget  timer units (of TIMER_START) an attempt may use;
        #                0 = the whole in-game timer
        #   stuck_steps  steps without a new furthest x before truncation;
        #                0 = off (default). Only the time budget then limits
        #                an attempt, so Mario can use his time.
        self.progress_weight = progress_weight
        self.coin_weight = coin_weight
        self.score_weight = score_weight
        self.completion_bonus = completion_bonus
        # When the action includes JUMP, hold A for `frame_skip + jump_hold_bonus`
        # additional ticks so Mario gets a real jump (Super Mario Land jump
        # height scales with A-hold duration up to ~12 frames).
        self.jump_hold_bonus = jump_hold_bonus
        # Starting level (see the class docstring): None = campaign, a
        # (w, l) tuple = fixed, "random", "sequential" or "marathon".
        self.start_level = parse_start_level(start_level)
        self._level_rng = _random_mod.Random()
        self._campaign_mode = self.start_level is None
        # Sequential mode: which entry of SML_ALL_LEVELS is being played; it
        # only advances after a real clear (flag set by step, used by reset).
        self._sequential_idx = 0
        self._sequential_advance_pending = False
        # Marathon mode: the level being played and the clears so far in
        # this episode. Every episode is the same marathon (from 1-1, one
        # life) whoever runs it, so training, evaluation and the Preview
        # screen mean the same thing.
        self._marathon_idx = 0
        self._marathon_clears = 0
        # A clear or death is credited once, the step the game-state byte
        # flips; the later world flip / life-counter drop must not credit it
        # again.
        self._clear_credited = False
        self._death_credited = False
        self._jump_actions = frozenset(i for i, evts in enumerate(self.ACTIONS) if A in evts)
        self._last_x = 0
        self._last_lives = 0
        self._last_world = (0, 0)
        self._last_score = 0
        self._last_coins = 0

    def hold_frames(self, action: int) -> int:
        # Hold ALL buttons longer when the action includes JUMP so Mario
        # actually clears tall obstacles (4 frames give ~4 tiles, which
        # won't clear a pipe with a goomba on it).
        return self.frame_skip + (self.jump_hold_bonus if action in self._jump_actions else 0)

    def _tiles(self) -> np.ndarray:
        # The game area (background tiles with the sprites drawn in) as
        # meanings: see mario_tile_lut. A fresh array each call.
        arr = MARIO_TILE_LUT[np.asarray(self.gw.game_area(), dtype=np.uint16)]
        # The HUD row collapses to 0.0 in this representation, so the agent
        # would know nothing about lives, coins, the timer, where it is in
        # the level or which level it plays. Seven normalised numbers go
        # into the top-left cells instead (the obs shape does not change).
        w, l = self.gw.world
        arr[0, 0] = min(max(self.gw.lives_left, 0), 9) / 9.0        # lives:  0 .. ~1
        arr[0, 1] = min(self.gw.coins, 99) / 99.0                    # coins:  0 .. ~1
        arr[0, 2] = min(max(self.gw.time_left, 0), 400) / 400.0      # timer:  0 .. 1
        arr[0, 3] = min(self.gw.level_progress, 4096) / 4096.0       # current x in level
        arr[0, 4] = min(max(int(w) - 1, 0), 3) / 3.0                 # world:  0, 1/3, 2/3, 1
        arr[0, 5] = min(max(int(l) - 1, 0), 2) / 2.0                 # level:  0, 1/2, 1
        # Power-up decides what Mario can survive and do (take a hit, break
        # bricks, throw superballs). Small Mario reads 0, so models trained
        # before this cell existed see the same input until a mushroom.
        arr[0, 6] = self.power_state() / 2.0                         # small 0, super .5, superball 1
        return arr

    def power_state(self) -> int:
        """POWER_SMALL / POWER_SUPER / POWER_SUPERBALL (see games.power_state)."""
        return power_state(self.pyboy)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self._level_rng.seed(seed)
        target = None
        if self.start_level == "random":
            target = self._level_rng.choice(SML_ALL_LEVELS)
        elif self.start_level == "sequential":
            if self._sequential_advance_pending:
                self._sequential_idx = (self._sequential_idx + 1) % len(SML_ALL_LEVELS)
                self._sequential_advance_pending = False
            target = SML_ALL_LEVELS[self._sequential_idx]
        elif self.start_level == "marathon":
            self._marathon_clears = 0
            self._marathon_idx = 0
            target = SML_ALL_LEVELS[0]
        elif isinstance(self.start_level, tuple):
            target = self.start_level

        if target is None:
            self._load_start()
        elif not self._load_level_state(target):
            # No save-state for the level: the slow path through the
            # wrapper (may hang for the levels it cannot boot;
            # `prepare_level_states` is meant to run first).
            self.gw.set_world_level(target[0], target[1])
            self.gw.start_game()
            self._started = True
        self._begin_attempt()
        return self._obs(), {}

    def _load_level_state(self, target: tuple[int, int]) -> bool:
        """Load the save-state of level `target` (the fast, race-free way to
        switch levels). False if there is no usable state file."""
        state_file = _level_state_path(*target)
        if not (state_file.exists() and state_file.stat().st_size > 1000):
            return False
        # The wrapper's "started" flag must be set for helpers such as
        # level_progress / lives_left to read the right memory after a load.
        if not self._started:
            self.gw.start_game()
            self._started = True
        with open(state_file, "rb") as f:
            self.pyboy.load_state(f)
        # Memory-mapped readings stay stale until a few frames ran; without
        # this the first step sees the life counter "drop" and ends the
        # episode at once.
        for _ in range(4):
            self.pyboy.tick()
        return True

    def _begin_attempt(self) -> None:
        """Start a fresh attempt from the emulator's current readings: reward
        deltas, the furthest-x tracker, the stall counter and the one-shot
        clear / death credits all reset."""
        self._last_x = self.gw.level_progress
        self._progress(self._last_x, restart=True)
        self._last_lives = self.gw.lives_left
        self._last_world = tuple(self.gw.world)
        self._last_score = self.gw.score
        self._last_coins = self.gw.coins
        self._clear_credited = False
        self._death_credited = False

    def _evaluate(self, action: int) -> tuple[float, bool, bool, dict]:
        x = self.gw.level_progress
        lives = self.gw.lives_left
        world = tuple(self.gw.world)
        score = self.gw.score
        coins = self.gw.coins
        is_game_over = self.gw.game_over()
        game_state = self.pyboy.get_memory_value(ADDR_GAME_STATE)

        # Level clear: credited the step Mario touches the goal (game-state
        # 0x07), ~70 steps before PyBoy's `world` flips. The world flip is
        # kept as a fallback detector and, in campaign mode, marks the
        # moment the next level actually starts.
        world_changed = world != self._last_world
        clear_now = game_state in SML_CLEAR_STATES and not self._clear_credited
        if clear_now:
            self._clear_credited = True
        level_cleared = clear_now or (world_changed and not self._clear_credited)
        if world_changed:
            self._clear_credited = False

        # Death: credited the step Mario dies (game-state 0x04 / 0x01), ~20
        # steps before the life counter drops at respawn. The life-counter
        # drop is the fallback detector for any death type whose state
        # value isn't mapped.
        lives_dropped = lives < self._last_lives
        death_now = game_state in SML_DEATH_STATES and not self._death_credited
        if death_now:
            self._death_credited = True
        died = death_now or (lives_dropped and not self._death_credited)
        if lives_dropped:
            self._death_credited = False

        # Furthest-x tracker and stall counter. A new attempt (respawn or the
        # next level in campaign mode) starts the count afresh; a step that
        # died or cleared does not count as standing still.
        if self._campaign_mode and (lives_dropped or world_changed):
            gained = self._progress(x, restart=True)
        else:
            gained = self._progress(x, count=not (died or level_cleared))

        reward = 0.0
        if level_cleared:
            reward += self.completion_bonus
        if died:
            reward -= self.death_penalty
        if not (level_cleared or died):
            # Only NEW forward territory pays; re-covering or backtracking
            # gives 0 (not extra negative) so the agent is free to reposition
            # (back up to jump on a brick) at only the per-step time cost.
            reward += self.progress_weight * gained
            dcoins = coins - self._last_coins
            if dcoins > 0:
                reward += self.coin_weight * dcoins
            dscore = score - self._last_score        # enemy kills, pipe bonuses…
            if dscore > 0:
                reward += self.score_weight * dscore
            # Standing still or backtracking still costs, like the in-game
            # timer: the only way to net positive reward is to get on.
            reward -= self.time_penalty

        # Sequential mode: only ADVANCE to the next level on a real clear; a
        # death retries the same level (the next reset consumes the flag).
        if self.start_level == "sequential" and level_cleared:
            self._sequential_advance_pending = True

        # Attempt limits. Time budget: checked only while actually playing
        # (game_state 0), so the level-end countdown, which drains the timer
        # into score, cannot trigger it. Every attempt starts with the timer
        # at TIMER_START (level start, respawn and marathon level loads all
        # reset it). Stall: steps without a new furthest x (off by default).
        time_used = max(0, TIMER_START - int(self.gw.time_left))
        in_play = game_state == 0 and not (died or level_cleared)
        over_budget = in_play and self.time_budget > 0 and time_used >= self.time_budget
        stalled = in_play and self._stalled()

        def info(**extra) -> dict:
            base = {
                "x": x, "max_x": self._max_x, "lives": lives,
                "world": world, "coins": coins, "score": score,
                "stuck": self._stuck, "game_over": is_game_over,
                "game_state": game_state, "died": died, "level_cleared": level_cleared,
                "power": POWER_NAMES[self.power_state()],
                "time_used": time_used, "time_budget_exceeded": over_budget, "stalled": stalled,
            }
            base.update(extra)
            return base

        def loaded_level_info(**extra) -> dict:
            """Info after `_reload_level_state`: trackers describe the new level."""
            return info(x=self._last_x, max_x=self._max_x, lives=self._last_lives,
                        world=self._last_world, coins=self._last_coins, score=self._last_score,
                        stuck=0, game_over=False, game_state=0, time_used=0, **extra)

        # Marathon mode: one life through every usable level. On a clear we
        # do NOT wait for the in-game level-end cutscene (~70 env-steps of
        # no learning signal): `level_cleared` fires the step the goal is
        # touched and the next level's save-state is loaded.
        if self.start_level == "marathon" and level_cleared:
            self._marathon_clears += 1
            self._marathon_idx += 1
            if self._marathon_idx >= len(SML_ALL_LEVELS):
                reward += self.completion_bonus * 3.0             # every level in one run
                self._remember(x, lives, world, score, coins)
                return reward, True, False, info(marathon_done=True,
                                                 marathon_clears=self._marathon_clears)
            next_target = SML_ALL_LEVELS[self._marathon_idx]
            if self._load_level_state(next_target):
                self._begin_attempt()
                return reward, False, False, loaded_level_info(
                    marathon_clears=self._marathon_clears, marathon_next_level=next_target)
            # No cached save-state for `next_target`: end the episode rather
            # than soft-locking in the level-end sequence (a misconfiguration;
            # `prepare_level_states` should have run first).
            self._remember(x, lives, world, score, coins)
            return reward, True, False, info(marathon_clears=self._marathon_clears,
                                             marathon_missing_state=next_target)

        # Termination by mode.
        if self._campaign_mode:
            # Death and level clear are transitions within the same episode;
            # only game over ends it, and a death on the last life IS game
            # over, so end right there instead of sitting through the
            # game-over screen. (`death_now`, not `died`: on the fallback path
            # `lives` has already dropped, so 0 there means "now on the last life".)
            terminated = is_game_over or (death_now and lives == 0)
        elif self.start_level == "marathon":
            terminated = died                  # the next reset() starts over at 1-1
        else:
            terminated = died or level_cleared # fixed / random / sequential

        # A truncation is only reported when the episode did not already end
        # for a real reason; SB3 bootstraps the value of truncated states.
        truncated = not terminated and (stalled or over_budget or is_game_over)
        result = info()
        self._remember(x, lives, world, score, coins)
        return reward, terminated, truncated, result

    def _remember(self, x, lives, world, score, coins) -> None:
        """Store this step's readings for the next step's deltas."""
        self._last_x = x
        self._last_lives = lives
        self._last_world = world
        self._last_score = score
        self._last_coins = coins


class KirbyEnv(GameBoyEnv):
    """Kirby's Dream Land: get further, score, and stay alive.

    Discrete(11) action space:
        0 NOOP        1 RIGHT        2 LEFT          3 JUMP (A; again in the
        air = fly)    4 RIGHT+JUMP   5 LEFT+JUMP     6 INHALE (B; also spits
        what Kirby holds)            7 RIGHT+INHALE  8 LEFT+INHALE
        9 UP (fly / enter a door)    10 DOWN (swallow / duck)

    Progress: the game has no position counter, so the screen's horizontal
    scroll is accumulated (it moves right as Kirby advances; wrapping every
    256 pixels is unwound) and Kirby's own screen position (ADDR_KIRBY_X)
    is added. Only NEW furthest progress pays, like Mario's.

    Per-step reward:
        + progress_weight * new furthest progress (pixels)
        + score_weight    * dscore     (enemies, items)
        - health_weight   * health lost
        - time_penalty
      and -death_penalty on a death. Every attempt starts at the beginning
      of the game (PyBoy's saved start); a death or the game over ends the
      episode. Kirby has no timer, so `time_budget` does nothing here; the
      stall rule (`stuck_steps`) and a hard cap of KIRBY_ATTEMPT_STEPS steps
      make sure an evaluation episode always ends.

    Tiles: PyBoy's `game_area()` (the 16 rows of the play field; Kirby,
    enemies and items are drawn in as their tile identifiers), scaled into
    [0, 1], with health, lives and progress in cells (0, 0..2).
    """

    ACTION_NAMES = (
        "NOOP", "RIGHT", "LEFT", "JUMP", "RIGHT+JUMP", "LEFT+JUMP",
        "INHALE", "RIGHT+INHALE", "LEFT+INHALE", "UP", "DOWN",
    )
    ACTIONS = (
        (),                     # 0 NOOP
        (RIGHT,),               # 1 RIGHT
        (LEFT,),                # 2 LEFT
        (A,),                   # 3 JUMP
        (RIGHT, A),             # 4 RIGHT+JUMP
        (LEFT, A),              # 5 LEFT+JUMP
        (B,),                   # 6 INHALE
        (RIGHT, B),             # 7 RIGHT+INHALE
        (LEFT, B),              # 8 LEFT+INHALE
        (UP,),                  # 9 UP
        (DOWN,),                # 10 DOWN
    )

    def __init__(self, pyboy: PyBoy, frame_skip: int = 4, stuck_steps: int = DEFAULT_STALL_STEPS,
                 time_budget: int = DEFAULT_TIME_BUDGET, progress_weight: float = 1.0,
                 score_weight: float = 0.05, health_weight: float = 50.0,
                 time_penalty: float = 0.03, death_penalty: float = 500.0,
                 obs_type: str = "pixels", tick_callback=None, start_level=None,
                 max_steps: int = KIRBY_ATTEMPT_STEPS):
        super().__init__(pyboy, frame_skip=frame_skip, obs_type=obs_type,
                         tick_callback=tick_callback, stuck_steps=stuck_steps,
                         time_budget=time_budget, time_penalty=time_penalty,
                         death_penalty=death_penalty)
        why = check_start_level("kirby", start_level)
        if why:
            raise ValueError(why)
        self.progress_weight = progress_weight
        self.score_weight = score_weight
        self.health_weight = health_weight
        self.max_steps = max_steps
        self._scroll = 0                    # accumulated horizontal scroll, in pixels
        self._last_scx = 0
        self._steps = 0
        self._last_score = 0
        self._last_health = 0
        self._last_lives = 0

    def _scroll_x(self) -> int:
        return int(self.pyboy.botsupport_manager().screen().tilemap_position()[0][0])

    def progress(self) -> int:
        """How far Kirby has come: scroll so far plus his place on the screen."""
        return self._scroll + int(self.pyboy.get_memory_value(ADDR_KIRBY_X))

    def _tiles(self) -> np.ndarray:
        arr = np.asarray(self.gw.game_area(), dtype=np.float32) / self.TILE_NORM
        arr = np.clip(arr, 0.0, 1.0)
        arr[0, 0] = min(max(int(self.gw.health), 0), KIRBY_MAX_HEALTH) / KIRBY_MAX_HEALTH
        arr[0, 1] = min(max(int(self.gw.lives_left), 0), 9) / 9.0
        arr[0, 2] = min(self.progress(), 8192) / 8192.0
        return arr

    def _begin_attempt(self) -> None:
        self._scroll = 0
        self._last_scx = self._scroll_x()
        self._steps = 0
        self._progress(self.progress(), restart=True)
        self._last_score = int(self.gw.score)
        self._last_health = int(self.gw.health)
        self._last_lives = int(self.gw.lives_left)

    def _evaluate(self, action: int) -> tuple[float, bool, bool, dict]:
        scx = self._scroll_x()
        self._scroll += (scx - self._last_scx + 128) % 256 - 128      # unwind the 256 px wrap
        self._last_scx = scx
        self._steps += 1
        x = self.progress()
        score, health, lives = int(self.gw.score), int(self.gw.health), int(self.gw.lives_left)
        is_game_over = bool(self.gw.game_over())
        died = lives < self._last_lives or is_game_over
        gained = self._progress(x, count=not died)

        reward = 0.0
        if died:
            reward -= self.death_penalty
        else:
            reward += self.progress_weight * gained
            dscore = score - self._last_score
            if dscore > 0:
                reward += self.score_weight * dscore
            lost = self._last_health - health
            if lost > 0:
                reward -= self.health_weight * lost
            reward -= self.time_penalty

        terminated = died
        stalled = not died and self._stalled()
        capped = not died and self._steps >= self.max_steps
        truncated = not terminated and (stalled or capped)
        info = {
            "x": x, "max_x": self._max_x, "lives": lives, "score": score, "health": health,
            "stuck": self._stuck, "game_over": is_game_over, "died": died,
            "level_cleared": False, "stalled": stalled, "step_cap_reached": capped,
        }
        self._last_score, self._last_health, self._last_lives = score, health, lives
        return reward, terminated, truncated, info


MARIO_TILE_LUT = mario_tile_lut()

ENV_CLASSES: dict[str, type[GameBoyEnv]] = {"mario": MarioEnv, "kirby": KirbyEnv}


def make_env(game: str, pyboy: PyBoy, *, frame_skip: int = 4, obs_type: str = "pixels",
             start_level=None, time_budget: int = DEFAULT_TIME_BUDGET,
             stall_steps: int = DEFAULT_STALL_STEPS, tick_callback=None) -> GameBoyEnv:
    """The game's environment class around an open PyBoy."""
    cls = ENV_CLASSES.get(game)
    if cls is None:
        raise ValueError(f"'{game}' has no AIboy environment; trainable games: "
                         f"{', '.join(ENV_CLASSES)}")
    why = check_start_level(game, start_level)
    if why:
        raise ValueError(why)
    kwargs = dict(frame_skip=frame_skip, obs_type=obs_type, time_budget=time_budget,
                  stuck_steps=stall_steps, tick_callback=tick_callback)
    if GAMES[game].levels:
        kwargs["start_level"] = start_level
    return cls(pyboy, **kwargs)


def wrap_vec_env(vec: VecEnv, obs_type: str, frame_stack: int) -> VecEnv:
    """Apply the observation wrappers a policy trained on `obs_type` expects.

    Pixels are transposed to channels-first for the CNN and stacked along
    that axis; tile grids stay HWC and stack along the channel dimension.
    Training, evaluation, CLI playback and the GUI all wrap through here so
    a saved model always sees the observation shape it was trained on.
    """
    if obs_type == "pixels":
        vec = VecTransposeImage(vec)
    if frame_stack > 1:
        order = "first" if obs_type == "pixels" else "last"
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order=order)
    return vec


def make_pyboy_env(
    game: str = "mario",
    window_type: str = "null",
    action_repeat: int = 4,
    seed: int | None = None,
    rom_dir: str | Path = ROM_DIR,
    emulation_speed: int | None = None,
    obs_type: str = "pixels",
    start_level=None,
    time_budget: int = DEFAULT_TIME_BUDGET,
    stall_steps: int = DEFAULT_STALL_STEPS,
) -> gym.Env:
    """Open the game's ROM in PyBoy and build its AIboy environment (see
    ENV_CLASSES: hold-button actions, shaped reward, pixels or tiles)."""
    if game not in GAMES:
        raise ValueError(f"Unknown game '{game}'. Available: {sorted(GAMES)}")
    if game not in ENV_CLASSES:
        raise ValueError(f"'{game}' cannot be trained: AIboy has environments for "
                         f"{', '.join(ENV_CLASSES)} (any ROM can still be played by hand)")
    spec = GAMES[game]
    rom_path = Path(rom_dir) / spec.rom_file
    if not rom_path.exists():
        raise FileNotFoundError(
            f"ROM not found: {rom_path}. Place your legally obtained copy of the "
            f"game at that path (see README → Install).")
    why = check_start_level(game, start_level)
    if why:
        raise ValueError(why)

    # Tile obs reads the tile map, so rendering can be disabled in null-window
    # mode for a small speedup. Pixel obs and the SDL2 window both require
    # a live framebuffer.
    disable_renderer = (window_type == "null" and obs_type != "pixels")
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

    try:
        env = make_env(game, pyboy, frame_skip=action_repeat, obs_type=obs_type,
                       start_level=start_level, time_budget=time_budget,
                       stall_steps=stall_steps)
    except Exception:
        pyboy.stop(save=False)
        raise
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
    time_budget: int = DEFAULT_TIME_BUDGET,
    stall_steps: int = DEFAULT_STALL_STEPS,
) -> gym.Env:
    """Picklable factory for SubprocVecEnv workers."""
    env = make_pyboy_env(
        game=game,
        window_type=window_type,
        action_repeat=action_repeat,
        seed=seed,
        obs_type=obs_type,
        start_level=start_level,
        time_budget=time_budget,
        stall_steps=stall_steps,
    )
    return Monitor(env)
