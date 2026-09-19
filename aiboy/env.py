"""The Super Mario Land Gymnasium environment AIboy trains in (PyBoy underneath)."""
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
    ADDR_GAME_STATE, ADDR_POWERUP_STATE, ADDR_SUPERBALL, DEFAULT_STALL_STEPS, DEFAULT_TIME_BUDGET,
    GAMES, LEVEL_MODES, LEVEL_STATES_DIR, TIMER_START,
    MULTI_LEVEL_MODES, OBS_TYPES, POWER_NAMES, POWER_SMALL, POWER_SUPER, POWER_SUPERBALL,
    ROM_DIR, ROM_SUFFIXES, SML_ALL_LEVELS, SML_BROKEN_LEVELS, SML_CLEAR_STATES,
    SML_DEATH_STATES, SUPPORTED_GAMES, GameSpec, RomInfo, _level_state_path,
    discover_roms, ensure_level_states, level_choices, level_targets, parse_start_level,
    power_state, prepare_level_states, probe_rom,
)


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
        216× smaller than pixels, trains ~10× faster on CPU. Seven HUD
        scalars (lives, coins, timer, x, world, level, power-up) are written
        into cells (0, 0..6); see `_obs`.

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

    metadata = {"render_modes": []}

    ACTION_NAMES = (
        "NOOP", "RIGHT", "LEFT", "JUMP",
        "RIGHT+JUMP", "RIGHT+RUN", "RIGHT+RUN+JUMP",
        "LEFT+JUMP", "LEFT+RUN", "LEFT+RUN+JUMP",
        "DOWN",
    )
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
        super().__init__()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type must be one of {OBS_TYPES}, got {obs_type!r}")
        self.pyboy = pyboy
        self.gw = pyboy.game_wrapper()
        self.frame_skip = frame_skip
        # Two ways an attempt can be cut short without a death:
        #   time_budget  timer units (of TIMER_START) an attempt may use;
        #                0 = the whole in-game timer
        #   stuck_steps  steps without a new furthest x before truncation;
        #                0 = off (default). Only the time budget then limits
        #                an attempt, so Mario can use his time.
        self.stuck_steps = stuck_steps
        self.time_budget = time_budget
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
        #   None           → SML's default (world 1-1) → CAMPAIGN mode
        #                     (respect lives, respawn on death, restart at
        #                     1-1 on game_over)
        #   (w, l)         → always start there (1-indexed, e.g. (2, 3))
        #                     (FIXED mode — die or clear ends episode)
        #   "random"       → pick a random level each reset
        #                     (RANDOM mode — die or clear ends episode)
        #   "sequential"   → cycle through SML_ALL_LEVELS; the index only
        #                     ADVANCES on a real level clear. Death retries
        #                     the same level (fresh lives via state reload).
        #   "marathon"     → one life through every usable level, from
        #                     1-1. On level clear we force-load the next
        #                     level's state IMMEDIATELY (skipping the ~5 s
        #                     in-game victory cutscene). ANY death ends the
        #                     episode; the next one starts at 1-1 again.
        #                     Clearing the last level ends the episode with
        #                     a large bonus.
        self.start_level = parse_start_level(start_level)
        self._level_rng = _random_mod.Random()
        # Campaign mode = play through the game respecting lives and level
        # progression (deaths respawn, level-clears advance). Only triggered
        # when no start_level is set.
        self._campaign_mode = self.start_level is None
        # Sequential mode state.
        # `_sequential_idx` = which entry in SML_ALL_LEVELS is currently
        # being played. It only advances after Mario actually CLEARS the
        # level; if he dies, the same level is retried with fresh lives
        # (loaded from the state file). Set by `step()` when it observes
        # a world change (level clear), consumed by the next `reset()`.
        self._sequential_idx = 0
        self._sequential_advance_pending = False
        # Marathon mode: index into SML_ALL_LEVELS of the level being played
        # and the number of clears so far in this episode. Every episode is
        # the same marathon (from 1-1, one life) whoever runs it, so a
        # training reward, an evaluation reward and what the Preview screen
        # shows all mean the same thing. The episode ends with a big bonus
        # after the last usable level.
        self._marathon_idx = 0
        self._marathon_clears = 0
        # Event bookkeeping for the instant clear / death detection (see
        # ADDR_GAME_STATE). A clear or death is credited once, the step the
        # game-state byte flips; the later world-flip / life-counter-drop
        # then must not credit it again.
        self._clear_credited = False
        self._death_credited = False
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
        # Power-up decides what Mario can survive and do (take a hit, break
        # bricks, throw superballs), so the policy should know it. Small
        # Mario reads 0, so models trained before this cell existed see the
        # exact same input until Mario picks up a mushroom.
        arr[0, 6] = self.power_state() / 2.0                         # small 0, super .5, superball 1
        return arr[..., np.newaxis]

    def power_state(self) -> int:
        """POWER_SMALL / POWER_SUPER / POWER_SUPERBALL (see games.power_state)."""
        return power_state(self.pyboy)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._level_rng.seed(seed)

        target = None
        if self.start_level == "random":
            target = self._level_rng.choice(SML_ALL_LEVELS)
        elif self.start_level == "sequential":
            # Advance only after Mario CLEARED the previous level. On death
            # we stay on the same level (state reload gives fresh lives).
            # Wraps around at the end of the list.
            if self._sequential_advance_pending:
                self._sequential_idx = (self._sequential_idx + 1) % len(SML_ALL_LEVELS)
                self._sequential_advance_pending = False
            target = SML_ALL_LEVELS[self._sequential_idx]
        elif self.start_level == "marathon":
            # Marathon: one life through the levels, always from the first.
            self._marathon_clears = 0
            self._marathon_idx = 0
            target = SML_ALL_LEVELS[0]
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
        self._sync_trackers()
        return self._obs(), {}

    def _sync_trackers(self) -> None:
        """Start a fresh attempt from the emulator's current readings: reward
        deltas, the furthest-x tracker, the stall counter and the one-shot
        clear / death credits all reset."""
        self._last_x = self.gw.level_progress
        self._max_x = self._last_x
        self._last_lives = self.gw.lives_left
        self._last_world = tuple(self.gw.world)
        self._last_score = self.gw.score
        self._last_coins = self.gw.coins
        self._stuck = 0
        self._clear_credited = False
        self._death_credited = False

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

        # Sequential mode: only ADVANCE to the next level on a real clear;
        # a death should retry the same level. Recorded as a flag for the
        # next reset() to consume.
        if self.start_level == "sequential" and level_cleared:
            self._sequential_advance_pending = True

        # Progress tracking for the stall rule. A new attempt (respawn or
        # next level in campaign mode) starts the count afresh.
        if self._campaign_mode and (lives_dropped or world_changed):
            self._max_x = x
            self._stuck = 0
        elif x > self._max_x:
            self._max_x = x
            self._stuck = 0
        elif not (died or level_cleared):
            self._stuck += 1

        # Attempt limits. Time budget: checked only while actually playing
        # (game_state 0), so the level-end countdown, which drains the timer
        # into score, cannot trigger it. Every attempt starts with the timer
        # at TIMER_START (level start, respawn and marathon level loads all
        # reset it). Stall: steps without a new furthest x (off by default).
        time_used = max(0, TIMER_START - int(self.gw.time_left))
        in_play = game_state == 0 and not (died or level_cleared)
        over_budget = in_play and self.time_budget > 0 and time_used >= self.time_budget
        stalled = in_play and self.stuck_steps > 0 and self._stuck >= self.stuck_steps

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
                # Cleared every usable level in one run: marathon win.
                reward += self.completion_bonus * 3.0
                self._remember(x, lives, world, score, coins)
                return self._obs(), float(reward), True, False, info(
                    marathon_done=True, marathon_clears=self._marathon_clears)
            next_target = SML_ALL_LEVELS[self._marathon_idx]
            if self._reload_level_state(next_target):
                return self._obs(), float(reward), False, False, loaded_level_info(
                    marathon_clears=self._marathon_clears, marathon_next_level=next_target)
            # No cached save-state for `next_target`: end the episode rather
            # than soft-locking in the level-end sequence. `prepare_level_states`
            # should have been called upfront, so this is a misconfiguration.
            self._remember(x, lives, world, score, coins)
            return self._obs(), float(reward), True, False, info(
                marathon_clears=self._marathon_clears, marathon_missing_state=next_target)

        # Termination by mode.
        if self._campaign_mode:
            # Death and level clear are transitions within the same episode;
            # only game over ends it, and a death on the last life IS game
            # over, so end right there instead of sitting through the
            # game-over screen. (`death_now`, not `died`: on the fallback path
            # `lives` has already dropped, so 0 there means "now on the last life".)
            terminated = is_game_over or (death_now and lives == 0)
        elif self.start_level == "marathon":
            # Any death ends the run; the next reset() starts over at 1-1.
            terminated = died
        else:
            # Fixed / random / sequential: any death or level clear ends the episode.
            terminated = died or level_cleared

        # A truncation is only reported when the episode did not already end
        # for a real reason; SB3 bootstraps the value of truncated states.
        truncated = not terminated and (stalled or over_budget or is_game_over)
        result = info()
        self._remember(x, lives, world, score, coins)
        return self._obs(), float(reward), terminated, truncated, result

    def _remember(self, x, lives, world, score, coins) -> None:
        """Store this step's readings for the next step's deltas."""
        self._last_x = x
        self._last_lives = lives
        self._last_world = world
        self._last_score = score
        self._last_coins = coins

    def _reload_level_state(self, target: tuple[int, int]) -> bool:
        """Mid-episode reload of a specific level's save-state. Used by
        marathon mode to jump directly to the next level on a clear
        (skipping the in-game victory cutscene). Refreshes all internal
        tracking so reward accounting starts clean for the new level.

        Returns True on success, False if the state file is missing or
        too small to be a valid save-state. Callers must handle False —
        continuing with stale trackers would soft-lock marathon mode.
        """
        state_file = _level_state_path(*target)
        if not (state_file.exists() and state_file.stat().st_size > 1000):
            return False
        with open(state_file, "rb") as f:
            self.pyboy.load_state(f)
        # Refresh memory-mapped state after load
        for _ in range(4):
            self.pyboy.tick()
        self._sync_trackers()
        return True

    def close(self):
        try:
            self.pyboy.stop(save=False)
        except Exception:
            pass


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
    """Build a PyBoy gym env.
    - 'mario' → MarioEnv (hold-button actions + shaped reward, pixels or tiles)
    - other games → PyBoy's default openai_gym + ActionRepeat (pixels only)
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game '{game}'. Available: {sorted(GAMES)}")
    spec = GAMES[game]
    rom_path = Path(rom_dir) / spec.rom_file
    if not rom_path.exists():
        raise FileNotFoundError(
            f"ROM not found: {rom_path}. Place your legally obtained copy of the "
            f"game at that path (see README → Install).")

    # Tile obs reads game_area(), so rendering can be disabled in null-window
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

    if game == "mario":
        env: gym.Env = MarioEnv(pyboy, frame_skip=action_repeat, obs_type=obs_type,
                                 start_level=start_level,
                                 time_budget=time_budget, stuck_steps=stall_steps)
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
