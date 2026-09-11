"""PyBoy gym environment factory for Game Boy AI training."""
from __future__ import annotations

import random as _random_mod
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy, WindowEvent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage

ROM_DIR = Path("ROMs")


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

# SML's game-state byte (mapped empirically by logging RAM transitions
# across clears / enemy deaths / pit deaths / game over on several levels):
#   0x00  playing
#   0x07 → 0x05 → 0x06  level-clear sequence: goal touched, walk-off,
#                       bonus-timer countdown (~70 env-steps at frame_skip 4
#                       before PyBoy's `world` tuple finally flips)
#   0x04 → 0x01  dying animation → waiting for respawn (pit deaths jump
#                straight to 0x01; 0x03 sometimes precedes 0x04 for one step)
#   0x02 / 0x08  next level / respawn loading (transient)
#   0x3A  game-over screen (PyBoy's game_over() checks 0xC0A4 == 0x39)
# Reading it lets the env credit a clear or a death the step it happens
# instead of ~20-70 steps later — a large training-throughput win since
# every step in a cutscene / death animation is a wasted sample.
# Not yet observed (no trained agent enters pipes): whether a pipe
# transition uses one of the death values. The life-counter fallback
# would not fire in that case, so a pipe entry would end a fixed/random
# episode as a "death". If that ever shows up, narrow SML_DEATH_STATES.
ADDR_GAME_STATE = 0xFFB3
SML_CLEAR_STATES = frozenset({0x05, 0x06, 0x07})
SML_DEATH_STATES = frozenset({0x01, 0x04})

# Where per-level save-state files live. Gitignored via models/.
LEVEL_STATES_DIR = Path("models") / "mario" / "_level_states"

# Level modes that are not a fixed "W-L" level (see MarioEnv docstring).
LEVEL_MODES = ("default", "random", "sequential", "marathon")
MULTI_LEVEL_MODES = ("random", "sequential", "marathon")


def level_choices() -> list[str]:
    """Every valid `--start-level` value: the four modes, then each usable level."""
    return list(LEVEL_MODES) + [f"{w}-{l}" for (w, l) in SML_ALL_LEVELS]


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


def level_targets(start_level) -> list[tuple[int, int]]:
    """Levels whose save-state a `start_level` spec needs. Empty for campaign."""
    parsed = parse_start_level(start_level)
    if parsed is None:
        return []
    if isinstance(parsed, tuple):
        return [parsed]
    return list(SML_ALL_LEVELS)


def prepare_level_states(start_level, rom_path: Path | None = None) -> list[tuple[int, int]]:
    """Bootstrap the save-states a `start_level` spec needs (idempotent).

    Returns the levels that could NOT be prepared (empty on success). Call
    it once in the parent process before spawning workers so they simply
    load the files instead of each running the fragile `start_game` path.
    """
    targets = level_targets(start_level)
    if not targets:
        return []
    rom = Path(rom_path) if rom_path is not None else ROM_DIR / GAMES["mario"].rom_file
    ok = set(ensure_level_states(rom, targets))
    return [t for t in targets if t not in ok]


def parse_start_level(spec):
    """Parse a start_level spec into None, "random", "sequential",
    "marathon", or a (world, level) tuple.

    Accepts: None, "default", "random", "sequential", "marathon",
    "W-L" string (e.g. "2-3"), or a (world, level) tuple.
    """
    if spec is None or spec == "default":
        return None
    if spec == "random":
        return "random"
    if spec == "sequential":
        return "sequential"
    if spec in ("marathon", "all_levels"):  # `all_levels` is a back-compat alias
        return "marathon"
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
    # Only Super Mario Land has a custom env, reward shaping and level
    # modes. The other titles run through PyBoy's generic openai_gym
    # wrapper (pixels only) and are experimental CLI-only extras.
    supported: bool = False


GAMES = {
    "mario": GameSpec("mario.gb", "SUPER MARIOLAN", supported=True),
    "kirby": GameSpec("kirby.gb", "KIRBY DREAM LA"),
    "wario": GameSpec("wario.gb", "SUPERMARIOLAND"),   # Super Mario Land 3: Wario Land
}
SUPPORTED_GAMES = tuple(name for name, spec in GAMES.items() if spec.supported)
ROM_SUFFIXES = (".gb", ".gbc")


@dataclass
class RomInfo:
    """A ROM file in the ROM folder plus what we know about running it.

    `title` / `has_wrapper` / `error` are filled in by `probe_rom`; until
    then `probed` is False and the status is "checking".
    """
    name: str                       # file stem, e.g. "mario"; also the --game value
    path: Path
    probed: bool = False
    title: str | None = None        # cartridge title reported by PyBoy
    has_wrapper: bool | None = None # PyBoy ships a game wrapper for this cartridge
    error: str | None = None        # boot failure, if any

    @property
    def spec(self) -> GameSpec | None:
        return GAMES.get(self.name)

    def status(self) -> tuple[str, str]:
        """(level, text) where level is one of
        "ok"            fully supported (custom env, presets, level modes)
        "experimental"  PyBoy has a wrapper; generic pixel env, CLI only
        "unsupported"   cannot run (no wrapper, unknown cartridge, boot error)
        "checking"      probe still running
        """
        if self.error:
            return "unsupported", f"cannot boot ROM: {self.error}"
        spec = self.spec
        if spec is not None and spec.supported:
            if self.probed and self.title != spec.cartridge_title:
                return "unsupported", (f"expected cartridge '{spec.cartridge_title}' "
                                       f"but the ROM reports '{self.title}'")
            return "ok", f"{self.title or spec.cartridge_title} — supported"
        if not self.probed:
            return "checking", "checking whether PyBoy can run this ROM…"
        if self.has_wrapper:
            return "experimental", (f"{self.title} — PyBoy game wrapper only; no custom "
                                    f"environment, presets or level modes in this version "
                                    f"(CLI: --game {self.name}, pixels)")
        return "unsupported", f"{self.title} — no PyBoy game wrapper; cannot be trained or played"

    @property
    def runnable(self) -> bool:
        return self.status()[0] == "ok"


def discover_roms(rom_dir: str | Path = ROM_DIR) -> list[RomInfo]:
    """Every .gb / .gbc file in the ROM folder, known games first."""
    rom_dir = Path(rom_dir)
    if not rom_dir.exists():
        return []
    files = [p for p in rom_dir.iterdir() if p.is_file() and p.suffix.lower() in ROM_SUFFIXES]
    order = {name: i for i, name in enumerate(GAMES)}
    files.sort(key=lambda p: (order.get(p.stem, len(order)), p.stem.lower()))
    return [RomInfo(name=p.stem, path=p) for p in files]


def probe_rom(rom_path: str | Path, timeout_sec: int = 30) -> tuple[str | None, bool, str | None]:
    """Boot a ROM headless in a subprocess and report
    (cartridge_title, has_game_wrapper, error). Isolated so a ROM PyBoy
    cannot handle can neither hang nor crash the caller."""
    import json
    import subprocess
    import sys as _sys
    code = (
        "import json, sys;"
        "from pyboy import PyBoy;"
        f"p = PyBoy({str(Path(rom_path).resolve())!r}, window_type='null', "
        "game_wrapper=True, disable_renderer=True);"
        "print(json.dumps({'title': p.cartridge_title(), "
        "'wrapper': p.game_wrapper() is not None}));"
        "p.stop(save=False)"
    )
    try:
        out = subprocess.run([_sys.executable, "-c", code], timeout=timeout_sec,
                             capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None, False, "PyBoy did not start within the time limit"
    if out.returncode != 0:
        last = (out.stderr.strip().splitlines() or ["unknown error"])[-1]
        return None, False, last[:200]
    try:
        line = [l for l in out.stdout.splitlines() if l.startswith("{")][-1]
        data = json.loads(line)
    except (IndexError, json.JSONDecodeError):
        return None, False, "unexpected output from PyBoy"
    return str(data.get("title", "")).strip(), bool(data.get("wrapper")), None


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
      - "marathon" (MARATHON): single-attempt speedrun through every
        usable level in one episode. On level clear we IMMEDIATELY
        force-load the next level's save-state — the in-game level-end
        cutscene (Mario walking off screen, bonus countdown, intro
        screen) is skipped entirely. ANY death ends the episode.
        Clearing the last usable level ends the episode with a big bonus.

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
        #   "marathon"     → single-attempt speedrun through every usable
        #                     level. On level clear we force-load the next
        #                     level's state IMMEDIATELY (skipping the ~5 s
        #                     in-game victory cutscene). ANY death ends the
        #                     episode. Clearing the last level ends the
        #                     episode with a large bonus.
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
        # Marathon mode: count of level clears within the current episode.
        # When it hits len(SML_ALL_LEVELS) the episode ends with a big bonus.
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
        return arr[..., np.newaxis]

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
            # Marathon: start with a natural game boot at 1-1 (like campaign
            # mode). The game handles level-to-level transitions itself; we
            # just count clears and intervene on game_over. Reset the clear
            # counter for the new episode.
            self._marathon_clears = 0
            target = None  # natural start, no state load
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
        self._clear_credited = False
        self._death_credited = False
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

        # Marathon mode: single-attempt speedrun through every usable level.
        # ANY death ends the episode. On level clear we do NOT wait for
        # the in-game level-end cutscene (walk-off + bonus countdown, ~70
        # env-steps that burn training samples for zero learning signal).
        # `level_cleared` fires the step the goal is touched, and we
        # force-load the next level's save-state right there.
        if self.start_level == "marathon" and level_cleared:
            self._marathon_clears += 1
            if self._marathon_clears >= len(SML_ALL_LEVELS):
                # Cleared every usable level in one run → marathon win.
                reward += self.completion_bonus * 3.0
                self._last_x = x
                self._last_lives = lives
                self._last_world = world
                self._last_score = score
                self._last_coins = coins
                self._max_x = x
                self._stuck = 0
                return self._obs(), float(reward), True, False, {
                    "x": x, "max_x": x, "lives": lives,
                    "world": world, "coins": coins, "score": score,
                    "stuck": 0, "game_over": is_game_over,
                    "game_state": game_state, "died": died, "level_cleared": True,
                    "marathon_done": True,
                    "marathon_clears": self._marathon_clears,
                }
            # Not the final clear — jump straight to the next usable level.
            # `_reload_level_state` handles load_state + memory refresh +
            # resets all per-level trackers, so we return immediately with
            # the accumulated clear reward and skip the rest of step().
            next_target = SML_ALL_LEVELS[self._marathon_clears]
            if self._reload_level_state(next_target):
                return self._obs(), float(reward), False, False, {
                    "x": self._last_x, "max_x": self._max_x,
                    "lives": self._last_lives, "world": self._last_world,
                    "coins": self._last_coins, "score": self._last_score,
                    "stuck": self._stuck, "game_over": False,
                    "game_state": 0, "died": False, "level_cleared": True,
                    "marathon_clears": self._marathon_clears,
                    "marathon_skipped_to": next_target,
                }
            # No cached save-state for `next_target` — we can't cleanly
            # skip the cutscene. Terminate the marathon episode explicitly
            # rather than soft-locking Mario in the level-end sequence.
            # Callers should have run `ensure_level_states()` upfront so
            # this is essentially "misconfigured env" territory.
            return self._obs(), float(reward), True, False, {
                "x": x, "max_x": self._max_x, "lives": lives,
                "world": world, "coins": coins, "score": score,
                "stuck": self._stuck, "game_over": is_game_over,
                "game_state": game_state, "died": died, "level_cleared": True,
                "marathon_clears": self._marathon_clears,
                "marathon_missing_state": next_target,
            }

        # Termination logic depends on mode
        if self._campaign_mode:
            # Play through: death and level clear are transitions within the
            # same episode. Only game over ends it — and a death on the
            # last life IS game over, so end right there instead of sitting
            # through the death animation + game-over screen (~50 steps).
            # (`death_now`, not `died`: on the fallback path `lives` has
            # already dropped, so 0 there means "now on the last life".)
            terminated = is_game_over or (death_now and lives == 0)
            # Reset per-attempt trackers when the new attempt actually
            # begins: respawn (life counter drops) or next level loaded
            # (world flips). Not at the instant-detection step — Mario is
            # still standing at the death spot / goal for a few steps.
            if lives_dropped or world_changed:
                self._max_x = x
                self._stuck = 0
        elif self.start_level == "marathon":
            # Marathon: any death ends the run. Level clears are handled
            # above (either fall through and continue, or terminate with a
            # bonus if it was the last usable level).
            terminated = died
        else:
            # Fixed / random / sequential: any death or level clear ends the episode.
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
            "game_state": game_state, "died": died, "level_cleared": level_cleared,
        }
        self._last_x = x
        self._last_lives = lives
        self._last_world = world
        self._last_score = score
        self._last_coins = coins
        return self._obs(), float(reward), terminated, truncated, info

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
        # Reset per-level trackers
        self._last_x = self.gw.level_progress
        self._max_x = self._last_x
        self._last_lives = self.gw.lives_left
        self._last_world = tuple(self.gw.world)
        self._last_score = self.gw.score
        self._last_coins = self.gw.coins
        self._stuck = 0
        self._clear_credited = False
        self._death_credited = False
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
