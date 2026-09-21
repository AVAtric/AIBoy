"""Game registry, ROM discovery, and the level facts of Super Mario Land
and the memory facts of Kirby's Dream Land the environments read.

Deliberately free of PyBoy / Stable-Baselines3 / torch imports so the GUI can
start instantly; the emulator only gets imported when something actually
runs (see env.py and player.py). Anything here that needs PyBoy does it in a
subprocess.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiboy.paths import app_command

ROM_DIR = Path("ROMs")

# Version of each game's learning task: reward shaping, observation layout,
# level logic. Bump it whenever one of those changes (the env class, the HUD
# cells, the level modes): scores recorded under an older version stay in
# the experience file for reading but are never reused or compared, because
# they would measure a different game.
#   mario-1  first version
#   mario-2  marathon training episodes start at 1-1 (were: a random level)
#   mario-3  tile 319 reads as empty (decoration PyBoy files under the blocks)
#   mario-4  all twelve levels (2-3 and 4-3, the submarine and the plane,
#            are in the marathon), UP actions for them, and the hazards
#            PyBoy's lists miss (SML_HAZARD_TILES) read 1.0
#   kirby-1  first version: scroll progress, score, health, lives
ENV_VERSIONS = {"mario": "mario-4", "kirby": "kirby-1"}
ENV_VERSION = ENV_VERSIONS["mario"]          # the first game's; prefer env_version(game)


def env_version(game: str) -> str:
    """The task version of `game` ("generic-1" for a ROM without an AIboy env)."""
    return ENV_VERSIONS.get(str(game), "generic-1")

# Super Mario Land has 4 worlds x 3 levels = 12 levels. PyBoy's
# `set_world_level(w, l)` docstring is wrong: it says the arguments are
# 0-indexed, but the game's level byte (RAM 0xFFB4, `hWorldAndLevel` in the
# disassembly) is 1-indexed: 0x11 = 1-1, 0x21 = 2-1, 0x43 = 4-3. Writing an
# invalid value silently loads a fallback level while `game_wrapper.world`
# still reports the patched tuple, so w, l are always passed 1-indexed.
#
# The third level of worlds 2 and 4 is a vehicle level (2-3 the submarine
# "Marine Pop", 4-3 the plane "Sky Pop", with Tatanga at the end). They boot
# and play like the others (verified 2026-09-21 with level_state_worker: the
# level byte reads 0x23 / 0x43, the timer runs, the vehicle moves); an
# earlier note that they could not be started was wrong. The game runs them
# in its auto-scroll state (SML_PLAY_STATES) and the vehicle needs UP to
# move up, which is why MarioEnv has UP actions.
SML_ALL_LEVELS = tuple((w, l) for w in range(1, 5) for l in range(1, 4))
SML_VEHICLE_LEVELS = frozenset({(2, 3), (4, 3)})

# SML's game-state byte (`hGameState` at 0xFFB3 in the disassembly, mapped
# there and by logging RAM transitions across clears / deaths / game over):
#   0x00  playing (walking levels)
#   0x0D  playing, auto-scroll (the vehicle levels 2-3 and 4-3)
#   0x07 -> 0x05 -> 0x06  level-clear sequence: goal touched (also the boss
#                       switch of an x-3 level: "Mario wins"), score
#                       countdown, winning (~70 env-steps at frame_skip 4
#                       before PyBoy's `world` tuple finally flips); after
#                       4-3 it goes on to 0x27 (Tatanga dying) and the ending
#   0x03 -> 0x04 -> 0x01  pre-dying, dying animation, waiting for respawn
#   0x02 / 0x08  respawn / next-level loading (transient)
#   0x09 .. 0x0C  pipe warps (not deaths)
#   0x12  the bonus game (reaching the top exit of a goal gate)
#   0x3A  game-over screen (PyBoy's game_over() checks 0xC0A4 == 0x39)
# Reading it lets the env credit a clear or a death the step it happens
# instead of ~20-70 steps later, a large training-throughput win since
# every step in a cutscene / death animation is a wasted sample.
ADDR_GAME_STATE = 0xFFB3
SML_PLAY_STATES = frozenset({0x00, 0x0D})
SML_CLEAR_STATES = frozenset({0x05, 0x06, 0x07})
SML_DEATH_STATES = frozenset({0x01, 0x04})

# What is solid in Super Mario Land, checked against the game rather than
# PyBoy's lists (tools/mario_tile_survey.py logs, over every level, the
# background tile under Mario whenever he stands still and the tiles his
# body overlaps while alive):
#   - everything Mario ever stands on is on PyBoy's block / pipe lists
#     (352-362, 142-143, 368-371, 383) or is a lift sprite (230, 238, 239),
#     so the tile observation already shows all the ground there is;
#   - the many tiles PyBoy does not list (palm trees 305-307 and 311, hills
#     310 / 350, clouds 320-327, the sky 300, ...) are decoration Mario
#     walks through, correctly shown as empty;
#   - tile 319 is on PyBoy's `neutral_blocks` list but is decoration too
#     (overlapped in 2-1, 3-1 and 4-2, never stood on). It is shown as
#     empty so the agent does not see a wall that is not there.
# Changing this changes what a trained model sees: bump ENV_VERSIONS["mario"].
SML_BACKGROUND_TILES = frozenset({319})

# Sprites PyBoy's enemy lists miss (the same survey, `--sprites`: every tile
# a sprite on screen used, per level, with where it was seen). Without
# these the observation showed the thing as empty, or, for a two-tile
# sprite with one tile listed, as half an enemy:
#   200            the bomb a stomped Nokobon leaves behind (1-2; 201 is
#                  listed, 200 was not, so the bomb was half visible)
#   168, 169, 184, 185   an enemy's other animation frames (1-1 at x 1500-
#                  2200 next to the moth 160-163; 2-3)
#   170, 171, 186, 187, 173   the enemy at the start of 3-1 and what it
#                  throws (172 is listed)
#   98             a sinking pair of enemies in the submarine level, also
#                  seen in 4-3
#   250, 251       the other frames of the bullet (249 is listed), 4-3
#   216            the other frame of the enemy at the start of 2-2 (198,
#                  199, 214, 215 are listed as big_sphinx; 216 was not)
# Harmless sprites stay empty: the score numbers that float up after a kill
# or a coin (88-93, 145), a block bouncing after a hit (246-248, 254), the
# submarine's torpedo (122) and the plane's missile (110).
SML_HAZARD_TILES = frozenset({98, 168, 169, 170, 171, 173, 184, 185, 186, 187, 200, 216, 250, 251})
SML_HARMLESS_SPRITES = frozenset({88, 89, 90, 91, 92, 93, 145, 246, 247, 248, 254, 122, 110})

# Mario's power-up, mapped by writing values and watching the sprite:
#   0xFF99  power-up state machine: 0 small, 1 growing, 2 super,
#           3 / 4 hit and shrinking back to small
#   0xFFB5  non-zero while Mario has the superball (pressing B throws one)
# The level timer starts at 400 units; one unit is 38.8 frames (0.65 s),
# measured in the emulator, so a full timer is ~258 s or ~3860 env steps
# at action repeat 4.
TIMER_START = 400
DEFAULT_TIME_BUDGET = 250      # timer units an attempt may use before it is truncated
DEFAULT_STALL_STEPS = 0        # steps without new progress before truncation; 0 = off
ADDR_POWERUP_STATE = 0xFF99
ADDR_SUPERBALL = 0xFFB5
POWER_SMALL, POWER_SUPER, POWER_SUPERBALL = 0, 1, 2
POWER_NAMES = ("small", "super", "superball")


def power_state(pyboy) -> int:
    """POWER_SMALL / POWER_SUPER / POWER_SUPERBALL from Super Mario Land's RAM.
    Transitions count as their destination: growing -> super, hit -> small."""
    state = pyboy.get_memory_value(ADDR_POWERUP_STATE)
    if state in (1, 2):
        return POWER_SUPERBALL if pyboy.get_memory_value(ADDR_SUPERBALL) else POWER_SUPER
    return POWER_SMALL

# Kirby's Dream Land (PyBoy's wrapper reads score, health and lives; the
# rest was mapped by watching RAM while playing):
#   0xD05C  Kirby's x on the screen (8..~150; the camera keeps him near 76)
#   0xD05D  Kirby's y on the screen
ADDR_KIRBY_X = 0xD05C
KIRBY_MAX_HEALTH = 6
KIRBY_ATTEMPT_STEPS = 6000     # hard cap per attempt: the game has no timer to end one

# Observation types. "tiles" is the 16x20 tile grid with seven HUD scalars
# (lives, coins, timer, x, world, level, power-up) in the top-left cells;
# "pixels" is the raw screen (CNN).
OBS_TYPES = ("tiles", "pixels")

# Where per-level save-state files live. Gitignored via models/.
LEVEL_STATES_DIR = Path("models") / "mario" / "_level_states"

# Level modes that are not a fixed "W-L" level (see MarioEnv docstring).
LEVEL_MODES = ("default", "random", "sequential", "marathon")
MULTI_LEVEL_MODES = ("random", "sequential", "marathon")


def level_choices() -> list[str]:
    """Every valid `--start-level` value: the four modes, then each level."""
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
    state_file.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    try:
        subprocess.run(
            app_command("level-state", str(rom_path.resolve()), str(world), str(level),
                        str(state_file.resolve())),
            timeout=timeout_sec, capture_output=True, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        state_file.unlink(missing_ok=True)
        return False
    return state_file.exists() and state_file.stat().st_size > 1000


def level_state_worker(rom_path: str, world: int, level: int, out_file: str) -> None:
    """Body of the `level-state` sub-command (runs in its own process)."""
    from pyboy import PyBoy
    # PyBoy's set_world_level encoding is 1-indexed (memory byte 0x11 = 1-1),
    # despite what its docstring claims. Pass w, l as-is (1-indexed).
    p = PyBoy(rom_path, window_type="null", game_wrapper=True, disable_renderer=True)
    p.game_wrapper().start_game(world_level=(world, level))
    # Let the level fully load before saving state so the tile map is
    # settled; avoids a state that boots into the "level intro" screen.
    for _ in range(60):
        p.tick()
    with open(out_file, "wb") as f:
        p.save_state(f)
    p.stop(save=False)


def ensure_level_states(rom_path: Path, targets=None) -> list[tuple[int, int]]:
    """Bootstrap save-state files for the given levels. If targets is None,
    bootstraps every level. Returns list of levels that succeeded.
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
    if (w, l) not in SML_ALL_LEVELS:
        raise ValueError(f"level {w}-{l} not in Super Mario Land (worlds 1-4, levels 1-3)")
    return (w, l)


@dataclass(frozen=True)
class GameSpec:
    rom_file: str
    cartridge_title: str
    # A supported game has an AIboy environment (env.ENV_CLASSES): shaped
    # reward, pixel or tile observations, presets. Any other ROM can be
    # played by hand, not trained.
    supported: bool = False
    name: str = ""                  # how the game is called in the window
    levels: bool = False            # has Super Mario Land's level modes (start_level)
    # Live-panel labels that differ from Mario's wording, e.g. ("power", "health").
    stat_labels: tuple[tuple[str, str], ...] = ()


def check_start_level(game: str, start_level) -> str | None:
    """Why `start_level` cannot be used with `game`, or None when it can: only
    Super Mario Land has level modes; every other game starts from its
    beginning ("default")."""
    if start_level in (None, "default"):
        return None
    spec = GAMES.get(game)
    if spec is not None and spec.levels:
        return None
    name = spec.name if spec is not None and spec.name else game
    return (f"{name} has no level modes: Start level must be 'default' "
            f"(got {start_level!r}). Level modes are for Super Mario Land.")


def display_name(game: str) -> str:
    """'Super Mario Land' for 'mario'; the ROM name for anything unknown."""
    spec = GAMES.get(game)
    return spec.name if spec is not None and spec.name else game


GAMES = {
    "mario": GameSpec("mario.gb", "SUPER MARIOLAN", supported=True, name="Super Mario Land",
                      levels=True),
    "kirby": GameSpec("kirby.gb", "KIRBY DREAM LA", supported=True, name="Kirby's Dream Land",
                      stat_labels=(("power", "health"),)),
    "wario": GameSpec("wario.gb", "SUPERMARIOLAND", name="Wario Land"),   # Super Mario Land 3
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
        "ok"            fully supported: train it and play it (its own env,
                        presets; level modes for Super Mario Land)
        "playable"      play it yourself; no training (no AIboy environment
                        for it, or an unknown cartridge under a known name)
        "unsupported"   cannot boot
        "checking"      probe still running

        AIboy is an emulator for every ROM in the folder and a trainer for
        the supported ones: anything that boots can be played by a person.
        """
        if self.error:
            return "unsupported", f"cannot boot ROM: {self.error}"
        spec = self.spec
        if spec is not None and spec.supported:
            if self.probed and self.title != spec.cartridge_title:
                return "playable", (f"expected cartridge '{spec.cartridge_title}' but the ROM "
                                    f"reports '{self.title}' — play it yourself; no training")
            return "ok", f"{self.title or spec.cartridge_title} — supported: train it, play it"
        if not self.probed:
            return "checking", "checking whether PyBoy can run this ROM…"
        return "playable", (f"{self.title} — play it yourself; no training for this game "
                            f"(AIboy has environments for Super Mario Land and Kirby's "
                            f"Dream Land)")

    @property
    def runnable(self) -> bool:
        """Can be trained and played by the agent here."""
        return self.status()[0] == "ok"

    @property
    def playable(self) -> bool:
        """Can be played by a person: everything that boots (or is still
        being checked)."""
        return self.error is None


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
    try:
        out = subprocess.run(app_command("probe-rom", str(Path(rom_path).resolve())),
                             timeout=timeout_sec, capture_output=True, text=True)
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


def probe_rom_worker(rom_path: str) -> None:
    """Body of the `probe-rom` sub-command: prints one JSON line."""
    import json
    from pyboy import PyBoy
    p = PyBoy(rom_path, window_type="null", game_wrapper=True, disable_renderer=True)
    print(json.dumps({"title": p.cartridge_title(), "wrapper": p.game_wrapper() is not None}),
          flush=True)
    p.stop(save=False)
