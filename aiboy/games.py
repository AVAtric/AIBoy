"""Game registry, ROM discovery and Super Mario Land level facts.

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

# Version of the learning task itself: reward shaping, observation layout,
# level logic. Bump it whenever one of those changes (MarioEnv, the HUD
# cells, the level modes): scores recorded under an older version stay in
# the experience file for reading but are never reused or compared, because
# they would measure a different game.
#   mario-1  first version
#   mario-2  marathon training episodes start at 1-1 (were: a random level)
ENV_VERSION = "mario-2"

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
