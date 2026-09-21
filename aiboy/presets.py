"""Training presets: built-in defaults + user-saved JSON.

A preset is a dict of training-tab field values (see PRESET_FIELDS).

Storage:
  - `builtin_presets.json`  — canonical source for built-in presets. Bundled
    with the code. If missing, the hardcoded `_HARDCODED_BUILTINS` fallback
    below is used (and the JSON file is regenerated).
  - `training_presets.json` — user-saved presets. Overrides built-ins on
    name collision.

Editing `builtin_presets.json` lets you tweak the shipped defaults without
touching Python code.
"""
from __future__ import annotations

import json

from aiboy.paths import BUNDLE_DIR, DATA_DIR, recommended_n_envs

BUILTINS_FILE = BUNDLE_DIR / "builtin_presets.json"     # shipped with the program
PRESETS_FILE = DATA_DIR / "training_presets.json"       # the user's own, next to the program

# Fields a preset can set. Keep in sync with gui/widgets.py's FIELDS (the
# parameter form) and the `train` flags in cli.py.
PRESET_FIELDS = (
    "game", "timesteps", "n_envs", "ent_coef", "learning_rate",
    "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
    "device", "obs_type", "start_level", "action_repeat", "frame_stack",
    "seed", "checkpoint_freq", "eval_freq", "n_eval_episodes",
    "time_budget", "stall_steps",
)

# Optional fields: presets that omit them (all shipped ones) get these,
# so `cfg.get(k, PRESET_DEFAULTS[k])` is the canonical read.
PRESET_DEFAULTS: dict[str, float] = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "time_budget": 250,     # timer units (of 400) an attempt may use; 0 = whole timer
    "stall_steps": 300,     # steps without a new furthest point before the attempt ends; 0 = off
}

# A complete configuration. `normalize()` fills any field a preset lacks
# from here, so hand-edited JSON with missing keys still loads and trains.
DEFAULT_CONFIG: dict = {
    "game": "mario",
    "obs_type": "tiles",
    "start_level": "default",
    "timesteps": 2_000_000,
    "n_envs": 10,
    "ent_coef": 0.01,
    "learning_rate": 2.5e-4,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 4,
    "action_repeat": 4,
    "frame_stack": 4,
    "seed": 0,
    "checkpoint_freq": 100_000,
    "eval_freq": 25_000,
    "n_eval_episodes": 3,
    "device": "cpu",
    **PRESET_DEFAULTS,
}

# All timings estimated on Apple Silicon (M1 Max) at ~3000-3500 fps with
# tile obs (n_envs=10, batch_size=256, torch.set_num_threads(1)). With 10
# envs x 512 steps a rollout is 5120 samples; batch 256 gives 20 minibatches
# x 4 epochs = 80 gradient steps per rollout and roughly halves the PPO
# update time compared with batch 128 (measured 0.49 s vs 0.83 s), during
# which all emulator processes sit idle. Pixel-obs runs are much slower
# (~40-70 fps).
#
# Modes (controlled by `start_level`):
#   - "default"      → CAMPAIGN. Mario plays through 1-1, respects lives,
#                       advances to next level on clear, restarts at 1-1
#                       when all lives are exhausted. Episodes span
#                       multiple levels / deaths.
#   - "random"       → each episode picks a random level from the twelve
#                       (1-1 … 4-3). Death or level clear ends the episode
#                       → next episode = new level. Best for generalisation
#                       across worlds.
#   - "sequential"   → cycles through all twelve levels. The cursor
#                       advances ONLY on a real clear; death retries the
#                       same level with fresh lives.
#   - "marathon"     → one life from 1-1 through all twelve levels in one
#                       episode. Level-end cutscene is skipped by
#                       force-loading the next level's state the instant
#                       Mario clears. ANY death ends the episode; the next
#                       one starts at 1-1 again.
#   - "1-1", "2-1"…  → FIXED level. Same level every episode; death or
#                       clear terminates and next episode replays the same
#                       state. Best for drilling one hard level.
def _preset(game: str, start_level: str, timesteps: int, *, checkpoint_freq: int,
            eval_freq: int, ent_coef: float = 0.01, n_envs: int = 10, n_steps: int = 512,
            batch_size: int = 256, n_eval_episodes: int = 3, obs_type: str = "tiles",
            **extra) -> dict:
    """A shipped preset: the common PPO settings plus what differs."""
    return {
        "game": game, "obs_type": obs_type, "start_level": start_level,
        "timesteps": timesteps, "n_envs": n_envs, "ent_coef": ent_coef,
        "learning_rate": 2.5e-4, "n_steps": n_steps, "batch_size": batch_size, "n_epochs": 4,
        "action_repeat": 4, "frame_stack": 4, "seed": 0,
        "checkpoint_freq": checkpoint_freq, "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes, "device": "cpu", **extra,
    }


def _mario(start_level: str, timesteps: int, **kw) -> dict:
    return _preset("mario", start_level, timesteps, **kw)


def _kirby(timesteps: int, **kw) -> dict:
    """A Kirby preset: no level modes, no timer, so the stall rule ends an
    attempt that stops getting further (see env.KirbyEnv)."""
    return _preset("kirby", "default", timesteps, stall_steps=400, ent_coef=0.02, **kw)


_HARDCODED_BUILTINS: dict[str, dict] = {
    # ---- Campaign (play through the game respecting lives) ----
    "Mario — Quick smoke test (30 s)": _mario(
        "default", 4_000, checkpoint_freq=4_000, eval_freq=4_000, n_envs=2, n_steps=128,
        batch_size=128, n_eval_episodes=1),
    "Mario — Campaign, recommended (~15 min)": _mario(
        "default", 2_000_000, checkpoint_freq=100_000, eval_freq=25_000),
    "Mario — Campaign, extended (~1 h)": _mario(
        "default", 8_000_000, checkpoint_freq=250_000, eval_freq=100_000),
    "Mario — Campaign, overnight (~8 h)": _mario(
        "default", 60_000_000, checkpoint_freq=500_000, eval_freq=250_000),
    # ---- Random-level training (generalises across all twelve levels).
    #      Slightly higher entropy (0.02) helps in every multi-level mode
    #      because the state distribution is wider than the campaign's. ----
    "Mario — Random levels (~1 h)": _mario(
        "random", 10_000_000, checkpoint_freq=250_000, eval_freq=100_000, ent_coef=0.02),
    "Mario — Random levels, overnight (~8 h)": _mario(
        "random", 60_000_000, checkpoint_freq=500_000, eval_freq=250_000, ent_coef=0.02),
    # ---- Sequential: cycle through all twelve levels. Death retries
    #      the same level; a clear advances to the next. ----
    "Mario — No return, cycle levels (~1 h)": _mario(
        "sequential", 10_000_000, checkpoint_freq=250_000, eval_freq=100_000, ent_coef=0.02),
    "Mario — No return, overnight (~8 h)": _mario(
        "sequential", 60_000_000, checkpoint_freq=500_000, eval_freq=250_000, ent_coef=0.02),
    # ---- Marathon: one life from 1-1 through all twelve levels; the
    #      level-end cutscene is skipped by loading the next level's state.
    #      Any death ends the episode and the next one starts at 1-1. ----
    "Mario — Marathon, all levels (~1 h)": _mario(
        "marathon", 10_000_000, checkpoint_freq=250_000, eval_freq=100_000, ent_coef=0.02),
    "Mario — Marathon, overnight (~8 h)": _mario(
        "marathon", 60_000_000, checkpoint_freq=500_000, eval_freq=250_000, ent_coef=0.02),
    # ---- Single-level drilling (example: 3-2, a tricky one). Copy this
    #      and change start_level to focus on any specific level. ----
    "Mario — Practice level 3-2 (~30 min)": _mario(
        "3-2", 4_000_000, checkpoint_freq=100_000, eval_freq=50_000, ent_coef=0.02),
    # ---- Pixel-based (slower but more info; use if tile obs plateaus) ----
    "Mario — Pixels CNN (~2 h for 500k)": _mario(
        "default", 500_000, checkpoint_freq=25_000, eval_freq=10_000, ent_coef=0.02,
        obs_type="pixels", n_envs=4, n_steps=256),
    # ---- Kirby's Dream Land: from the start of the game, every attempt
    #      ends on a death; reward for getting further, score and health. ----
    "Kirby — Quick smoke test (30 s)": _kirby(
        4_000, checkpoint_freq=4_000, eval_freq=4_000, n_envs=2, n_steps=128, batch_size=128,
        n_eval_episodes=1),
    "Kirby — Dream Land, recommended (~15 min)": _kirby(
        2_000_000, checkpoint_freq=100_000, eval_freq=25_000),
    "Kirby — Dream Land, extended (~1 h)": _kirby(
        8_000_000, checkpoint_freq=250_000, eval_freq=100_000),
    "Kirby — Dream Land, overnight (~8 h)": _kirby(
        60_000_000, checkpoint_freq=500_000, eval_freq=250_000),
}

RECOMMENDED_PRESET = "Mario — Campaign, recommended (~15 min)"
SMOKE_TEST_PRESET = "Mario — Quick smoke test (30 s)"
# The preset a game's list opens on (the first sorted built-in otherwise).
RECOMMENDED_BY_GAME = {"mario": RECOMMENDED_PRESET,
                       "kirby": "Kirby — Dream Land, recommended (~15 min)"}


def _load_builtin_from_file() -> dict[str, dict] | None:
    """Load built-in presets from `builtin_presets.json` if it exists and
    parses cleanly. Returns None otherwise (caller falls back to hardcoded).
    """
    if not BUILTINS_FILE.exists():
        return None
    try:
        data = json.loads(BUILTINS_FILE.read_text())
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _write_builtin_file(data: dict[str, dict]) -> None:
    """Write the built-ins with every field present, so the hand-editable
    file shows all knobs (including ones added later, such as time_budget)."""
    complete = {name: {**DEFAULT_CONFIG, **cfg} for name, cfg in data.items()}
    try:
        BUILTINS_FILE.write_text(json.dumps(complete, indent=2, sort_keys=False))
    except OSError:
        pass


def fit_to_machine(cfg: dict) -> dict:
    """A shipped preset assumes a ~10-core machine. Cap its parallel
    emulators at what this machine can actually run (see
    `paths.recommended_n_envs`); more processes than cores only add
    scheduling overhead and memory. User presets are taken literally."""
    n_envs = cfg.get("n_envs")
    if isinstance(n_envs, int) and n_envs > recommended_n_envs():
        return {**cfg, "n_envs": recommended_n_envs()}
    return cfg


def _init_builtins() -> dict[str, dict]:
    """Load bundled JSON if present; otherwise write hardcoded defaults
    to disk. In either case the returned dict is `hardcoded ⊕ file` — the
    hardcoded set is the baseline (so any newly-added built-in preset in
    the codebase automatically appears), and the JSON file overrides on
    name collision (so users can hand-edit built-ins). Missing JSON is
    written from the current hardcoded defaults. Every built-in is fitted
    to this machine's core count.
    """
    from_file = _load_builtin_from_file()
    merged = dict(_HARDCODED_BUILTINS)
    if from_file is not None:
        merged.update(from_file)
    else:
        _write_builtin_file(_HARDCODED_BUILTINS)
    return {name: fit_to_machine(cfg) for name, cfg in merged.items()}


BUILTIN_PRESETS: dict[str, dict] = _init_builtins()


def load_all() -> dict[str, dict]:
    """Return builtin + user presets, each normalized to a complete config.
    User overrides builtin on collision. Built-ins are re-read from JSON on
    every call so hand-edits are picked up.
    """
    merged = dict(_init_builtins())
    for name, cfg in load_user().items():
        merged[name] = cfg
    return {name: normalize(cfg) for name, cfg in merged.items()}


def load_user() -> dict[str, dict]:
    if not PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(PRESETS_FILE.read_text())
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError):
        return {}


def save_user(user: dict[str, dict]) -> None:
    PRESETS_FILE.write_text(json.dumps(user, indent=2, sort_keys=True))


def upsert(name: str, config: dict) -> None:
    """Save or overwrite a user preset.

    Saving under a built-in name stores a user *override* that shadows the
    shipped values (see `is_overridden` / `reset`); the built-in itself is
    never modified.
    """
    user = load_user()
    user[name] = normalize(config)
    save_user(user)


def apply_improvement(name: str, overrides: dict) -> dict:
    """Change a preset's hyperparameters to values AIboy found better (see
    experience.Experience.improvements). A built-in gets a user override,
    so 'Reset to default' still brings the shipped values back. Returns the
    new configuration."""
    current = load_all().get(name)
    if current is None:
        raise ValueError(f"No preset named '{name}'.")
    cfg = {**current, **overrides}
    upsert(name, cfg)
    return normalize(cfg)


def is_overridden(name: str) -> bool:
    """A built-in preset that the user has edited (override present)."""
    return name in BUILTIN_PRESETS and name in load_user()


def reset(name: str) -> None:
    """Drop the user override of a built-in preset, restoring shipped values."""
    if name not in BUILTIN_PRESETS:
        raise ValueError(f"'{name}' is not a built-in preset.")
    user = load_user()
    if name in user:
        del user[name]
        save_user(user)


def kind(name: str, user: dict | None = None) -> str:
    """'built-in' | 'modified' (built-in with override) | 'user'.

    Pass `user=load_user()` when classifying many names at once so the
    file is read only once."""
    if user is None:
        user = load_user()
    if name in BUILTIN_PRESETS:
        return "modified" if name in user else "built-in"
    return "user"


def rename(old: str, new: str) -> None:
    """Rename a user preset. Built-ins cannot be renamed; names must be unique."""
    if old in BUILTIN_PRESETS:
        raise ValueError(f"'{old}' is a built-in preset and cannot be renamed.")
    if new in BUILTIN_PRESETS:
        raise ValueError(f"'{new}' is a built-in preset; pick a different name.")
    user = load_user()
    if old not in user:
        raise ValueError(f"No user preset named '{old}'.")
    if new != old and new in user:
        raise ValueError(f"A preset named '{new}' already exists.")
    user[new] = user.pop(old)
    save_user(user)


def delete(name: str) -> None:
    """Delete a user preset. For a built-in this removes the override only."""
    if name in BUILTIN_PRESETS and name not in load_user():
        raise ValueError(f"Cannot delete built-in preset '{name}'.")
    user = load_user()
    if name in user:
        del user[name]
        save_user(user)


def is_builtin(name: str) -> bool:
    return name in BUILTIN_PRESETS


def is_user(name: str) -> bool:
    """True for a user preset or a user override of a built-in."""
    return name in load_user()


def recommended_for(game: str) -> str:
    """The recommended preset of `game` (Mario's for a game without one)."""
    return RECOMMENDED_BY_GAME.get(game, RECOMMENDED_PRESET)


def sorted_names(all_presets: dict[str, dict], game: str | None = None) -> list[str]:
    """Preset names in display order: the game's recommended one first, then
    the other built-ins, then user presets, each group alphabetical. `game`
    restricts the list to presets for that game."""
    names = [n for n, cfg in all_presets.items()
             if game is None or cfg.get("game", "mario") == game]
    top = set(RECOMMENDED_BY_GAME.values())
    return sorted(names, key=lambda n: (n not in top, not is_builtin(n), n.lower()))


def normalize(cfg: dict) -> dict:
    """Preset fields only, every field present (missing ones from DEFAULT_CONFIG)."""
    out = {k: v for k, v in cfg.items() if k in PRESET_FIELDS}
    for k, v in DEFAULT_CONFIG.items():
        out.setdefault(k, v)
    return out
