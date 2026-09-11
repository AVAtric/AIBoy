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
from pathlib import Path

_HERE = Path(__file__).resolve().parent
BUILTINS_FILE = _HERE / "builtin_presets.json"
PRESETS_FILE = _HERE / "training_presets.json"

# Fields a preset can set. Keep in sync with gui.py's training-tab StringVars.
PRESET_FIELDS = (
    "game", "timesteps", "n_envs", "ent_coef", "learning_rate",
    "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
    "device", "obs_type", "start_level", "action_repeat", "frame_stack",
    "seed", "checkpoint_freq", "eval_freq", "n_eval_episodes",
)

# Optional fields: presets that omit them (all shipped ones) get these,
# so `cfg.get(k, PRESET_DEFAULTS[k])` is the canonical read.
PRESET_DEFAULTS: dict[str, float] = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
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
#   - "random"       → each episode picks a random level from the 10
#                       usable ones (1-1 … 4-2, skipping 2-3 and 4-3 which
#                       PyBoy's SML wrapper cannot boot). Death or level
#                       clear ends the episode → next episode = new level.
#                       Best for generalisation across worlds.
#   - "sequential"   → cycles through all 10 usable levels. The cursor
#                       advances ONLY on a real clear; death retries the
#                       same level with fresh lives.
#   - "marathon"     → single-attempt speedrun through every usable level
#                       in one episode. Level-end cutscene is skipped by
#                       force-loading the next level's state the instant
#                       Mario clears. ANY death ends the episode.
#   - "1-1", "2-1"…  → FIXED level. Same level every episode; death or
#                       clear terminates and next episode replays the same
#                       state. Best for drilling one hard level.
_HARDCODED_BUILTINS: dict[str, dict] = {
    # ---- Campaign (play through the game respecting lives) ----
    "Mario — Quick smoke test (30 s)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "default",
        "timesteps": 4_000,
        "n_envs": 2,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 128,
        "batch_size": 128,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 4_000,
        "eval_freq": 4_000,
        "n_eval_episodes": 1,
        "device": "cpu",
    },
    "Mario — Campaign, recommended (~15 min)": {
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
    },
    "Mario — Campaign, extended (~1 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "default",
        "timesteps": 8_000_000,
        "n_envs": 10,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 250_000,
        "eval_freq": 100_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    "Mario — Campaign, overnight (~8 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "default",
        "timesteps": 60_000_000,
        "n_envs": 10,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 500_000,
        "eval_freq": 250_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    # ---- Random-level training (generalises across 10 usable levels) ----
    "Mario — Random levels (~1 h)": {
        # Each episode's level is picked fresh from the 10 usable levels.
        # Slightly higher entropy (0.02) helps because the state
        # distribution is wider than campaign-mode.
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "random",
        "timesteps": 10_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 250_000,
        "eval_freq": 100_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    "Mario — Random levels, overnight (~8 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "random",
        "timesteps": 60_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 500_000,
        "eval_freq": 250_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    # ---- Sequential: cycle through all 10 usable levels. Death retries
    #      the same level; a clear advances to the next. Best for training
    #      an agent that has to reliably beat every level, not just the
    #      easy ones the current best-model happens to be good at. ----
    "Mario — No return, cycle levels (~1 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "sequential",
        "timesteps": 10_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 250_000,
        "eval_freq": 100_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    "Mario — No return, overnight (~8 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "sequential",
        "timesteps": 60_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 500_000,
        "eval_freq": 250_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    # ---- Marathon: one episode = single-attempt speedrun through ALL 10
    #      usable levels. On level clear the in-game cutscene is skipped by
    #      force-loading the next level's state (instant). ANY death ends
    #      the episode. Big bonus if Mario clears the final level. ----
    "Mario — Marathon, all levels (~1 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "marathon",
        "timesteps": 10_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 250_000,
        "eval_freq": 100_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    "Mario — Marathon, overnight (~8 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "marathon",
        "timesteps": 60_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 500_000,
        "eval_freq": 250_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    # ---- Single-level drilling (example: 3-2, a tricky one). Copy this
    #      and change start_level to focus on any specific level. ----
    "Mario — Practice level 3-2 (~30 min)": {
        "game": "mario",
        "obs_type": "tiles",
        "start_level": "3-2",
        "timesteps": 4_000_000,
        "n_envs": 10,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 100_000,
        "eval_freq": 50_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
    # ---- Pixel-based (slower but more info; use if tile obs plateaus) ----
    "Mario — Pixels CNN (~2 h for 500k)": {
        "game": "mario",
        "obs_type": "pixels",
        "start_level": "default",
        "timesteps": 500_000,
        "n_envs": 4,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 4,
        "action_repeat": 4,
        "frame_stack": 4,
        "seed": 0,
        "checkpoint_freq": 25_000,
        "eval_freq": 10_000,
        "n_eval_episodes": 3,
        "device": "cpu",
    },
}

RECOMMENDED_PRESET = "Mario — Campaign, recommended (~15 min)"
SMOKE_TEST_PRESET = "Mario — Quick smoke test (30 s)"


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
    try:
        BUILTINS_FILE.write_text(json.dumps(data, indent=2, sort_keys=False))
    except OSError:
        pass


def _init_builtins() -> dict[str, dict]:
    """Load bundled JSON if present; otherwise write hardcoded defaults
    to disk. In either case the returned dict is `hardcoded ⊕ file` — the
    hardcoded set is the baseline (so any newly-added built-in preset in
    the codebase automatically appears), and the JSON file overrides on
    name collision (so users can hand-edit built-ins). Missing JSON is
    written from the current hardcoded defaults.
    """
    from_file = _load_builtin_from_file()
    merged = dict(_HARDCODED_BUILTINS)
    if from_file is not None:
        merged.update(from_file)
    else:
        _write_builtin_file(_HARDCODED_BUILTINS)
    return merged


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


def sorted_names(all_presets: dict[str, dict], game: str | None = None) -> list[str]:
    """Preset names in display order: the recommended one first, then the
    other built-ins, then user presets, each group alphabetical. `game`
    restricts the list to presets for that game."""
    names = [n for n, cfg in all_presets.items()
             if game is None or cfg.get("game", "mario") == game]
    return sorted(names, key=lambda n: (n != RECOMMENDED_PRESET, not is_builtin(n), n.lower()))


def normalize(cfg: dict) -> dict:
    """Preset fields only, every field present (missing ones from DEFAULT_CONFIG)."""
    out = {k: v for k, v in cfg.items() if k in PRESET_FIELDS}
    for k, v in DEFAULT_CONFIG.items():
        out.setdefault(k, v)
    return out
