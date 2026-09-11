"""Training presets: built-in defaults + user-saved JSON.

A preset is a dict of training-tab field values.

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
    "n_steps", "batch_size", "device", "obs_type", "start_level",
    "action_repeat", "frame_stack", "n_epochs",
    "seed", "checkpoint_freq", "eval_freq", "n_eval_episodes",
)

# All timings estimated on Apple Silicon (M1 Max) at ~2500-3000 fps with
# tile obs (default n_envs=10, batch_size=128, torch.set_num_threads(1)).
# Pixel-obs runs are much slower (~40-70 fps).
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
        "batch_size": 128,
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
    # ---- Other games ----
    "Kirby — Default pixels (~1 h for 500k)": {
        "game": "kirby",
        "obs_type": "pixels",
        "start_level": "default",
        "timesteps": 500_000,
        "n_envs": 4,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 256,
        "batch_size": 64,
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
    """Return builtin + user presets. User overrides builtin on collision.
    Built-ins are re-read from JSON on every call so hand-edits are picked up.
    """
    merged = dict(_init_builtins())
    for name, cfg in load_user().items():
        merged[name] = cfg
    return merged


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
    """Save or overwrite a user preset. Reject overwriting a builtin."""
    if name in BUILTIN_PRESETS:
        raise ValueError(f"'{name}' is a built-in preset; pick a different name.")
    user = load_user()
    user[name] = {k: v for k, v in config.items() if k in PRESET_FIELDS}
    save_user(user)


def delete(name: str) -> None:
    if name in BUILTIN_PRESETS:
        raise ValueError(f"Cannot delete built-in preset '{name}'.")
    user = load_user()
    if name in user:
        del user[name]
        save_user(user)


def is_builtin(name: str) -> bool:
    return name in BUILTIN_PRESETS
