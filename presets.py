"""Training presets: built-in defaults + user-saved JSON.

A preset is a dict of training-tab field values. User presets live in
`training_presets.json` next to this file; they override built-ins on
name collision. Built-ins can be viewed and used but not deleted.
"""
from __future__ import annotations

import json
from pathlib import Path

PRESETS_FILE = Path(__file__).resolve().parent / "training_presets.json"

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
#   - "1-1", "2-1"…  → FIXED level. Same level every episode; death or
#                       clear terminates and next episode replays the same
#                       state. Best for drilling one hard level.
BUILTIN_PRESETS: dict[str, dict] = {
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


def load_all() -> dict[str, dict]:
    """Return builtin + user presets. User overrides builtin on collision."""
    merged = dict(BUILTIN_PRESETS)
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
