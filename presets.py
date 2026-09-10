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
    "n_steps", "batch_size", "device", "obs_type",
)

BUILTIN_PRESETS: dict[str, dict] = {
    # Roughly-timed on Apple Silicon (M1 Max) at ~2000 fps with tile obs.
    "Mario — Balanced tiles (recommended, ~15 min)": {
        "game": "mario",
        "obs_type": "tiles",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 64,
        "device": "cpu",
    },
    "Mario — Quick smoke test (30 s)": {
        "game": "mario",
        "obs_type": "tiles",
        "timesteps": 4_000,
        "n_envs": 2,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 128,
        "batch_size": 64,
        "device": "cpu",
    },
    "Mario — Extended (~1 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "timesteps": 8_000_000,
        "n_envs": 8,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 64,
        "device": "cpu",
    },
    "Mario — Overnight (~8 h)": {
        "game": "mario",
        "obs_type": "tiles",
        "timesteps": 60_000_000,
        "n_envs": 8,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 64,
        "device": "cpu",
    },
    "Mario — Pixels (slow, CNN, ~2 h for 500k)": {
        "game": "mario",
        "obs_type": "pixels",
        "timesteps": 500_000,
        "n_envs": 4,
        "ent_coef": 0.02,
        "learning_rate": 2.5e-4,
        "n_steps": 256,
        "batch_size": 256,
        "device": "cpu",
    },
    "Kirby — Default pixels": {
        "game": "kirby",
        "obs_type": "pixels",
        "timesteps": 500_000,
        "n_envs": 4,
        "ent_coef": 0.01,
        "learning_rate": 2.5e-4,
        "n_steps": 256,
        "batch_size": 64,
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
