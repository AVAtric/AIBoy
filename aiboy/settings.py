"""The few things the app remembers about how the user wants it to behave.

A flat JSON object in `settings.json` next to the user's other data (see
paths.py; `AIBOY_SETTINGS_FILE` overrides the location). Missing file,
unreadable file and unknown keys all fall back to the defaults, so the app
never depends on it existing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from aiboy.paths import DATA_DIR

# Tests point this at a scratch file so they never change the user's settings.
SETTINGS_FILE = Path(os.environ.get("AIBOY_SETTINGS_FILE") or DATA_DIR / "settings.json")

DEFAULTS: dict = {
    # Change presets to the settings AIboy has measured to be clearly better
    # (see experience.Experience.improvements). Off = only report them.
    "auto_improve_presets": True,
}


def load(path: Path | None = None) -> dict:
    """Every setting, defaults filled in. `path` defaults to SETTINGS_FILE
    as it is *now*, so tests can point the module elsewhere."""
    data: dict = {}
    try:
        raw = json.loads(Path(path or SETTINGS_FILE).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    except (OSError, ValueError):
        pass
    return {**DEFAULTS, **data}


def get(key: str, path: Path | None = None):
    return load(path).get(key, DEFAULTS.get(key))


def put(key: str, value, path: Path | None = None) -> None:
    path = Path(path or SETTINGS_FILE)
    data = load(path)
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
