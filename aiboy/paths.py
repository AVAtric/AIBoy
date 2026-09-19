"""Where the program lives and where it keeps its data.

Two layouts are supported:

  source tree          the project folder holds aiboy/ (the code) next to
                       ROMs/, models/, assets/, builtin_presets.json,
                       training_presets.json and experience.jsonl
  frozen release       (built by build_release.py with PyInstaller) the code and
                       the read-only files (assets/, builtin_presets.json,
                       system_profile.json) live inside the bundle; the user's
                       files (ROMs/, models/, training_presets.json,
                       experience.jsonl, the error log) live next to the app so
                       they survive an update.

Every entry point makes DATA_DIR the working directory, so the relative
`ROMs/` and `models/` paths used throughout keep working in both layouts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))

if FROZEN:
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    _exe = Path(sys.executable).resolve()
    if _exe.parent.name == "MacOS" and len(_exe.parents) > 3 and _exe.parents[2].suffix == ".app":
        DATA_DIR = _exe.parents[3]          # the folder that holds AIboy.app
    else:
        DATA_DIR = _exe.parent              # onedir build on Windows / Linux
else:
    BUNDLE_DIR = Path(__file__).resolve().parents[1]     # the project folder (aiboy/ is inside it)
    DATA_DIR = BUNDLE_DIR

PROFILE_FILE = BUNDLE_DIR / "system_profile.json"
ASSET_DIR = BUNDLE_DIR / "assets"


def local_or_shipped(name: str, assets: Path = ASSET_DIR) -> Path:
    """Path of an asset that exists in two versions: the one the repository
    ships as `assets/<name>` (the AIboy artwork: the photo made by
    tools/make_interface.py, the boot video by tools/make_intro.py) and an
    original, `assets/orig_<name>`, that is never distributed (.gitignore,
    build_release.py) but is used instead when someone has put one there.
    AIBOY_SHIPPED_ASSETS=1 ignores the originals (README screenshots, tests)."""
    orig = assets / f"orig_{name}"
    if os.environ.get("AIBOY_SHIPPED_ASSETS") or not orig.exists():
        return assets / name
    return orig

# Everything AIboy has learned from its trials and runs (see experience.py).
# Tests point this at a scratch file so they never touch the real memory.
EXPERIENCE_FILE = Path(os.environ.get("AIBOY_EXPERIENCE_FILE") or DATA_DIR / "experience.jsonl")


def app_command(*args: str) -> list[str]:
    """Command line that runs this program with `args` (a sub-command and its
    flags) in a child process: the frozen executable itself, or the Python
    interpreter with main.py. `-u` keeps the trainer's log lines flowing to
    the GUI as they are printed."""
    if FROZEN:
        return [sys.executable, *args]
    return [sys.executable, "-u", "main.py", *args]


def cpu_count() -> int:
    return os.cpu_count() or 4


def system_profile() -> dict:
    """Machine facts recorded at build time (see build_release.py): cores,
    memory, chip and the number of emulators worth running in parallel.
    Empty in the source tree or when the file is missing / unreadable."""
    try:
        data = json.loads(PROFILE_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def recommended_n_envs() -> int:
    """Emulator processes worth running on this machine: the build-time
    profile if there is one, else one per core, at most 12 (beyond that the
    PPO update, not the rollout, dominates). Never more than the cores of the
    machine actually running, in case a release was built elsewhere."""
    cores = cpu_count()
    profile = system_profile()
    n = profile.get("n_envs")
    if isinstance(n, int) and n > 0:
        return max(1, min(n, cores))
    return max(1, min(12, cores))
