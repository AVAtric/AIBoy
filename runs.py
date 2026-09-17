"""Training-run artefacts: paths, model discovery and the trainer command line.

Everything here is Tk-free and shared by the CLI (`main.py`), the GUI and
the tuner so all three agree on where a run lives and how it is launched.

Layout (relative to the project directory, which every entry point makes
the working directory):

    models/<game>/<run>/checkpoints/ppo_<steps>_steps.zip   periodic snapshots
    models/<game>/<run>/checkpoints/final.zip               saved on exit
    models/<game>/<run>/logs/best_model.zip                 best eval reward
    models/<game>/<run>/logs/evaluations.npz                eval history
    models/<game>/<run>/tensorboard/                        TensorBoard events

Run names starting with "_" are internal caches (`_level_states`, `_tune`)
and are hidden from every listing.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from paths import app_command, cpu_count, recommended_n_envs  # noqa: F401  (re-exported)
from presets import PRESET_FIELDS, normalize

MODELS_ROOT = Path("models")
INTERNAL_PREFIX = "_"


def is_run_name(name: str) -> bool:
    """False for internal cache directories such as `_level_states`."""
    return bool(name) and not name.startswith(INTERNAL_PREFIX)


def run_paths(game: str, run_name: str, root: Path = MODELS_ROOT) -> dict[str, Path]:
    base = root / game / run_name
    return {
        "base": base,
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
        "tensorboard": base / "tensorboard",
    }


RUN_MANIFEST = "run.json"


def write_run_config(game: str, run_name: str, cfg: dict, root: Path = MODELS_ROOT) -> None:
    """Record the training settings of a run so playback can reproduce the
    observation setup (obs_type, action_repeat, frame_stack, start_level)."""
    base = run_paths(game, run_name, root)["base"]
    base.mkdir(parents=True, exist_ok=True)
    data = {k: cfg[k] for k in PRESET_FIELDS if k in cfg}
    data["game"] = game
    (base / RUN_MANIFEST).write_text(json.dumps(data, indent=2, sort_keys=True))


def read_run_config(game: str, run_name: str, root: Path = MODELS_ROOT) -> dict | None:
    path = run_paths(game, run_name, root)["base"] / RUN_MANIFEST
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def run_of_model(path: Path) -> tuple[str, str] | None:
    """(game, run_name) for a model file under models/<game>/<run>/{logs,checkpoints}/;
    None for a file that does not sit in that layout."""
    path = Path(path)
    if path.parent.name not in ("logs", "checkpoints"):
        return None
    run_dir = path.parent.parent
    if not is_run_name(run_dir.name) or not run_dir.parent.name:
        return None
    return run_dir.parent.name, run_dir.name


def list_runs(game: str, root: Path = MODELS_ROOT) -> list[str]:
    game_dir = root / game
    if not game_dir.exists():
        return []
    return sorted(d.name for d in game_dir.iterdir() if d.is_dir() and is_run_name(d.name))


def _snapshot_steps(path: Path) -> int:
    """`ppo_25000_steps.zip` -> 25000 (0 if the name is not of that form)."""
    parts = path.stem.split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


KEEP_CHECKPOINTS = 5    # step snapshots kept per run (best_model.zip / final.zip are separate)


def prune_checkpoints(ckpt_dir: Path, keep: int | None = KEEP_CHECKPOINTS) -> list[Path]:
    """Delete all but the newest `keep` step snapshots (`ppo_<N>_steps.zip`).
    `keep=0` deletes every snapshot; `keep=None` disables pruning.
    Returns the deleted paths."""
    if keep is None or keep < 0 or not ckpt_dir.exists():
        return []
    snaps = sorted(ckpt_dir.glob("ppo_*_steps.zip"), key=_snapshot_steps)
    doomed = snaps[:-keep] if keep else snaps
    for p in doomed:
        try:
            p.unlink()
        except OSError:
            pass
    return doomed


def redundant_checkpoints(game: str, run_name: str, root: Path = MODELS_ROOT,
                          keep: int = KEEP_CHECKPOINTS) -> list[Path]:
    """The step snapshots `compact_run` would delete: every one if `final.zip`
    exists (it holds the last state), otherwise all but the newest `keep`."""
    ckpt = run_paths(game, run_name, root)["checkpoints"]
    if not ckpt.exists():
        return []
    snaps = sorted(ckpt.glob("ppo_*_steps.zip"), key=_snapshot_steps)
    return snaps if (ckpt / "final.zip").exists() else snaps[:-keep] if keep else snaps


def compact_run(game: str, run_name: str, root: Path = MODELS_ROOT,
                keep: int = KEEP_CHECKPOINTS) -> int:
    """Free space in a finished run without losing anything needed to play
    or resume it (see `redundant_checkpoints`). Returns bytes freed.
    `best_model.zip`, `final.zip`, eval history and TensorBoard events are
    never touched."""
    freed = 0
    for p in redundant_checkpoints(game, run_name, root, keep):
        try:
            size = p.stat().st_size
            p.unlink()
            freed += size
        except OSError:
            pass
    return freed


def slim_trial_run(game: str, run_name: str, root: Path = MODELS_ROOT) -> int:
    """A tuning trial only needs its eval history (and best model) to be
    scored; drop its checkpoints and TensorBoard events. Returns bytes freed."""
    paths = run_paths(game, run_name, root)
    freed = 0
    for d in (paths["checkpoints"], paths["tensorboard"]):
        if d.exists():
            freed += dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
    return freed


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.zip"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def best_model_for_run(game: str, run_name: str, root: Path = MODELS_ROOT) -> Path | None:
    """Best available model of a run, or None if it has none yet.

    Order: best eval-reward model, final model, newest step snapshot.
    """
    paths = run_paths(game, run_name, root)
    candidates = [paths["logs"] / "best_model.zip", paths["checkpoints"] / "final.zip"]
    snaps = sorted(paths["checkpoints"].glob("ppo_*_steps.zip"), key=_snapshot_steps)
    if snaps:
        candidates.append(snaps[-1])
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_model_path(model_arg: str | None, game: str, run_name: str,
                       root: Path = MODELS_ROOT) -> Path:
    """Explicit `--model` path, otherwise the run's best model. Raises if none."""
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    found = best_model_for_run(game, run_name, root)
    if found is None:
        base = run_paths(game, run_name, root)["base"]
        raise FileNotFoundError(
            f"No trained model for game='{game}' run='{run_name}' under {base}. "
            f"Expected logs/best_model.zip, checkpoints/final.zip or a ppo_*_steps.zip snapshot."
        )
    return found


def list_models(game_filter: str | None = None, root: Path = MODELS_ROOT) -> list[tuple[str, Path]]:
    """(label, path) for every model file of every run, best models first.

    Labels look like `mario/default — best`, `mario/default — final` and
    `mario/default — 25,000 steps` and double as the GUI dropdown text.
    """
    if game_filter:
        games = [game_filter]
    elif root.exists():
        games = sorted(d.name for d in root.iterdir() if d.is_dir())
    else:
        games = []
    bests, finals, snapshots = [], [], []
    for game in games:
        for run in list_runs(game, root):
            paths = run_paths(game, run, root)
            best = paths["logs"] / "best_model.zip"
            if best.exists():
                bests.append((f"{game}/{run} — best", best))
            final = paths["checkpoints"] / "final.zip"
            if final.exists():
                finals.append((f"{game}/{run} — final", final))
            for p in sorted(paths["checkpoints"].glob("ppo_*_steps.zip"), key=_snapshot_steps):
                snapshots.append((f"{game}/{run} — {_snapshot_steps(p):,} steps", p))
    return bests + finals + snapshots


# ------------------------- housekeeping -------------------------

TRIAL_NAME = re.compile(r"^(?P<prefix>.+)-(?P<config>\d{3})(?:-s(?P<seed>\d+))?$")


def dir_size(path: Path) -> int:
    """Bytes used below `path` (0 if it does not exist). Tolerates files and
    folders disappearing mid-scan: the GUI measures on a background thread
    while runs may be deleted (tuner pruning, Delete run)."""
    total = 0
    for dirpath, _dirs, files in os.walk(path):        # unreadable dirs are skipped
        for name in files:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def format_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def delete_run(game: str, run_name: str, root: Path = MODELS_ROOT) -> int:
    """Delete a run directory (checkpoints, logs, tensorboard, manifests).
    Returns the bytes freed. Refuses internal directories."""
    if not is_run_name(run_name):
        raise ValueError(f"'{run_name}' is not a run directory.")
    base = run_paths(game, run_name, root)["base"]
    if not base.exists():
        return 0
    freed = dir_size(base)
    shutil.rmtree(base)
    return freed


def tune_trial_runs(game: str, prefix: str, root: Path = MODELS_ROOT) -> list[str]:
    """Run names produced by a sweep with this prefix (`<prefix>-NNN[-sK]`)."""
    out = []
    for name in list_runs(game, root):
        m = TRIAL_NAME.match(name)
        if m and m.group("prefix") == prefix:
            out.append(name)
    return out


def tune_data_size(game: str, prefix: str, root: Path = MODELS_ROOT) -> tuple[int, int]:
    """(number of trial runs, bytes) a sweep prefix occupies, results file included."""
    names = tune_trial_runs(game, prefix, root)
    total = sum(dir_size(run_paths(game, n, root)["base"]) for n in names)
    results = root / game / "_tune" / f"{prefix}.json"
    if results.exists():
        total += results.stat().st_size
    return len(names), total


def delete_tune_data(game: str, prefix: str, root: Path = MODELS_ROOT) -> tuple[int, int]:
    """Delete every trial run of a sweep prefix and its results file.
    Returns (runs deleted, bytes freed)."""
    names = tune_trial_runs(game, prefix, root)
    freed = sum(delete_run(game, n, root) for n in names)
    results = root / game / "_tune" / f"{prefix}.json"
    if results.exists():
        freed += results.stat().st_size
        results.unlink()
    return len(names), freed


def keep_awake(pid: int) -> "subprocess.Popen | None":
    """macOS: prevent idle sleep while process `pid` runs (caffeinate exits
    with it). Elsewhere a no-op. Overnight runs depend on this."""
    if sys.platform != "darwin" or shutil.which("caffeinate") is None:
        return None
    try:
        return subprocess.Popen(["caffeinate", "-i", "-w", str(pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def open_in_file_manager(path: Path) -> None:
    """Reveal a folder in Finder / Explorer / the desktop's file manager."""
    path = Path(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


def build_train_cmd(cfg: dict, run_name: str, *, resume: bool = False,
                    timesteps: int | None = None, checkpoint_freq: int | None = None,
                    eval_freq: int | None = None) -> list[str]:
    """`train` command line for a preset-style config (see presets.PRESET_FIELDS).

    The Train tab, the tuner and the wizard all launch training through this
    so every run gets exactly the same flag set. Fields missing from `cfg`
    take the preset defaults. `timesteps`, `checkpoint_freq` and `eval_freq`
    override the config (the tuner uses them to give every trial the same
    length and scoring cadence).
    """
    cfg = normalize(cfg)
    if timesteps is not None:
        cfg["timesteps"] = timesteps
    if checkpoint_freq is not None:
        cfg["checkpoint_freq"] = checkpoint_freq
    if eval_freq is not None:
        cfg["eval_freq"] = eval_freq
    cmd = app_command("train")
    for key in PRESET_FIELDS:
        cmd += [f"--{key.replace('_', '-')}", str(cfg[key])]
    cmd += ["--run-name", run_name]
    if resume:
        cmd.append("--resume")
    return cmd
