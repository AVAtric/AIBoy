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

import os
import sys
from pathlib import Path

from presets import PRESET_DEFAULTS

MODELS_ROOT = Path("models")
INTERNAL_PREFIX = "_"

# Baseline cadence used when a config does not specify one.
DEFAULT_CHECKPOINT_FREQ = 25_000
DEFAULT_EVAL_FREQ = 10_000
DEFAULT_N_EVAL_EPISODES = 3


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
    entries: list[tuple[str, Path]] = []
    games = [game_filter] if game_filter else sorted(
        d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []
    for game in games:
        for run in list_runs(game, root):
            paths = run_paths(game, run, root)
            best = paths["logs"] / "best_model.zip"
            if best.exists():
                entries.append((f"{game}/{run} — best", best))
    for game in games:
        for run in list_runs(game, root):
            paths = run_paths(game, run, root)
            final = paths["checkpoints"] / "final.zip"
            if final.exists():
                entries.append((f"{game}/{run} — final", final))
    for game in games:
        for run in list_runs(game, root):
            snaps = sorted(run_paths(game, run, root)["checkpoints"].glob("ppo_*_steps.zip"),
                           key=_snapshot_steps)
            for p in snaps:
                entries.append((f"{game}/{run} — {_snapshot_steps(p):,} steps", p))
    return entries


def build_train_cmd(cfg: dict, run_name: str, *, resume: bool = False,
                    timesteps: int | None = None, checkpoint_freq: int | None = None,
                    eval_freq: int | None = None, script: str = "main.py") -> list[str]:
    """`main.py train` argv for a preset-style config (see presets.PRESET_FIELDS).

    The Train tab, the tuner and the wizard all launch training through this
    so every run gets exactly the same flag set. `timesteps`,
    `checkpoint_freq` and `eval_freq` override the config (the tuner uses
    them to give every trial the same length and scoring cadence).
    """
    cmd = [
        sys.executable, "-u", script, "train",
        "--game", str(cfg.get("game", "mario")),
        "--n-envs", str(cfg["n_envs"]),
        "--timesteps", str(timesteps if timesteps is not None else cfg["timesteps"]),
        "--ent-coef", str(cfg["ent_coef"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--n-steps", str(cfg["n_steps"]),
        "--batch-size", str(cfg["batch_size"]),
        "--obs-type", str(cfg.get("obs_type", "tiles")),
        "--start-level", str(cfg.get("start_level", "default")),
        "--device", str(cfg.get("device", "cpu")),
        "--action-repeat", str(cfg.get("action_repeat", 4)),
        "--frame-stack", str(cfg.get("frame_stack", 4)),
        "--n-epochs", str(cfg.get("n_epochs", 4)),
        "--gamma", str(cfg.get("gamma", PRESET_DEFAULTS["gamma"])),
        "--gae-lambda", str(cfg.get("gae_lambda", PRESET_DEFAULTS["gae_lambda"])),
        "--clip-range", str(cfg.get("clip_range", PRESET_DEFAULTS["clip_range"])),
        "--seed", str(cfg.get("seed", 0)),
        "--checkpoint-freq", str(checkpoint_freq if checkpoint_freq is not None
                                 else cfg.get("checkpoint_freq", DEFAULT_CHECKPOINT_FREQ)),
        "--eval-freq", str(eval_freq if eval_freq is not None
                           else cfg.get("eval_freq", DEFAULT_EVAL_FREQ)),
        "--n-eval-episodes", str(cfg.get("n_eval_episodes", DEFAULT_N_EVAL_EPISODES)),
        "--run-name", run_name,
    ]
    if resume:
        cmd.append("--resume")
    return cmd
