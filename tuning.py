"""Hyperparameter-search helpers behind the GUI's Tune tab.

Deliberately Tk-free so the sweep logic (templates, grid/random expansion,
metric extraction, result persistence, time estimates) is unit-testable
and reusable from scripts.
"""
from __future__ import annotations

import itertools
import json
import random
import time
import zlib
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from presets import PRESET_DEFAULTS, PRESET_FIELDS

# Fields a sweep may vary. `timesteps` is fixed per trial, `seed` is handled
# by "seeds per config", and the checkpoint/eval cadence is derived from the
# trial length so every trial is scored the same way.
TUNABLE_FIELDS: frozenset[str] = frozenset(PRESET_FIELDS) - {
    "game", "timesteps", "seed", "checkpoint_freq", "eval_freq", "n_eval_episodes",
}

# Curated starting points for Super Mario Land with tile observations
# (MlpPolicy). Each is centred on the shipped defaults (ent_coef 0.01,
# lr 2.5e-4, n_steps 512, batch 256, n_epochs 4, gamma 0.99, gae 0.95,
# clip 0.2) and spans the range where PPO on this task is known to behave.
SWEEP_TEMPLATES: dict[str, dict[str, list]] = {
    "Quick start — entropy × learning rate (6 configs)": {
        "ent_coef": [0.005, 0.01, 0.03],
        "learning_rate": [1e-4, 3e-4],
    },
    "Learning rate (5 configs)": {
        "learning_rate": [5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3],
    },
    "Exploration — entropy coefficient (5 configs)": {
        "ent_coef": [0.0, 0.005, 0.01, 0.02, 0.05],
    },
    "Rollout shape — n_steps × batch size (6 configs)": {
        "n_steps": [256, 512, 1024],
        "batch_size": [256, 512],
    },
    "Update intensity — epochs × clip range (6 configs)": {
        "n_epochs": [3, 5, 10],
        "clip_range": [0.1, 0.2],
    },
    "Horizon — gamma × GAE lambda (6 configs)": {
        "gamma": [0.98, 0.99, 0.995],
        "gae_lambda": [0.9, 0.95],
    },
    "Input — frame stack × action repeat (4 configs)": {
        "frame_stack": [2, 4],
        "action_repeat": [4, 6],
    },
    "Broad random search (use Random, ~12 trials)": {
        "learning_rate": [1e-4, 2.5e-4, 5e-4],
        "ent_coef": [0.005, 0.01, 0.02, 0.05],
        "n_steps": [256, 512, 1024],
        "batch_size": [256, 512],
        "n_epochs": [3, 4, 6],
        "gamma": [0.99, 0.995],
    },
}

TEMPLATE_NOTES: dict[str, str] = {
    "Quick start — entropy × learning rate (6 configs)":
        "The two knobs with the largest effect. Run this first.",
    "Learning rate (5 configs)":
        "Too high diverges after the first level, too low never leaves 1-1.",
    "Exploration — entropy coefficient (5 configs)":
        "Higher = more random actions. Raise it if the agent gets stuck at one obstacle.",
    "Rollout shape — n_steps × batch size (6 configs)":
        "Longer rollouts see whole levels; bigger batches = fewer, smoother updates.",
    "Update intensity — epochs × clip range (6 configs)":
        "More epochs squeeze more out of each rollout but risk over-fitting to it.",
    "Horizon — gamma × GAE lambda (6 configs)":
        "How far ahead the agent values reward — long levels favour higher gamma.",
    "Input — frame stack × action repeat (4 configs)":
        "Changes the observation shape, so trials are not resumable across values.",
    "Broad random search (use Random, ~12 trials)":
        "Coarse map of the whole space; follow up with a focused grid.",
}

DEFAULT_TEMPLATE = "Quick start — entropy × learning rate (6 configs)"


# ------------------------- sweep expansion -------------------------

def validate_sweep(text: str) -> tuple[dict[str, list], str | None]:
    """Parse sweep JSON. Returns (sweep, None) or ({}, error_message)."""
    try:
        sweep = json.loads(text)
    except json.JSONDecodeError as e:
        return {}, f"not valid JSON: {e.msg} (line {e.lineno})"
    if not isinstance(sweep, dict) or not sweep:
        return {}, "sweep must be a non-empty JSON object: {\"param\": [values]}"
    for k, v in sweep.items():
        if k not in TUNABLE_FIELDS:
            return {}, f"'{k}' is not tunable (allowed: {', '.join(sorted(TUNABLE_FIELDS))})"
        if not isinstance(v, list) or not v:
            return {}, f"'{k}' must map to a non-empty list of values"
        if len(set(map(repr, v))) != len(v):
            return {}, f"'{k}' has duplicate values"
    return sweep, None


def n_grid_configs(sweep: dict[str, list]) -> int:
    n = 1
    for v in sweep.values():
        n *= len(v)
    return n


def expand_grid(sweep: dict[str, list]) -> list[dict]:
    keys = list(sweep)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(sweep[k] for k in keys))]


def sweep_seed(sweep: dict[str, list]) -> int:
    """Deterministic seed derived from the sweep itself.

    Random sampling MUST be reproducible: trials are named by their index,
    and "reuse finished trials" maps an index back to a config. A fresh
    random sample on every start would silently attach old results to
    different configs. Hashing the canonical JSON gives the same sample
    for the same sweep text across GUI restarts.
    """
    canonical = json.dumps(sweep, sort_keys=True, separators=(",", ":"))
    return zlib.crc32(canonical.encode("utf-8"))


def sample_random(sweep: dict[str, list], n: int, seed: int | None = None) -> list[dict]:
    """`n` distinct configs from the grid (the whole grid if it has <= n).

    With `seed=None` the sample is seeded from the sweep (see `sweep_seed`)
    so the same sweep always yields the same configs in the same order.
    """
    grid = expand_grid(sweep)
    if len(grid) <= n:
        return grid
    if seed is None:
        seed = sweep_seed(sweep)
    return random.Random(seed).sample(grid, n)


def describe_overrides(overrides: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in overrides.items())


# ------------------------- timing -------------------------

def estimate_fps(cfg: dict) -> float:
    """Rough env-steps/sec for a config on Apple-Silicon-class CPUs."""
    n_envs = max(1, int(cfg.get("n_envs", 10)))
    if cfg.get("obs_type", "tiles") == "pixels":
        return min(80.0, 15.0 * n_envs)
    return min(3300.0, 350.0 * n_envs)


def estimate_seconds(n_trials: int, trial_steps: int, cfg: dict) -> float:
    # ~8 s per trial for process start-up, env boot and the final eval.
    return n_trials * (trial_steps / estimate_fps(cfg) + 8.0)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def format_steps(n: int) -> str:
    """Compact step counts for progress labels: 4,000 / 250k / 1.25M."""
    n = int(n)
    if n < 10_000:
        return f"{n:,}"
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.2f}M"


class RateEstimator:
    """Throughput over the most recent progress reports.

    Feed it (time, units_done) at every report. The rate is measured between
    the oldest and newest sample in the window, so it excludes everything
    before the first report (process start-up, emulator boot, save-state
    bootstrapping) and follows changes in throughput (a preview running,
    evaluation pauses) within about a window's worth of reports.
    """

    def __init__(self, window: int = 20, min_span: float = 2.0):
        self._samples: deque[tuple[float, float]] = deque(maxlen=window)
        self._min_span = min_span

    def reset(self) -> None:
        self._samples.clear()

    def add(self, t: float, done: float) -> None:
        if self._samples and done < self._samples[-1][1]:
            self.reset()                     # counter went backwards: new run
        self._samples.append((t, done))

    def rate(self) -> float | None:
        """Units per second, or None until two reports at least `min_span`
        seconds apart exist."""
        if len(self._samples) < 2:
            return None
        (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
        if t1 - t0 < self._min_span or d1 <= d0:
            return None
        return (d1 - d0) / (t1 - t0)

    def eta(self, remaining: float) -> float | None:
        rate = self.rate()
        if rate is None:
            return None
        return max(0.0, remaining / rate)


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return " · ETA " + ("done" if seconds <= 0 else format_duration(seconds))


def progress_text(done: int, total: int, eta: float | None) -> str:
    """'1.25M / 2.00M · 62% · ETA 9 min' for a progress label."""
    pct = min(100.0, 100.0 * done / max(1, total))
    return f"{format_steps(done)} / {format_steps(total)} · {pct:.0f}%" + format_eta(eta)


class SweepEta:
    """Time-to-finish for a sweep of trials.

    Knows which planned trials will be reused (instant), measures how long
    each trained trial actually took, and follows the current trial's live
    step rate. Before any trial has finished, the current trial's projected
    duration stands in for the average.
    """

    TRIAL_TAIL = 3.0        # final evaluation + model save after the last step

    def __init__(self, n_trials_to_train: int, trial_steps: int):
        self.remaining_to_train = n_trials_to_train    # not yet started, will be trained
        self.trial_steps = trial_steps
        self.durations: list[float] = []
        self._rate = RateEstimator(window=15)
        self._trial_started: float | None = None

    def trial_started(self, now: float) -> None:
        self._trial_started = now
        self._rate.reset()
        self.remaining_to_train = max(0, self.remaining_to_train - 1)

    def trial_finished(self, now: float) -> None:
        if self._trial_started is not None:
            self.durations.append(now - self._trial_started)
        self._trial_started = None

    def report(self, now: float, steps: int) -> float | None:
        """ETA in seconds after a progress report of the current trial."""
        self._rate.add(now, steps)
        avg = sum(self.durations) / len(self.durations) if self.durations else None
        current = self._rate.eta(max(0, self.trial_steps - steps))
        if current is not None:
            current += self.TRIAL_TAIL
        elif avg is not None and self._trial_started is not None:
            current = max(0.0, avg - (now - self._trial_started))
        else:
            return None
        if avg is None and self._trial_started is not None:
            avg = (now - self._trial_started) + current
        return current + self.remaining_to_train * (avg or 0.0)

    def between_trials(self) -> float | None:
        """ETA while no trial is running (after one finished)."""
        if not self.durations:
            return None
        return self.remaining_to_train * (sum(self.durations) / len(self.durations))


# ------------------------- trial results -------------------------

METRIC_BEST = "best eval reward"
METRIC_FINAL = "final eval reward"
METRIC_MEAN = "mean eval reward"
METRIC_PER_MINUTE = "reward per compute-minute"
METRICS = (METRIC_BEST, METRIC_FINAL, METRIC_MEAN, METRIC_PER_MINUTE)
METRIC_NOTES = {
    METRIC_BEST: "Highest evaluation reward reached during the trial.",
    METRIC_FINAL: "Evaluation reward at the end of the trial.",
    METRIC_MEAN: "Average over all evaluations (rewards a fast, steady learner).",
    METRIC_PER_MINUTE: "Best eval reward divided by the trial's wall-clock minutes: how much "
                       "reward each minute of compute bought. Candidates that train slower "
                       "(more epochs, smaller batches, pixels) pay for it here. Negative "
                       "rewards count as zero.",
}


@dataclass
class TrialMetrics:
    best: float
    final: float
    mean: float
    ep_len: float
    timesteps: int
    n_evals: int
    duration: float | None = None      # wall-clock seconds the trial took (None = unknown)

    @property
    def per_minute(self) -> float:
        """Best reward per compute-minute; negative rewards count as zero."""
        if not self.duration or self.duration <= 0:
            return float("nan")
        return max(self.best, 0.0) / (self.duration / 60.0)

    def by_name(self, metric: str) -> float:
        return {METRIC_BEST: self.best, METRIC_FINAL: self.final,
                METRIC_MEAN: self.mean, METRIC_PER_MINUTE: self.per_minute}[metric]


def read_trial_metrics(eval_file: Path, duration: float | None = None) -> TrialMetrics | None:
    """Summarise SB3's evaluations.npz. None if missing/unreadable."""
    if not eval_file.exists():
        return None
    try:
        data = np.load(eval_file)
        means = data["results"].mean(axis=1)
        lens = data["ep_lengths"].mean(axis=1)
        best_i = int(np.argmax(means))
        return TrialMetrics(
            best=float(means[best_i]), final=float(means[-1]), mean=float(means.mean()),
            ep_len=float(lens[best_i]), timesteps=int(data["timesteps"][-1]),
            n_evals=int(len(means)), duration=duration,
        )
    except Exception:
        return None


def trial_run_name(prefix: str, config_idx: int, seed_idx: int, n_seeds: int) -> str:
    name = f"{prefix}-{config_idx:03d}"
    return f"{name}-s{seed_idx}" if n_seeds > 1 else name


MANIFEST_NAME = "trial.json"


def write_trial_manifest(run_dir: Path, config: dict, trial_steps: int) -> None:
    """Record which config a trial directory was trained with, so a later
    sweep only reuses it when the config still matches."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "trial_steps": trial_steps, "created_at": time.time()}
    (run_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True))


def finish_trial_manifest(run_dir: Path, duration: float) -> None:
    """Record how long the trial took, so a reused trial keeps its measured
    compute time for the reward-per-minute metric."""
    data = read_trial_manifest(run_dir) or {}
    data["duration"] = float(duration)
    data["finished_at"] = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / MANIFEST_NAME).write_text(json.dumps(data, indent=2, sort_keys=True))


def trial_duration(run_dir: Path, config: dict, trial_steps: int) -> float:
    """Measured wall-clock seconds from the manifest; for trials that predate
    duration recording, the throughput estimate for this config."""
    manifest = read_trial_manifest(run_dir)
    if manifest and isinstance(manifest.get("duration"), (int, float)) and manifest["duration"] > 0:
        return float(manifest["duration"])
    return estimate_seconds(1, trial_steps, config)


def read_trial_manifest(run_dir: Path) -> dict | None:
    path = run_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _same_config(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > 1e-12:
                return False
        elif str(va) != str(vb):
            return False
    return True


def trial_is_complete(run_dir: Path, trial_steps: int, config: dict | None = None) -> bool:
    """True if a previous run of this trial reached (about) its target.

    When `config` is given and the run has a manifest, the manifest's config
    must match too; a run with a different config is never reused even if
    it carries the same name. Runs without a manifest (older sweeps) are
    accepted on step count alone.
    """
    m = read_trial_metrics(run_dir / "logs" / "evaluations.npz")
    if m is None or m.timesteps < 0.9 * trial_steps:
        return False
    if config is None:
        return True
    manifest = read_trial_manifest(run_dir)
    if manifest is None:
        return True
    return _same_config(manifest.get("config", {}), config)


@dataclass
class ConfigResult:
    index: int                      # 1-based config number
    overrides: dict
    config: dict                    # full config (without seed)
    metric: str
    values: list[float] = field(default_factory=list)   # one per seed
    ep_lens: list[float] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    timesteps: int = 0
    duration: float = 0.0
    reused: int = 0                 # seeds skipped because results already existed
    pruned: bool = False            # run folders deleted to save space (scores kept)

    @property
    def score(self) -> float:
        return float(np.mean(self.values)) if self.values else float("-inf")

    @property
    def spread(self) -> float:
        return float(np.std(self.values)) if len(self.values) > 1 else 0.0

    @property
    def ep_len(self) -> float:
        return float(np.mean(self.ep_lens)) if self.ep_lens else float("nan")

    @property
    def label(self) -> str:
        return describe_overrides(self.overrides)

    def score_text(self) -> str:
        if not self.values or np.isnan(self.score):
            return "—"
        fmt = "{:.0f}" if abs(self.score) >= 100 else "{:.1f}"
        text = fmt.format(self.score)
        return f"{text} ± {fmt.format(self.spread)}" if len(self.values) > 1 else text


def results_file(project_dir: Path, game: str, prefix: str) -> Path:
    return project_dir / "models" / game / "_tune" / f"{prefix}.json"


def save_results(path: Path, meta: dict, results: list[ConfigResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {**meta, "saved_at": time.time()},
               "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(payload, indent=2))


def load_results(path: Path) -> tuple[dict, list[ConfigResult]]:
    payload = json.loads(path.read_text())
    results = [ConfigResult(**r) for r in payload.get("results", [])]
    return payload.get("meta", {}), results


def full_config(base: dict, overrides: dict) -> dict:
    """Base preset + defaults for optional fields + sweep overrides."""
    return {**PRESET_DEFAULTS, **base, **overrides}


def prune_trial_runs(results, keep: int, game: str, delete_run) -> list[str]:
    """Delete the run folders of every config outside the `keep` best.

    Scores stay in `results` (and in the saved results file); only the
    training artefacts on disk go. Unscored configs (crashed trials) are
    pruned too. `keep <= 0` disables pruning. Returns the deleted run names.
    """
    if keep <= 0:
        return []
    ranked = rank_results(results)
    deleted: list[str] = []
    for res in ranked[keep:]:
        if res.pruned:
            continue
        for run_name in res.runs:
            try:
                delete_run(game, run_name)
                deleted.append(run_name)
            except (OSError, ValueError):
                pass
        res.pruned = True
    return deleted


def _sortable(score: float) -> float:
    return float("-inf") if np.isnan(score) else score


def best_result(results) -> ConfigResult | None:
    """Highest-scoring config that actually produced a score."""
    scored = [r for r in results if r.values and not np.isnan(r.score)]
    return max(scored, key=lambda r: r.score) if scored else None


def rank_results(results) -> list[ConfigResult]:
    """Best first; unscored configs last in index order."""
    return sorted(results, key=lambda r: (-_sortable(r.score), r.index))
