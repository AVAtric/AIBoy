"""Hyperparameter-search helpers behind the Tune tab and the wizard.

Deliberately Tk-free so the sweep logic (templates, grid / random / explicit
expansion, metric extraction, result persistence, time estimates) is
unit-testable and reusable from scripts. What AIboy remembers about past
trials lives in experience.py, which builds on this module (never the other
way round).

A sweep is either a grid, `{"param": [values, ...]}`, or an explicit list of
candidates, `[{"param": value, ...}, ...]` (what the experience-based
suggestions produce). Both expand to a list of override dicts.
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

from aiboy.paths import cpu_count
from aiboy.presets import DEFAULT_CONFIG, PRESET_DEFAULTS, PRESET_FIELDS

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

# ------------------------- plain language (wizard) -------------------------
# The wizard is for people who have never heard of a learning rate. Every
# template, tunable field and search effort has a plain name here; the
# expert names stay on the Tune tab.

TEMPLATE_PLAIN: dict[str, str] = {
    "Quick start — entropy × learning rate (6 configs)":
        "Curiosity × learning speed — the two that matter most (6 variations)",
    "Learning rate (5 configs)": "Learning speed only (5 variations)",
    "Exploration — entropy coefficient (5 configs)":
        "Curiosity: how often it tries something new (5 variations)",
    "Rollout shape — n_steps × batch size (6 configs)":
        "How much play it reviews per lesson (6 variations)",
    "Update intensity — epochs × clip range (6 configs)":
        "How hard it studies each lesson (6 variations)",
    "Horizon — gamma × GAE lambda (6 configs)": "How far ahead it plans (6 variations)",
    "Input — frame stack × action repeat (4 configs)":
        "How it sees the screen and how quickly it reacts (4 variations)",
    "Broad random search (use Random, ~12 trials)": "A bit of everything: 12 random mixes",
}
TEMPLATE_FROM_PLAIN = {plain: name for name, plain in TEMPLATE_PLAIN.items()}
TEMPLATE_PLAIN_NOTES: dict[str, str] = {
    "Quick start — entropy × learning rate (6 configs)": "Start here. Usually the biggest win.",
    "Learning rate (5 configs)": "Too fast and it forgets; too slow and it never gets past 1-1.",
    "Exploration — entropy coefficient (5 configs)":
        "More curiosity helps when it keeps getting stuck at one spot.",
    "Rollout shape — n_steps × batch size (6 configs)":
        "Longer reviews see whole levels; bigger batches learn more smoothly.",
    "Update intensity — epochs × clip range (6 configs)":
        "Studying harder gets more out of each lesson but can over-fit it.",
    "Horizon — gamma × GAE lambda (6 configs)": "Long levels favour planning further ahead.",
    "Input — frame stack × action repeat (4 configs)":
        "Changes what the agent sees; the winner cannot continue an older run.",
    "Broad random search (use Random, ~12 trials)":
        "A rough map of everything at once; follow up with a focused search.",
}


def template_search(name: str) -> tuple[str, int]:
    """(search type, sample size) a template is meant to be run with: the
    broad template's full grid is hundreds of configs, so it samples."""
    if name == "Broad random search (use Random, ~12 trials)":
        return "random", 12
    return "grid", 0


PLAIN_KEYS: dict[str, str] = {
    "ent_coef": "curiosity", "learning_rate": "learning speed", "n_steps": "review length",
    "batch_size": "batch size", "n_epochs": "study passes", "clip_range": "step size",
    "gamma": "foresight", "gae_lambda": "smoothing", "frame_stack": "frames seen",
    "action_repeat": "reaction time", "n_envs": "parallel games", "obs_type": "eyes",
    "start_level": "level", "device": "device", "time_budget": "time budget",
    "stall_steps": "stall limit",
}
BASELINE_PLAIN = "the goal's own settings"

# Level modes as table cells (the Presets and Experience tabs).
MODE_LABEL = {"default": "campaign", "random": "random levels", "sequential": "level by level",
              "marathon": "marathon"}


def mode_label(cfg: dict) -> str:
    """'campaign', 'random levels', … or 'level 3-2' for a fixed level."""
    level = str(cfg.get("start_level", "default"))
    return MODE_LABEL.get(level, f"level {level}")

SHORT_KEYS = {"learning_rate": "lr", "ent_coef": "ent", "n_steps": "steps", "batch_size": "batch",
              "n_epochs": "epochs", "gamma": "gamma", "gae_lambda": "gae", "clip_range": "clip",
              "n_envs": "envs", "action_repeat": "repeat", "frame_stack": "stack",
              "obs_type": "obs", "start_level": "level", "device": "device",
              "time_budget": "budget", "stall_steps": "stall", "seed": "seed"}
# The knobs shown when a whole configuration is summarised in one cell.
KEY_PARAMS = ("learning_rate", "ent_coef", "n_steps", "batch_size", "n_epochs", "gamma")


def format_value(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def compact_overrides(overrides: dict) -> str:
    """'ent 0.03 · lr 0.0003' — readable summary of tuned values."""
    return " · ".join(f"{SHORT_KEYS.get(k, k)} {format_value(v)}" for k, v in overrides.items())


def compact_config(cfg: dict, keys=KEY_PARAMS) -> str:
    """The key knobs of a configuration: 'lr 0.00025 · ent 0.01 · steps 512 · …'."""
    return compact_overrides({k: cfg[k] for k in keys if k in cfg})


def config_diff(cfg: dict, base: dict, keys=None) -> dict:
    """The tunable fields on which `cfg` differs from `base` (a field missing
    from either side counts as its default)."""
    keys = TUNABLE_FIELDS if keys is None else keys
    out = {}
    for k in keys:
        if k in cfg and not _same_value(cfg[k], base.get(k, DEFAULT_CONFIG.get(k))):
            out[k] = cfg[k]
    return out


def _same_value(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12
    return str(a) == str(b)


def plain_overrides(overrides: dict) -> str:
    """'curiosity 0.03 · learning speed 0.0003' for a set of tuned values;
    the baseline (no overrides) reads as `BASELINE_PLAIN`."""
    if not overrides:
        return BASELINE_PLAIN
    parts = []
    for key, value in overrides.items():
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{PLAIN_KEYS.get(key, key)} {text}")
    return " · ".join(parts)


# Search effort: (seeds per variation, trial length as a share of the
# suggested 5 % of the goal). One seed cannot separate luck from skill;
# two is the sweet spot, three for a decision that matters.
EFFORT_LEVELS: dict[str, tuple[int, float]] = {
    "Quick": (1, 0.5), "Normal": (2, 1.0), "Thorough": (3, 2.0),
}
DEFAULT_EFFORT = "Normal"
EFFORT_NOTES = {
    "Quick": "each variation trained once, briefly — a first impression",
    "Normal": "each variation trained twice — a fair comparison (recommended)",
    "Thorough": "each variation trained three times, twice as long — for a decision that matters",
}


def effort_plan(goal_timesteps: int, level: str) -> tuple[int, int]:
    """(seeds, steps per trial) for an effort level and a training goal."""
    seeds, factor = EFFORT_LEVELS[level]
    steps = suggested_trial_steps(goal_timesteps) * factor
    return seeds, int(round(max(50_000, min(1_000_000, steps)) / 50_000) * 50_000)


def explain_winner(results) -> tuple["ConfigResult | None", str]:
    """`pick_winner` with a plain-language explanation for the wizard."""
    winner, _why = pick_winner(results)
    if winner is None:
        return None, "No variation could be scored."
    best = best_result(results)
    baseline = next((r for r in results if not r.overrides and r.values), None)
    if baseline is None or best is baseline:
        return winner, (f"{plain_overrides(winner.overrides).capitalize()} scored best "
                        f"({winner.score_text()})." + ("" if baseline is None
                                                        else " No variation did better."))
    if winner is best:
        return winner, (f"'{plain_overrides(best.overrides)}' scored {best.score_text()}, clearly "
                        f"better than {BASELINE_PLAIN} ({baseline.score_text()}).")
    return winner, (f"'{plain_overrides(best.overrides)}' scored {best.score_text()} against "
                    f"{baseline.score_text()} for {BASELINE_PLAIN}: too close to call, so the "
                    f"goal's own settings are kept.")


# ------------------------- sweep expansion -------------------------

Sweep = "dict[str, list] | list[dict]"


def validate_sweep(text: str) -> tuple["dict[str, list] | list[dict]", str | None]:
    """Parse sweep JSON: a grid `{"param": [values]}` or an explicit list of
    candidates `[{"param": value}, ...]`. Returns (sweep, None) or
    ({}, error_message)."""
    try:
        sweep = json.loads(text)
    except json.JSONDecodeError as e:
        return {}, f"not valid JSON: {e.msg} (line {e.lineno})"
    if isinstance(sweep, list):
        if not sweep or not all(isinstance(c, dict) for c in sweep):
            return {}, "a candidate list must be a non-empty JSON list of objects: [{\"param\": value}]"
        for cand in sweep:
            for k in cand:
                if k not in TUNABLE_FIELDS:
                    return {}, f"'{k}' is not tunable (allowed: {', '.join(sorted(TUNABLE_FIELDS))})"
        if len({json.dumps(c, sort_keys=True) for c in sweep}) != len(sweep):
            return {}, "the candidate list has duplicates"
        return sweep, None
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


def n_grid_configs(sweep) -> int:
    if isinstance(sweep, list):
        return len(sweep)
    n = 1
    for v in sweep.values():
        n *= len(v)
    return n


def expand_grid(sweep) -> list[dict]:
    """Every candidate of a sweep: the grid's cartesian product, or the
    explicit list as it is."""
    if isinstance(sweep, list):
        return [dict(c) for c in sweep]
    keys = list(sweep)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(sweep[k] for k in keys))]


def sweep_seed(sweep) -> int:
    """Deterministic seed derived from the sweep itself.

    Random sampling MUST be reproducible: trials are named by their index,
    and "reuse finished trials" maps an index back to a config. A fresh
    random sample on every start would silently attach old results to
    different configs. Hashing the canonical JSON gives the same sample
    for the same sweep text across GUI restarts.
    """
    canonical = json.dumps(sweep, sort_keys=True, separators=(",", ":"))
    return zlib.crc32(canonical.encode("utf-8"))


def sample_random(sweep, n: int, seed: int | None = None) -> list[dict]:
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


BASELINE_LABEL = "(base preset, unchanged)"


def with_baseline(combos: list[dict]) -> list[dict]:
    """The base preset itself as candidate 1, so a sweep can only 'win' by
    beating what the user already had."""
    return [{}] + [c for c in combos if c]


def suggested_trial_steps(goal_timesteps: int) -> int:
    """Trial length for the wizard: 5% of the final run, 100k..1M, rounded
    to 50k. Short trials rank learning speed, not final skill; 5% is the
    smallest share that has been informative here."""
    steps = int(goal_timesteps) * 0.05
    steps = min(1_000_000, max(100_000, steps))
    return int(round(steps / 50_000) * 50_000)


MIN_WIN_MARGIN = 0.05     # a winner must lead by at least this share of the baseline score


def pick_winner(results) -> tuple["ConfigResult | None", str]:
    """The candidate to train with, and why.

    The best-scoring candidate wins only if it beats the baseline (overrides
    == {}) by more than the larger of the two seed spreads, and by at least
    `MIN_WIN_MARGIN` of the baseline's score (single-seed searches have no
    spread to speak of, and PPO scores wobble by a few percent between
    otherwise identical runs); otherwise the baseline is kept. Without a
    baseline the best candidate wins.
    """
    best = best_result(results)
    if best is None:
        return None, "no scored candidate"
    baseline = next((r for r in results if not r.overrides and r.values), None)
    if baseline is None or best is baseline:
        return best, ("the base preset scored best; no candidate beat it" if best is baseline
                      else "best candidate")
    margin = max(best.spread, baseline.spread, MIN_WIN_MARGIN * abs(baseline.score))
    if best.score - baseline.score > margin:
        return best, (f"beats the base preset by {best.score - baseline.score:.0f} "
                      f"(spread {margin:.0f})")
    return baseline, (f"'{best.label}' leads by only {best.score - baseline.score:.0f}, within "
                      f"the seed spread of {margin:.0f}; keeping the base preset")


# ------------------------- timing -------------------------

def estimate_fps(cfg: dict, measured: float | None = None) -> float:
    """Env-steps/sec for a config: what this computer measured on earlier
    trials and runs with the same setup (see experience.Experience.fps),
    else a rough guess for Apple-Silicon-class CPUs."""
    if measured is not None and measured > 0:
        return float(measured)
    n_envs = max(1, int(cfg.get("n_envs", 10)))
    if cfg.get("obs_type", "tiles") == "pixels":
        return min(80.0, 15.0 * n_envs)
    return min(3300.0, 350.0 * n_envs)


TRIAL_OVERHEAD = 8.0    # seconds per trial for process start-up, env boot and the final eval


def estimate_seconds(n_trials: int, trial_steps: int, cfg: dict,
                     measured_fps: float | None = None) -> float:
    return n_trials * (trial_steps / estimate_fps(cfg, measured_fps) + TRIAL_OVERHEAD)


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
METRIC_LATE = "late eval reward"
METRIC_FINAL = "final eval reward"
METRIC_MEAN = "mean eval reward"
METRIC_PER_MINUTE = "reward per compute-minute"
METRICS = (METRIC_LATE, METRIC_BEST, METRIC_FINAL, METRIC_MEAN, METRIC_PER_MINUTE)
METRIC_NOTES = {
    METRIC_LATE: "Average of the last third of the evaluations: where the trial ended up, "
                 "smoothed over several evals. The most reliable predictor of a long run; "
                 "the wizard uses it.",
    METRIC_BEST: "Highest single evaluation reached during the trial (rewards lucky spikes).",
    METRIC_FINAL: "Evaluation reward at the very end of the trial (a single, noisy sample).",
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
    late: float = float("nan")         # mean of the last third of the evaluations

    @property
    def per_minute(self) -> float:
        """Best reward per compute-minute; negative rewards count as zero."""
        if not self.duration or self.duration <= 0:
            return float("nan")
        return max(self.best, 0.0) / (self.duration / 60.0)

    def by_name(self, metric: str) -> float:
        return {METRIC_BEST: self.best, METRIC_LATE: self.late, METRIC_FINAL: self.final,
                METRIC_MEAN: self.mean, METRIC_PER_MINUTE: self.per_minute}[metric]


def evals_from_npz(eval_file: Path) -> list[list[float]] | None:
    """SB3's evaluations.npz as `[[timesteps, mean reward, mean episode length], …]`,
    one entry per evaluation. None if the file is missing or unreadable. This
    is the form the experience file stores, so scores can be recomputed for
    any metric long after the run folder is gone."""
    if not Path(eval_file).exists():
        return None
    try:
        data = np.load(eval_file)
        steps, results, lengths = data["timesteps"], data["results"], data["ep_lengths"]
        return [[int(t), float(np.mean(r)), float(np.mean(l))]
                for t, r, l in zip(steps, results, lengths)]
    except Exception:
        return None


def metrics_from_evals(evals, duration: float | None = None) -> TrialMetrics | None:
    """Summarise an eval history (see `evals_from_npz`). None if empty."""
    if not evals:
        return None
    try:
        means = np.array([e[1] for e in evals], dtype=float)
        lens = np.array([e[2] for e in evals], dtype=float)
        best_i = int(np.argmax(means))
        n_late = min(len(means), max(2, (len(means) + 2) // 3))   # last third, at least 2 evals
        return TrialMetrics(
            best=float(means[best_i]), final=float(means[-1]), mean=float(means.mean()),
            ep_len=float(lens[best_i]), timesteps=int(evals[-1][0]),
            n_evals=int(len(means)), duration=duration, late=float(means[-n_late:].mean()),
        )
    except (IndexError, TypeError, ValueError):
        return None


def read_trial_metrics(eval_file: Path, duration: float | None = None) -> TrialMetrics | None:
    """Summarise SB3's evaluations.npz. None if missing/unreadable."""
    return metrics_from_evals(evals_from_npz(eval_file), duration)


def record_metrics(record: dict) -> TrialMetrics | None:
    """Metrics of an experience record (a dict as stored in experience.jsonl)."""
    return metrics_from_evals(record.get("evals") or [], record.get("duration"))


def trial_run_name(prefix: str, config_idx: int, seed_idx: int, n_seeds: int) -> str:
    name = f"{prefix}-{config_idx:03d}"
    return f"{name}-s{seed_idx}" if n_seeds > 1 else name


@dataclass
class ConfigResult:
    index: int                      # 1-based config number
    overrides: dict
    config: dict                    # full config (without seed)
    metric: str
    values: list[float] = field(default_factory=list)   # one per seed, under `metric`
    ep_lens: list[float] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    timesteps: int = 0
    duration: float = 0.0
    reused: int = 0                 # seeds whose result AIboy already knew (not retrained)
    pruned: bool = False            # run folders deleted to save space (scores kept)
    records: list[dict] = field(default_factory=list)   # experience records, one per scored seed

    def add_record(self, record: dict) -> None:
        """Take a trial's experience record and score it under `metric`."""
        self.records.append(record)
        m = record_metrics(record)
        if m is not None:
            self.values.append(m.by_name(self.metric))
            self.ep_lens.append(m.ep_len)
            self.timesteps = m.timesteps

    def rescore(self, metric: str) -> None:
        """Recompute the scores under another metric from the stored eval
        histories, so switching metrics needs no run folder on disk."""
        self.metric, self.values, self.ep_lens = metric, [], []
        for record in self.records:
            m = record_metrics(record)
            if m is not None:
                self.values.append(m.by_name(metric))
                self.ep_lens.append(m.ep_len)

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
        return describe_overrides(self.overrides) or BASELINE_LABEL

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


def trial_config(base: dict, overrides: dict, trial_steps: int, seed_offset: int = 0) -> dict:
    """The complete configuration one trial is trained with: the base preset
    with the sweep's overrides, the trial length, and the seed of this
    repetition. This is what the experience file is keyed on."""
    cfg = {**DEFAULT_CONFIG, **full_config(base, overrides)}
    cfg["timesteps"] = int(trial_steps)
    cfg["seed"] = int(base.get("seed", 0)) + seed_offset
    cfg["n_envs"] = min(int(cfg["n_envs"]), cpu_count())     # what the trainer will actually run
    return cfg


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
