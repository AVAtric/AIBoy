"""What AIboy has learned: every trial and training run it has finished.

The trainer appends one record per run to `experience.jsonl` (next to the
user's data, see paths.EXPERIENCE_FILE): the complete configuration, the
evaluation history, how long it took and where it came from. Nothing is ever
computed twice from that file's point of view:

  * a sweep that plans a trial AIboy already knows reuses the stored result
    instead of training again (`find`), whatever the earlier sweep was
    called and even after its run folders were deleted;
  * the wizard's "Let AIboy choose" search asks for the best known settings
    of a goal (`best_for_task`) and for untested variations around them
    (`suggest`, `auto_plan`), so every wizard run explores something new;
  * time estimates use the throughput this computer actually achieved
    (`fps`) instead of a guess.

Two identities matter. The *outcome signature* is everything that decides a
trial's result (OUTCOME_FIELDS + the environment version): two records with
the same signature are repeats of the same experiment. The *task* is the
subset that defines what is being learned and for how long (TASK_FIELDS):
scores are only compared within a task, because a marathon reward and a
campaign reward, or a 50k-step trial and a 500k-step trial, mean different
things.

The file is append-only JSON lines: cheap to write from a subprocess, robust
to a crash mid-write (a torn last line is skipped), and easy to inspect.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiboy import tuning
from aiboy.games import ENV_VERSION
from aiboy.paths import EXPERIENCE_FILE, cpu_count
from aiboy.presets import DEFAULT_CONFIG, normalize

# Fields that decide how a run turns out. Not in the list: the checkpoint /
# eval cadence, the device (numerically irrelevant) and the run name.
OUTCOME_FIELDS = (
    "game", "obs_type", "start_level", "action_repeat", "frame_stack", "time_budget",
    "stall_steps", "timesteps", "n_envs", "learning_rate", "ent_coef", "n_steps", "batch_size",
    "n_epochs", "gamma", "gae_lambda", "clip_range", "seed",
)
# Fields that define the task: what is learned, how the agent sees it and
# for how long. Everything else in OUTCOME_FIELDS is a hyperparameter.
TASK_FIELDS = ("game", "obs_type", "start_level", "action_repeat", "frame_stack", "time_budget",
               "stall_steps", "timesteps")
HYPER_FIELDS = tuple(k for k in OUTCOME_FIELDS if k not in TASK_FIELDS and k != "seed")

COMPLETE_SHARE = 0.9        # a run that reached this share of its target counts as finished
SOURCES = ("cli", "train", "tune", "wizard")
KIND_TRIAL, KIND_RUN = "trial", "run"


def _canon(value):
    """Values as they are compared and hashed: numbers by value, the rest as text."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return str(value)


def signature_of(cfg: dict, fields=OUTCOME_FIELDS, env_version: str = ENV_VERSION) -> str:
    """Canonical text of the outcome-deciding fields of a configuration."""
    cfg = {**DEFAULT_CONFIG, **cfg}
    parts = {k: _canon(cfg[k]) for k in fields if k in cfg}
    parts["env_version"] = env_version
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def task_of(cfg: dict, env_version: str = ENV_VERSION) -> str:
    return signature_of(cfg, TASK_FIELDS, env_version)


@dataclass
class Record:
    """One finished (or interrupted) training process."""
    kind: str                        # "trial" (short search run) | "run" (a real training)
    config: dict                     # complete preset-style config; `timesteps` is the target
    evals: list                      # [[timesteps, mean reward, mean episode length], …]
    duration: float | None           # wall-clock seconds
    complete: bool                   # reached (about) its target and was not interrupted
    source: str = "cli"              # cli | train | tune | wizard
    run_name: str = ""
    resumed: bool = False
    env_version: str = ENV_VERSION
    created_at: float = field(default_factory=time.time)
    id: str = ""

    def __post_init__(self):
        self.config = normalize(self.config)
        if not self.id:
            raw = f"{self.signature}|{self.created_at}|{self.run_name}"
            self.id = hashlib.sha1(raw.encode()).hexdigest()[:12]

    @property
    def signature(self) -> str:
        return signature_of(self.config, env_version=self.env_version)

    @property
    def task(self) -> str:
        return task_of(self.config, self.env_version)

    @property
    def game(self) -> str:
        return str(self.config.get("game", "mario"))

    @property
    def steps_done(self) -> int:
        return int(self.evals[-1][0]) if self.evals else 0

    @property
    def fps(self) -> float | None:
        if self.duration and self.duration > 0 and self.steps_done > 0:
            return self.steps_done / self.duration
        return None

    def metrics(self) -> tuning.TrialMetrics | None:
        return tuning.metrics_from_evals(self.evals, self.duration)

    def score(self, metric: str = tuning.METRIC_LATE) -> float:
        m = self.metrics()
        return m.by_name(metric) if m is not None else float("nan")

    def usable(self) -> bool:
        """Good enough to stand in for a new trial with the same signature."""
        return self.complete and not self.resumed and bool(self.evals) \
            and self.env_version == ENV_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        known.setdefault("kind", KIND_RUN)
        known.setdefault("evals", [])
        known.setdefault("duration", None)
        known.setdefault("complete", False)
        return cls(**known)


def make_record(cfg: dict, evals, duration: float | None, *, completed: bool, source: str,
                run_name: str, resumed: bool = False) -> Record:
    """Build the record of a run from what the trainer knows when it ends."""
    cfg = normalize(cfg)
    done = int(evals[-1][0]) if evals else 0
    complete = bool(completed) and done >= COMPLETE_SHARE * int(cfg.get("timesteps", 0))
    kind = KIND_TRIAL if source in ("tune", "wizard") else KIND_RUN
    return Record(kind=kind, config=cfg, evals=[list(map(float, e)) for e in (evals or [])],
                  duration=duration, complete=complete, source=source, run_name=run_name,
                  resumed=resumed)


def append_record(record: Record, path: Path = EXPERIENCE_FILE) -> None:
    """Append one record (a single line). Used by the trainer process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")


def load_records(path: Path = EXPERIENCE_FILE) -> list[Record]:
    """Every record in the file, oldest first. Unreadable lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[Record] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                out.append(Record.from_dict(data))
        except (ValueError, TypeError):
            continue
    return out


class Experience:
    """The in-memory view of the experience file, shared by the tabs.

    `reload()` re-reads the file (the trainer subprocess appends to it);
    every query works on the snapshot in memory. Safe to call from the
    sweep thread and the Tk thread: the record list is replaced, never
    mutated in place.
    """

    def __init__(self, path: Path = EXPERIENCE_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.records: list[Record] = []
        self.reload()

    # ---------- storage ----------

    def reload(self) -> None:
        records = load_records(self.path)
        with self._lock:
            self.records = records

    def add(self, record: Record) -> None:
        append_record(record, self.path)
        with self._lock:
            self.records = [*self.records, record]

    def forget(self, ids) -> int:
        """Delete records by id; rewrites the file. Returns how many went."""
        ids = set(ids)
        with self._lock:
            keep = [r for r in self.records if r.id not in ids]
            gone = len(self.records) - len(keep)
            if gone:
                self._write_all(keep)
                self.records = keep
        return gone

    def clear(self) -> int:
        with self._lock:
            n = len(self.records)
            self._write_all([])
            self.records = []
        return n

    def _write_all(self, records: list[Record]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), separators=(",", ":")) + "\n")
        os.replace(tmp, self.path)

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    # ---------- queries ----------

    def for_game(self, game: str | None = None) -> list[Record]:
        return [r for r in self.records if game is None or r.game == game]

    def latest_for_run(self, run_name: str, since: float = 0.0) -> Record | None:
        """The newest record a run of this name produced after `since`,
        complete or not (a sweep reads its trial's result this way)."""
        for r in reversed(self.records):
            if r.run_name == run_name and r.created_at >= since:
                return r
        return None

    def find(self, cfg: dict) -> Record | None:
        """The newest usable record of exactly this experiment (same outcome
        signature: settings, seed, trial length, environment version)."""
        sig = signature_of(cfg)
        for r in reversed(self.records):
            if r.usable() and r.signature == sig:
                return r
        return None

    def known(self, cfg: dict) -> bool:
        return self.find(cfg) is not None

    def known_seeds(self, base: dict, overrides: dict, trial_steps: int, n_seeds: int) -> int:
        """How many of the `n_seeds` repetitions of a candidate are already known."""
        return sum(1 for s in range(n_seeds)
                   if self.known(tuning.trial_config(base, overrides, trial_steps, s)))

    def trials_of_task(self, cfg: dict, *, any_length: bool = False) -> list[Record]:
        """Usable trial records of the same task as `cfg`. With `any_length`
        the trial length is ignored (the longest trials come first)."""
        if any_length:
            fields = tuple(k for k in TASK_FIELDS if k != "timesteps")
            want = signature_of(cfg, fields)
            found = [r for r in self.records if r.kind == KIND_TRIAL and r.usable()
                     and signature_of(r.config, fields) == want]
            return sorted(found, key=lambda r: -int(r.config.get("timesteps", 0)))
        want = task_of(cfg)
        return [r for r in self.records if r.kind == KIND_TRIAL and r.usable() and r.task == want]

    def best_for_task(self, cfg: dict, metric: str = tuning.METRIC_LATE) -> "Knowledge | None":
        """The best known hyperparameters for the task of `cfg`, averaged over
        seeds. Prefers trials of the same length; falls back to the longest
        trials of the task at another length."""
        trials = self.trials_of_task(cfg)
        if not trials:
            longer = self.trials_of_task(cfg, any_length=True)
            if not longer:
                return None
            length = int(longer[0].config["timesteps"])
            trials = [r for r in longer if int(r.config["timesteps"]) == length]
        groups: dict[str, list[Record]] = {}
        hyper = tuple(k for k in OUTCOME_FIELDS if k != "seed")
        for r in trials:
            groups.setdefault(signature_of(r.config, hyper), []).append(r)
        best_key, best_score, best_group = None, -math.inf, []
        for key, group in groups.items():
            scores = [s for s in (r.score(metric) for r in group) if not math.isnan(s)]
            if not scores:
                continue
            mean = statistics.fmean(scores)
            if mean > best_score:
                best_key, best_score, best_group = key, mean, group
        if best_key is None:
            return None
        return Knowledge(config=dict(best_group[-1].config), score=best_score,
                         n_seeds=len(best_group), n_trials=len(trials),
                         trial_steps=int(best_group[-1].config["timesteps"]),
                         when=max(r.created_at for r in best_group))

    def fps(self, cfg: dict, kinds=(KIND_TRIAL, KIND_RUN)) -> float | None:
        """Median env-steps/sec this computer achieved on runs with the same
        game, observation type and number of emulators. None if unknown."""
        want = (str(cfg.get("game", "mario")), str(cfg.get("obs_type", "tiles")),
                min(int(cfg.get("n_envs", 0)), cpu_count()))   # the trainer clamps to the cores
        rates = []
        for r in reversed(self.records):
            if r.kind not in kinds or r.fps is None or r.steps_done < 2000:
                continue
            got = (r.game, str(r.config.get("obs_type")), int(r.config.get("n_envs", 0)))
            if got == want:
                rates.append(r.fps)
            if len(rates) >= 8:
                break
        return statistics.median(rates) if rates else None

    def summary(self, game: str | None = None) -> dict:
        recs = self.for_game(game)
        return {"trials": sum(1 for r in recs if r.kind == KIND_TRIAL),
                "runs": sum(1 for r in recs if r.kind == KIND_RUN),
                "incomplete": sum(1 for r in recs if not r.complete),
                "compute_hours": sum(r.duration or 0.0 for r in recs) / 3600.0}

    def effects(self, cfg: dict, metric: str = tuning.METRIC_LATE) -> dict[str, list[tuple]]:
        """For the task of `cfg`: per hyperparameter, the mean score of every
        value tried, best first: {param: [(value, mean_score, n), …]}."""
        trials = self.trials_of_task(cfg) or self.trials_of_task(cfg, any_length=True)
        out: dict[str, list[tuple]] = {}
        for key in HYPER_FIELDS:
            by_value: dict[float, list[float]] = {}
            for r in trials:
                s = r.score(metric)
                if math.isnan(s):
                    continue
                by_value.setdefault(_canon(r.config.get(key)), []).append(s)
            if len(by_value) > 1:
                rows = [(v, statistics.fmean(ss), len(ss)) for v, ss in by_value.items()]
                out[key] = sorted(rows, key=lambda t: -t[1])
        return out

    # ---------- suggestions ----------

    def suggest(self, base: dict, trial_steps: int, n_seeds: int, n: int = 6,
                metric: str = tuning.METRIC_LATE,
                incumbent: "Knowledge | None" = None) -> list[dict]:
        """Up to `n` untested variations (override dicts relative to `base`)
        around the best known settings of the base preset's task, or around
        the base itself when nothing is known yet. Single-parameter moves
        first, in order of how much each parameter usually matters; two-step
        moves fill up if the near neighbourhood is exhausted."""
        centre_cfg = incumbent.config if incumbent is not None else tuning.full_config(base, {})
        centre = tuning.config_diff(centre_cfg, tuning.full_config(base, {}), HYPER_FIELDS)
        seen: set[str] = set()
        out: list[dict] = []

        def consider(overrides: dict) -> None:
            if len(out) >= n:
                return
            key = json.dumps(overrides, sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            if not overrides:
                return                                  # the baseline is always a candidate anyway
            if self.known_seeds(base, overrides, trial_steps, n_seeds) >= n_seeds:
                return
            out.append(overrides)

        for distance in (1, 2):
            for param, new_value in neighbour_values(centre_cfg, distance):
                consider(_merge(centre, {param: new_value}, base))
                if len(out) >= n:
                    return out
        return out

    def auto_plan(self, base: dict, trial_steps: int, n_seeds: int, n: int = 6,
                  metric: str = tuning.METRIC_LATE) -> tuple[list[dict], str]:
        """The wizard's "Let AIboy choose": the standard first search while
        most of it is still unknown, then untested variations around the best
        known settings. Returns (candidate overrides, plain explanation)."""
        grid = tuning.expand_grid(tuning.SWEEP_TEMPLATES[tuning.DEFAULT_TEMPLATE])
        known = [c for c in grid if self.known_seeds(base, c, trial_steps, n_seeds) >= n_seeds]
        if len(known) < (len(grid) + 1) // 2:
            if known:
                why = (f"{len(known)} of the {len(grid)} standard variations are already known and "
                       f"will not be trained again; the other {len(grid) - len(known)} will be.")
            else:
                why = ("Nothing is known about this goal yet, so AIboy starts with the standard "
                       "search: curiosity × learning speed, the two that matter most.")
            return grid, why
        best = self.best_for_task(tuning.trial_config(base, {}, trial_steps), metric)
        candidates = self.suggest(base, trial_steps, n_seeds, n, metric, best)
        if best is not None:
            incumbent = tuning.config_diff(best.config, tuning.full_config(base, {}), HYPER_FIELDS)
            if incumbent:
                candidates = [incumbent] + [c for c in candidates if c != incumbent]
            around = tuning.plain_overrides(incumbent)
            why = (f"AIboy knows {best.n_trials} earlier tests of this goal. Best so far: {around} "
                   f"(score {best.score:.0f}). It now tries {len(candidates) - bool(incumbent)} "
                   f"untested changes around those settings.")
        else:
            why = (f"AIboy knows {len(known)} standard variations of this goal and now tries "
                   f"{len(candidates)} untested changes.")
        if not candidates:
            why = ("Every nearby variation of this goal has been tested; only the known "
                   "results will be compared.")
        return candidates, why


@dataclass
class Knowledge:
    """What AIboy knows about the best settings for one task."""
    config: dict
    score: float
    n_seeds: int
    n_trials: int
    trial_steps: int
    when: float


# ---------- neighbourhood of a configuration ----------

# How each hyperparameter is stepped when looking for a better neighbour:
# ("log", factor, lo, hi) multiplies / divides; ("ladder", values) moves to
# the adjacent entry. Ordered by how much the parameter usually matters.
MOVES: dict[str, tuple] = {
    "learning_rate": ("log", 2.0, 1e-5, 3e-3),
    "ent_coef": ("log", 2.0, 0.0025, 0.2),
    "n_steps": ("log", 2.0, 128, 2048),
    "n_epochs": ("ladder", [2, 3, 4, 6, 8, 10]),
    "batch_size": ("log", 2.0, 64, 1024),
    "gamma": ("ladder", [0.98, 0.99, 0.995, 0.999]),
    "clip_range": ("ladder", [0.1, 0.2, 0.3]),
    "gae_lambda": ("ladder", [0.9, 0.95, 0.98]),
}


def _round_sig(x: float, digits: int = 2) -> float:
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def neighbour_values(cfg: dict, distance: int = 1) -> list[tuple[str, object]]:
    """(param, value) moves `distance` steps up and down from `cfg`, parameter
    by parameter in order of importance, up before down. Values are rounded
    to two significant digits so they read well and compare exactly."""
    out: list[tuple[str, object]] = []
    for param, move in MOVES.items():
        current = cfg.get(param, DEFAULT_CONFIG.get(param))
        if move[0] == "log":
            _, factor, lo, hi = move
            cur = float(current)
            if cur <= 0:
                cur = lo / factor                           # e.g. ent_coef 0 -> the smallest step
            for direction in (1, -1):
                val = cur * factor ** (direction * distance)
                if not (lo <= val <= hi):
                    continue
                if param in ("n_steps", "batch_size"):
                    val = int(round(val))               # powers of two stay exact
                else:
                    val = _round_sig(val)
                if val != current:
                    out.append((param, val))
        else:
            ladder = move[1]
            nearest = min(range(len(ladder)), key=lambda i: abs(float(ladder[i]) - float(current)))
            for direction in (1, -1):
                i = nearest + direction * distance
                if 0 <= i < len(ladder) and ladder[i] != current:
                    out.append((param, ladder[i]))
    return out


def _merge(centre: dict, move: dict, base: dict) -> dict:
    """Overrides for a candidate: the centre's deviations from the base plus
    one move; a move back onto the base's own value drops that key."""
    out = {**centre, **move}
    base_full = tuning.full_config(base, {})
    return {k: v for k, v in out.items() if not tuning._same_value(v, base_full.get(k))}


def describe_record(record: Record) -> str:
    """One line for logs and status texts."""
    m = record.metrics()
    score = f"{m.late:.0f}" if m is not None else "—"
    return (f"{record.kind} {record.run_name or record.id} · {tuning.compact_config(record.config)} "
            f"· {tuning.format_steps(record.steps_done)} steps · late {score}")

