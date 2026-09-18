import json
import tempfile
import time
import unittest
from pathlib import Path

from aiboy import experience, presets, tuning
from aiboy.games import ENV_VERSION

BASE = presets.normalize({"game": "mario", "start_level": "default", "timesteps": 2_000_000})


def evals_for(score: float, steps: int = 100_000):
    """A rising eval history ending at `score` (late metric ≈ score)."""
    return [[steps * 0.25, score * 0.2, 90.0], [steps * 0.5, score * 0.6, 100.0],
            [steps * 0.75, score, 110.0], [steps, score, 120.0]]


def trial(overrides: dict, score: float, seed: int = 0, steps: int = 100_000, **kw) -> experience.Record:
    cfg = tuning.trial_config(BASE, overrides, steps, seed)
    return experience.make_record(cfg, evals_for(score, steps), duration=steps / 2000.0,
                                  completed=True, source=kw.pop("source", "wizard"),
                                  run_name=kw.pop("run_name", f"t-{seed}"), **kw)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "experience.jsonl"
        self.exp = experience.Experience(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_records_survive_a_round_trip_and_bad_lines(self):
        self.assertEqual(self.exp.records, [])
        rec = trial({"ent_coef": 0.03}, 500.0)
        self.exp.add(rec)
        with self.path.open("a") as f:
            f.write("{ torn line\n\n")
        again = experience.Experience(self.path)
        self.assertEqual([r.id for r in again.records], [rec.id])
        loaded = again.records[0]
        self.assertEqual(loaded.config["ent_coef"], 0.03)
        self.assertEqual(loaded.kind, experience.KIND_TRIAL)
        self.assertTrue(loaded.complete and loaded.usable())
        self.assertAlmostEqual(loaded.score(), 500.0)
        self.assertAlmostEqual(loaded.fps, 2000.0)
        self.assertEqual(set(loaded.config), set(presets.PRESET_FIELDS))
        # minimal / old lines still load
        self.path.write_text(json.dumps({"config": {"game": "mario"}}) + "\n")
        old = experience.load_records(self.path)[0]
        self.assertEqual((old.kind, old.evals, old.complete), (experience.KIND_RUN, [], False))
        self.assertFalse(old.usable())

    def test_signature_and_task_identity(self):
        a = tuning.trial_config(BASE, {"ent_coef": 0.03}, 100_000)
        b = {**a, "checkpoint_freq": 1, "device": "mps", "eval_freq": 7}
        self.assertEqual(experience.signature_of(a), experience.signature_of(b))
        self.assertNotEqual(experience.signature_of(a), experience.signature_of({**a, "seed": 1}))
        self.assertNotEqual(experience.signature_of(a), experience.signature_of({**a, "ent_coef": 0.01}))
        self.assertEqual(experience.task_of(a), experience.task_of({**a, "ent_coef": 0.01, "seed": 3}))
        self.assertNotEqual(experience.task_of(a), experience.task_of({**a, "start_level": "marathon"}))
        self.assertNotEqual(experience.task_of(a), experience.task_of({**a, "timesteps": 50_000}))
        self.assertIn(ENV_VERSION, experience.signature_of(a))
        # ints and floats compare by value
        self.assertEqual(experience.signature_of({**a, "n_steps": 512}),
                         experience.signature_of({**a, "n_steps": 512.0}))

    def test_find_known_and_reuse_rules(self):
        self.exp.add(trial({"ent_coef": 0.03}, 500.0, seed=0))
        self.exp.add(trial({"ent_coef": 0.03}, 520.0, seed=1))
        wanted = tuning.trial_config(BASE, {"ent_coef": 0.03}, 100_000, 1)
        self.assertAlmostEqual(self.exp.find(wanted).score(), 520.0)
        self.assertIsNone(self.exp.find(tuning.trial_config(BASE, {"ent_coef": 0.03}, 100_000, 2)))
        self.assertIsNone(self.exp.find(tuning.trial_config(BASE, {"ent_coef": 0.03}, 50_000, 0)))
        self.assertEqual(self.exp.known_seeds(BASE, {"ent_coef": 0.03}, 100_000, 3), 2)
        self.assertEqual(self.exp.known_seeds(BASE, {}, 100_000, 1), 0)
        # incomplete, interrupted or resumed runs are remembered but never reused
        cfg = tuning.trial_config(BASE, {"ent_coef": 0.05}, 100_000)
        partial = experience.make_record(cfg, evals_for(900.0)[:2], 10.0, completed=False,
                                         source="wizard", run_name="x")
        self.assertFalse(partial.complete)
        self.exp.add(partial)
        self.assertIsNone(self.exp.find(cfg))
        resumed = experience.make_record(cfg, evals_for(900.0), 10.0, completed=True, source="train",
                                         run_name="x", resumed=True)
        self.exp.add(resumed)
        self.assertIsNone(self.exp.find(cfg))
        stale = trial({"ent_coef": 0.07}, 999.0)
        stale.env_version = "mario-0"
        self.exp.add(stale)
        self.assertIsNone(self.exp.find(stale.config))
        # the newest usable record wins when an experiment was repeated
        self.exp.add(trial({"ent_coef": 0.03}, 530.0, seed=1))
        self.assertAlmostEqual(self.exp.find(wanted).score(), 530.0)

    def test_best_for_task_prefers_same_length_and_averages_seeds(self):
        self.assertIsNone(self.exp.best_for_task(tuning.trial_config(BASE, {}, 100_000)))
        self.exp.add(trial({}, 400.0, seed=0))
        self.exp.add(trial({"ent_coef": 0.03}, 600.0, seed=0))
        self.exp.add(trial({"ent_coef": 0.03}, 500.0, seed=1))
        self.exp.add(trial({"ent_coef": 0.05}, 580.0, seed=0))
        self.exp.add(trial({"ent_coef": 0.09}, 9000.0, seed=0, steps=50_000))   # another length
        know = self.exp.best_for_task(tuning.trial_config(BASE, {}, 100_000))
        self.assertEqual(know.config["ent_coef"], 0.05)          # 580 beats the 550 average
        self.assertEqual((know.n_seeds, know.n_trials, know.trial_steps), (1, 4, 100_000))
        other = self.exp.best_for_task(tuning.trial_config(BASE, {}, 500_000))
        self.assertEqual(other.trial_steps, 100_000)              # falls back to the longest known
        marathon = self.exp.best_for_task(tuning.trial_config({**BASE, "start_level": "marathon"}, {}, 100_000))
        self.assertIsNone(marathon)
        effects = self.exp.effects(tuning.trial_config(BASE, {}, 100_000))
        self.assertEqual([v for v, _s, _n in effects["ent_coef"]], [0.05, 0.03, 0.01])
        self.assertNotIn("learning_rate", effects)                # only one value tried

    def test_fps_is_measured_per_setup(self):
        cfg = tuning.trial_config(BASE, {}, 100_000)
        self.assertIsNone(self.exp.fps(cfg))
        for i in range(3):
            self.exp.add(trial({}, 1.0, seed=i))                   # 2000 steps/s
        self.assertAlmostEqual(self.exp.fps(cfg), 2000.0)
        self.assertIsNone(self.exp.fps({**cfg, "n_envs": 2}))
        self.assertIsNone(self.exp.fps({**cfg, "obs_type": "pixels"}))
        summary = self.exp.summary("mario")
        self.assertEqual((summary["trials"], summary["runs"]), (3, 0))
        self.assertGreater(summary["compute_hours"], 0)

    def test_neighbours_and_suggestions_skip_known(self):
        centre = tuning.full_config(BASE, {})
        moves = experience.neighbour_values(centre)
        self.assertEqual(moves[:2], [("learning_rate", 0.0005), ("learning_rate", 0.00013)])
        self.assertIn(("ent_coef", 0.02), moves)
        self.assertIn(("ent_coef", 0.005), moves)
        self.assertIn(("n_steps", 1024), moves)
        self.assertIn(("n_epochs", 6), moves)
        self.assertIn(("gamma", 0.995), moves)
        far = experience.neighbour_values(centre, 2)
        self.assertEqual(far[0], ("learning_rate", 0.001))
        zero_ent = experience.neighbour_values({**centre, "ent_coef": 0.0})
        self.assertIn(("ent_coef", 0.0025), zero_ent)               # 0 steps to the smallest value
        sug = self.exp.suggest(BASE, 100_000, 1, n=4)
        self.assertEqual(sug, [{"learning_rate": 0.0005}, {"learning_rate": 0.00013},
                               {"ent_coef": 0.02}, {"ent_coef": 0.005}])
        self.exp.add(trial({"learning_rate": 0.0005}, 10.0))
        sug = self.exp.suggest(BASE, 100_000, 1, n=4)
        self.assertNotIn({"learning_rate": 0.0005}, sug)            # already known
        self.assertEqual(len(sug), 4)
        # around an incumbent that differs from the base, moves keep its other deviations
        know = experience.Knowledge(config={**centre, "ent_coef": 0.03, "learning_rate": 0.0005},
                                    score=1.0, n_seeds=1, n_trials=1, trial_steps=100_000,
                                    when=time.time())
        sug = self.exp.suggest(BASE, 100_000, 1, n=2, incumbent=know)
        # a move back onto the base value drops the key (0.00025 is the base lr)
        self.assertEqual(sug, [{"ent_coef": 0.03, "learning_rate": 0.001}, {"ent_coef": 0.03}])

    def test_auto_plan_starts_standard_then_explores(self):
        grid = tuning.expand_grid(tuning.SWEEP_TEMPLATES[tuning.DEFAULT_TEMPLATE])
        combos, why = self.exp.auto_plan(BASE, 100_000, 1)
        self.assertEqual(combos, grid)
        self.assertIn("Nothing is known", why)
        for i, c in enumerate(grid[:2]):
            self.exp.add(trial(c, 100.0 + i))
        combos, why = self.exp.auto_plan(BASE, 100_000, 1)
        self.assertEqual(combos, grid)
        self.assertIn("2 of the 6", why)
        for i, c in enumerate(grid):
            self.exp.add(trial(c, 100.0 + i))                       # the last one scores best
        self.exp.add(trial({}, 50.0))
        combos, why = self.exp.auto_plan(BASE, 100_000, 1, n=4)
        self.assertEqual(combos[0], grid[-1])                       # incumbent first
        self.assertEqual(len(combos), 5)
        for c in combos[1:]:
            self.assertNotIn(c, grid)
            self.assertEqual(self.exp.known_seeds(BASE, c, 100_000, 1), 0)
        self.assertIn("Best so far", why)
        # two seeds wanted, only one known: still counts as untested
        combos2, _ = self.exp.auto_plan(BASE, 100_000, 2)
        self.assertEqual(combos2, grid)

    def test_related_knowledge_is_borrowed_only_while_the_task_is_unknown(self):
        marathon = {**BASE, "start_level": "marathon"}
        task = tuning.trial_config(marathon, {}, 100_000)
        self.assertIsNone(self.exp.knowledge_for(task))
        self.exp.add(trial({}, 400.0))                              # campaign tests
        self.exp.add(trial({"ent_coef": 0.03}, 600.0))
        know = self.exp.knowledge_for(task)
        self.assertEqual(know.borrowed_from, "campaign")
        self.assertEqual(know.config["ent_coef"], 0.03)
        self.assertIsNone(self.exp.best_for_task(task))             # the task itself: still unknown
        # a pixels preset is not related (another network)
        self.assertIsNone(self.exp.knowledge_for({**task, "obs_type": "pixels"}))
        # the first auto plan tries the borrowed settings before the standard grid
        combos, why = self.exp.auto_plan(marathon, 100_000, 1)
        grid = tuning.expand_grid(tuning.SWEEP_TEMPLATES[tuning.DEFAULT_TEMPLATE])
        self.assertEqual(combos[0], {"ent_coef": 0.03})
        self.assertEqual(combos[1:], grid)
        self.assertIn("worked best for the campaign", why)
        # borrowed settings that are part of the grid anyway are not repeated
        self.exp.add(trial({"ent_coef": 0.03, "learning_rate": 1e-4}, 900.0))
        combos, _ = self.exp.auto_plan(marathon, 100_000, 1)
        self.assertEqual(combos, grid)
        # once the task has its own tests nothing is borrowed
        cfg = tuning.trial_config(marathon, {}, 100_000, 0)
        self.exp.add(experience.make_record(cfg, evals_for(50.0), 10.0, completed=True,
                                            source="wizard", run_name="m-0"))
        self.assertIsNone(self.exp.knowledge_for(task).borrowed_from)

    def test_improvements_follow_the_winner_rule_and_are_recorded(self):
        presets_by_name = {"goal": dict(BASE), "long goal": {**BASE, "timesteps": 8_000_000},
                           "marathon goal": {**BASE, "start_level": "marathon"}}
        self.assertEqual(self.exp.improvements(presets_by_name), [])
        self.exp.add(trial({"ent_coef": 0.03}, 600.0))
        self.assertEqual(self.exp.improvements(presets_by_name), [])   # own settings untested
        self.exp.add(trial({}, 590.0))
        self.assertEqual(self.exp.improvements(presets_by_name), [])   # within the 5 % margin
        self.exp.add(trial({}, 400.0, seed=1))                          # own: mean 495, spread 95
        self.exp.add(trial({"ent_coef": 0.03}, 620.0, seed=1))          # best: mean 610, spread 10
        found = self.exp.improvements(presets_by_name)
        self.assertEqual([i.preset for i in found], ["goal", "long goal"])   # same task, any length
        imp = found[0]
        self.assertEqual(imp.overrides, {"ent_coef": 0.03})
        self.assertEqual(imp.config["ent_coef"], 0.03)
        self.assertEqual(imp.config["timesteps"], BASE["timesteps"])
        self.assertEqual((round(imp.score), round(imp.baseline_score), imp.n_trials), (610, 495, 4))
        self.assertIn("score 610 vs 495", imp.summary())
        # the spread rule: a wide spread of the challenger blocks the change
        self.exp.add(trial({"ent_coef": 0.03}, 300.0, seed=2))
        self.assertEqual(self.exp.improvements(presets_by_name), [])
        self.exp.forget([self.exp.records[-1].id])
        # applying it: recorded, visible in the history, and no longer an improvement
        rec = self.exp.record_improvement(imp, presets_by_name["goal"])
        self.assertEqual(rec.kind, experience.KIND_IMPROVEMENT)
        self.assertFalse(rec.usable())
        self.assertIn("ent 0.03 (was ent 0.01)", rec.note)
        again = experience.Experience(self.path)
        self.assertEqual([r.id for r in again.improvement_history("goal")], [rec.id])
        self.assertEqual(again.summary("mario")["improvements"], 1)
        self.assertIsNone(again.latest_for_run("goal"))               # never mistaken for a trial
        improved = {"goal": imp.config}
        self.assertEqual(again.improvements(improved), [])
        self.assertEqual(again.best_for_task(tuning.trial_config(BASE, {}, 100_000)).n_trials, 4)

    def test_forget_and_clear(self):
        a, b = trial({}, 1.0, seed=0), trial({}, 2.0, seed=1)
        self.exp.add(a); self.exp.add(b)
        self.assertEqual(self.exp.forget([a.id]), 1)
        self.assertEqual([r.id for r in experience.load_records(self.path)], [b.id])
        self.assertEqual(self.exp.forget(["nope"]), 0)
        self.assertEqual(self.exp.clear(), 1)
        self.assertEqual(experience.load_records(self.path), [])
        self.assertIn("late", experience.describe_record(b))


if __name__ == "__main__":
    unittest.main()
