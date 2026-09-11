import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tuning


class SweepTests(unittest.TestCase):
    def test_validate_rejects_bad_input(self):
        self.assertIsNotNone(tuning.validate_sweep("not json")[1])
        self.assertIsNotNone(tuning.validate_sweep("[]")[1])
        self.assertIsNotNone(tuning.validate_sweep('{"timesteps": [1]}')[1])
        self.assertIsNotNone(tuning.validate_sweep('{"ent_coef": []}')[1])
        self.assertIsNotNone(tuning.validate_sweep('{"ent_coef": [0.1, 0.1]}')[1])

    def test_templates_are_valid(self):
        for name, sweep in tuning.SWEEP_TEMPLATES.items():
            parsed, err = tuning.validate_sweep(json.dumps(sweep))
            self.assertIsNone(err, name)
            self.assertEqual(parsed, sweep)
            self.assertIn(name, tuning.TEMPLATE_NOTES)
        self.assertIn(tuning.DEFAULT_TEMPLATE, tuning.SWEEP_TEMPLATES)

    def test_grid_expansion(self):
        sweep = {"a": [1, 2, 3], "b": ["x", "y"]}
        grid = tuning.expand_grid(sweep)
        self.assertEqual(len(grid), tuning.n_grid_configs(sweep))
        self.assertEqual(grid[0], {"a": 1, "b": "x"})
        self.assertEqual(len({json.dumps(g, sort_keys=True) for g in grid}), 6)

    def test_random_sample_is_deterministic_for_same_sweep(self):
        sweep = {"a": list(range(10)), "b": list(range(10))}
        first = tuning.sample_random(sweep, 7)
        second = tuning.sample_random(sweep, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(len({json.dumps(g, sort_keys=True) for g in first}), 7)
        # Key order in the JSON must not change the sample.
        reordered = {"b": list(range(10)), "a": list(range(10))}
        self.assertEqual(tuning.sweep_seed(sweep), tuning.sweep_seed(reordered))
        # Whole grid when it is small enough.
        self.assertEqual(len(tuning.sample_random({"a": [1, 2]}, 5)), 2)

    def test_full_config_and_run_names(self):
        base = {"game": "mario", "ent_coef": 0.01}
        cfg = tuning.full_config(base, {"ent_coef": 0.05})
        self.assertEqual(cfg["ent_coef"], 0.05)
        self.assertEqual(cfg["gamma"], 0.99)
        self.assertEqual(tuning.trial_run_name("tune", 3, 0, 1), "tune-003")
        self.assertEqual(tuning.trial_run_name("tune", 3, 1, 2), "tune-003-s1")

    def test_progress_helpers(self):
        self.assertEqual(tuning.format_steps(4000), "4,000")
        self.assertEqual(tuning.format_steps(250_000), "250k")
        self.assertEqual(tuning.format_steps(1_250_000), "1.25M")
        self.assertIsNone(tuning.eta_seconds(0, 100, 10))          # nothing done yet
        self.assertIsNone(tuning.eta_seconds(50, 100, 1))          # too early to say
        self.assertAlmostEqual(tuning.eta_seconds(25, 100, 60), 180)
        self.assertEqual(tuning.eta_seconds(100, 100, 60), 0.0)
        self.assertEqual(tuning.progress_text(0, 2_000_000, 0.0), "0 / 2.00M · 0%")
        self.assertEqual(tuning.progress_text(500_000, 2_000_000, 120.0),
                         "500k / 2.00M · 25% · ETA 6 min")
        self.assertTrue(tuning.progress_text(2_000_000, 2_000_000, 300).endswith("ETA done"))

    def test_format_duration(self):
        self.assertEqual(tuning.format_duration(30), "30 s")
        self.assertEqual(tuning.format_duration(600), "10 min")
        self.assertEqual(tuning.format_duration(7200), "2.0 h")


class ResultTests(unittest.TestCase):
    def _write_evals(self, run_dir: Path, timesteps, results):
        (run_dir / "logs").mkdir(parents=True)
        np.savez(run_dir / "logs" / "evaluations.npz",
                 timesteps=np.array(timesteps),
                 results=np.array(results),
                 ep_lengths=np.ones_like(np.array(results)) * 100)

    def test_read_trial_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self._write_evals(run, [1000, 2000, 3000], [[10, 20], [50, 70], [30, 30]])
            m = tuning.read_trial_metrics(run / "logs" / "evaluations.npz")
            self.assertEqual(m.best, 60)
            self.assertEqual(m.final, 30)
            self.assertEqual(m.timesteps, 3000)
            self.assertEqual(m.n_evals, 3)
            self.assertEqual(m.by_name("best eval reward"), 60)
            self.assertIsNone(tuning.read_trial_metrics(run / "missing.npz"))

    def test_trial_reuse_requires_matching_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "tune-001"
            self._write_evals(run, [9500], [[1.0]])
            cfg = {"ent_coef": 0.01, "learning_rate": 3e-4}
            self.assertTrue(tuning.trial_is_complete(run, 10_000))          # no manifest: accept
            self.assertFalse(tuning.trial_is_complete(run, 20_000))         # too short
            tuning.write_trial_manifest(run, cfg, 10_000)
            self.assertTrue(tuning.trial_is_complete(run, 10_000, dict(cfg)))
            self.assertTrue(tuning.trial_is_complete(run, 10_000, {**cfg, "ent_coef": 0.010}))
            self.assertFalse(tuning.trial_is_complete(run, 10_000, {**cfg, "ent_coef": 0.05}))

    def test_results_roundtrip_and_ranking(self):
        a = tuning.ConfigResult(index=1, overrides={"ent_coef": 0.01}, config={}, metric="m",
                                values=[10.0, 20.0], ep_lens=[100.0], runs=["r1"])
        b = tuning.ConfigResult(index=2, overrides={"ent_coef": 0.05}, config={}, metric="m",
                                values=[40.0], ep_lens=[80.0], runs=["r2"])
        c = tuning.ConfigResult(index=3, overrides={}, config={}, metric="m")
        self.assertEqual(a.score, 15.0)
        self.assertEqual(a.score_text(), "15 ± 5")
        self.assertEqual(c.score_text(), "—")
        self.assertIs(tuning.best_result([a, b, c]), b)
        self.assertEqual([r.index for r in tuning.rank_results([c, a, b])], [2, 1, 3])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            tuning.save_results(path, {"metric": "m"}, [a, b])
            meta, loaded = tuning.load_results(path)
            self.assertEqual(meta["metric"], "m")
            self.assertEqual([r.index for r in loaded], [1, 2])
            self.assertEqual(loaded[0].values, [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
