import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiboy import tuning


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

    def test_mode_label(self):
        self.assertEqual(tuning.mode_label({"start_level": "default"}), "campaign")
        self.assertEqual(tuning.mode_label({}), "campaign")
        self.assertEqual(tuning.mode_label({"start_level": "sequential"}), "level by level")
        self.assertEqual(tuning.mode_label({"start_level": "3-2"}), "level 3-2")

    def test_progress_helpers(self):
        self.assertEqual(tuning.format_steps(4000), "4,000")
        self.assertEqual(tuning.format_steps(250_000), "250k")
        self.assertEqual(tuning.format_steps(1_250_000), "1.25M")
        self.assertEqual(tuning.progress_text(0, 2_000_000, None), "0 / 2.00M · 0%")
        self.assertEqual(tuning.progress_text(500_000, 2_000_000, 360.0),
                         "500k / 2.00M · 25% · ETA 6 min")
        self.assertTrue(tuning.progress_text(2_000_000, 2_000_000, 0.0).endswith("ETA done"))

    def test_rate_estimator_ignores_startup_and_follows_recent_rate(self):
        est = tuning.RateEstimator(window=5)
        self.assertIsNone(est.rate())
        est.add(100.0, 5000)                 # first report 100 s after launch: launch time irrelevant
        self.assertIsNone(est.rate())         # one sample is not a rate
        est.add(101.0, 10000)
        self.assertIsNone(est.rate())         # span below min_span
        est.add(102.0, 15000)
        self.assertAlmostEqual(est.rate(), 5000.0)
        self.assertAlmostEqual(est.eta(50_000), 10.0)
        for t in range(103, 110):             # throughput halves; window slides to the new rate
            est.add(float(t), 15000 + (t - 102) * 2500)
        self.assertAlmostEqual(est.rate(), 2500.0)
        est.add(200.0, 100)                   # counter reset (new run) clears the window
        self.assertIsNone(est.rate())

    def test_sweep_eta(self):
        eta = tuning.SweepEta(n_trials_to_train=3, trial_steps=1000)
        self.assertIsNone(eta.between_trials())
        eta.trial_started(0.0)
        self.assertIsNone(eta.report(1.0, 100))               # no rate yet
        # 100 steps/s -> current trial needs 7 s + tail; two more trials like this one
        e = eta.report(3.0, 300)
        self.assertIsNotNone(e)
        projected_trial = 3.0 + 7.0 + tuning.SweepEta.TRIAL_TAIL
        self.assertAlmostEqual(e, (7.0 + tuning.SweepEta.TRIAL_TAIL) + 2 * projected_trial)
        eta.trial_finished(12.0)
        self.assertAlmostEqual(eta.between_trials(), 2 * 12.0)  # measured duration drives the rest
        eta.trial_started(20.0)
        self.assertAlmostEqual(eta.report(21.0, 50), 11.0 + 1 * 12.0)   # avg minus elapsed, plus one more
        eta.trial_finished(32.0)
        eta.trial_started(40.0); eta.trial_finished(52.0)
        self.assertAlmostEqual(eta.between_trials(), 0.0)

    def test_plain_language_covers_every_template_and_effort(self):
        self.assertEqual(set(tuning.TEMPLATE_PLAIN), set(tuning.SWEEP_TEMPLATES))
        self.assertEqual(set(tuning.TEMPLATE_PLAIN_NOTES), set(tuning.SWEEP_TEMPLATES))
        self.assertEqual(len(tuning.TEMPLATE_FROM_PLAIN), len(tuning.SWEEP_TEMPLATES))  # names unique
        self.assertEqual(tuning.template_search(tuning.DEFAULT_TEMPLATE), ("grid", 0))
        self.assertEqual(tuning.template_search("Broad random search (use Random, ~12 trials)"),
                         ("random", 12))
        self.assertEqual(set(tuning.EFFORT_NOTES), set(tuning.EFFORT_LEVELS))
        self.assertEqual(tuning.effort_plan(2_000_000, "Normal"), (2, 100_000))
        self.assertEqual(tuning.effort_plan(2_000_000, "Quick"), (1, 50_000))
        self.assertEqual(tuning.effort_plan(60_000_000, "Thorough"), (3, 1_000_000))
        self.assertEqual(tuning.plain_overrides({"ent_coef": 0.03, "learning_rate": 3e-4}),
                         "curiosity 0.03 · learning speed 0.0003")
        self.assertEqual(tuning.plain_overrides({}), tuning.BASELINE_PLAIN)
        for key in tuning.TUNABLE_FIELDS:
            self.assertIn(key, tuning.PLAIN_KEYS, key)

    def test_explain_winner(self):
        mk = lambda i, ov, vals: tuning.ConfigResult(index=i, overrides=ov, config={}, metric="m",
                                                    values=vals, runs=["r"])
        base = mk(1, {}, [100.0, 120.0])
        close = mk(2, {"ent_coef": 0.03}, [115.0, 125.0])
        clear = mk(3, {"ent_coef": 0.05}, [200.0, 210.0])
        w, why = tuning.explain_winner([base, close])
        self.assertIs(w, base); self.assertIn("too close to call", why); self.assertNotIn("ent_coef", why)
        w, why = tuning.explain_winner([base, close, clear])
        self.assertIs(w, clear); self.assertIn("clearly better", why); self.assertIn("curiosity 0.05", why)
        w, why = tuning.explain_winner([base]); self.assertIs(w, base); self.assertIn("No variation", why)
        self.assertEqual(tuning.explain_winner([])[0], None)

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

    def test_per_minute_metric_and_eval_histories(self):
        m = tuning.TrialMetrics(best=600.0, final=1.0, mean=1.0, ep_len=1.0, timesteps=1,
                                n_evals=1, duration=120.0)
        self.assertAlmostEqual(m.by_name(tuning.METRIC_PER_MINUTE), 300.0)
        m.best = -50.0
        self.assertEqual(m.per_minute, 0.0)                       # failures earn nothing
        m.duration = None
        self.assertTrue(np.isnan(m.per_minute))
        self.assertEqual(set(tuning.METRIC_NOTES), set(tuning.METRICS))
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "t"
            self._write_evals(run, [50_000, 95_000], [[100.0], [600.0]])
            evals = tuning.evals_from_npz(run / "logs" / "evaluations.npz")
            self.assertEqual(evals, [[50_000, 100.0, 100.0], [95_000, 600.0, 100.0]])
            self.assertIsNone(tuning.evals_from_npz(run / "missing.npz"))
            got = tuning.metrics_from_evals(evals, duration=42.5)
            self.assertAlmostEqual(got.per_minute, 600.0 / (42.5 / 60))
            self.assertEqual(got.timesteps, 95_000)
            self.assertEqual(tuning.read_trial_metrics(run / "logs" / "evaluations.npz").best, 600.0)
        self.assertIsNone(tuning.metrics_from_evals([]))
        self.assertIsNone(tuning.record_metrics({"evals": []}))

    def test_prune_trial_runs_keeps_best(self):
        mk = lambda i, v, run: tuning.ConfigResult(index=i, overrides={}, config={}, metric="m",
                                                  values=[v] if v is not None else [], runs=[run])
        results = [mk(1, 10.0, "t-001"), mk(2, 50.0, "t-002"), mk(3, None, "t-003"), mk(4, 30.0, "t-004")]
        deleted = []
        gone = tuning.prune_trial_runs(results, 2, "mario", lambda g, r: deleted.append(r))
        self.assertEqual(sorted(gone), ["t-001", "t-003"])          # worst scored + unscored
        self.assertEqual(sorted(deleted), ["t-001", "t-003"])
        self.assertTrue(results[0].pruned and results[2].pruned)
        self.assertFalse(results[1].pruned or results[3].pruned)
        self.assertEqual(tuning.prune_trial_runs(results, 2, "mario", lambda g, r: deleted.append(r)), [])
        self.assertEqual(tuning.prune_trial_runs(results, 0, "mario", lambda g, r: deleted.append(r)), [])
        # a later, better result pushes an older one out
        results.append(mk(5, 99.0, "t-005"))
        self.assertEqual(tuning.prune_trial_runs(results, 2, "mario", lambda g, r: deleted.append(r)), ["t-004"])

    def test_late_metric_baseline_and_winner_rule(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self._write_evals(run, [1, 2, 3, 4, 5, 6], [[10], [20], [30], [40], [50], [60]])
            m = tuning.read_trial_metrics(run / "logs" / "evaluations.npz")
            self.assertAlmostEqual(m.late, 55.0)                    # mean of the last third (2 evals)
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self._write_evals(run, [1, 2, 3, 4, 5], [[10], [20], [30], [40], [60]])
            self.assertAlmostEqual(tuning.read_trial_metrics(run / "logs" / "evaluations.npz").late, 50.0)
            self.assertEqual(m.by_name(tuning.METRIC_LATE), m.late)
        self.assertEqual(tuning.with_baseline([{"a": 1}, {"a": 2}]), [{}, {"a": 1}, {"a": 2}])
        self.assertEqual(tuning.with_baseline([{}, {"a": 1}]), [{}, {"a": 1}])
        self.assertEqual(tuning.suggested_trial_steps(2_000_000), 100_000)
        self.assertEqual(tuning.suggested_trial_steps(10_000_000), 500_000)
        self.assertEqual(tuning.suggested_trial_steps(60_000_000), 1_000_000)
        mk = lambda i, ov, vals: tuning.ConfigResult(index=i, overrides=ov, config={}, metric="m",
                                                    values=vals, runs=["r"])
        base = mk(1, {}, [100.0, 120.0])
        self.assertEqual(base.label, tuning.BASELINE_LABEL)
        close = mk(2, {"ent_coef": 0.03}, [115.0, 125.0])       # leads by 10, spread 10 -> not clear
        clear = mk(3, {"ent_coef": 0.05}, [200.0, 210.0])
        w, why = tuning.pick_winner([base, close]); self.assertIs(w, base); self.assertIn("keeping", why)
        w, why = tuning.pick_winner([base, close, clear]); self.assertIs(w, clear); self.assertIn("beats", why)
        w, _ = tuning.pick_winner([close, clear]); self.assertIs(w, clear)   # no baseline: best wins
        w, why = tuning.pick_winner([base, mk(4, {"x": 1}, [50.0])]); self.assertIs(w, base)
        self.assertEqual(tuning.pick_winner([]), (None, "no scored candidate"))
        # single seed: a lead below MIN_WIN_MARGIN of the baseline is noise
        one = mk(1, {}, [1000.0]); near = mk(2, {"ent_coef": 0.03}, [1040.0]); far = mk(3, {"ent_coef": 0.05}, [1100.0])
        self.assertIs(tuning.pick_winner([one, near])[0], one)
        self.assertIs(tuning.pick_winner([one, near, far])[0], far)

    def test_nan_scores_rank_last(self):
        a = tuning.ConfigResult(index=1, overrides={}, config={}, metric="m", values=[float("nan")],
                                runs=["r"])
        b = tuning.ConfigResult(index=2, overrides={}, config={}, metric="m", values=[5.0], runs=["r"])
        self.assertEqual([r.index for r in tuning.rank_results([a, b])], [2, 1])
        self.assertIs(tuning.best_result([a, b]), b)
        self.assertEqual(a.score_text(), "—")
        self.assertEqual(b.score_text(), "5.0")

    def test_config_result_scores_from_records(self):
        rec = {"evals": [[1000, 10.0, 50.0], [2000, 30.0, 60.0], [3000, 20.0, 70.0]], "duration": 60.0}
        r = tuning.ConfigResult(index=1, overrides={}, config={}, metric=tuning.METRIC_LATE)
        r.add_record(rec)
        self.assertEqual(r.values, [25.0])                       # last third, at least 2 evals
        self.assertEqual(r.ep_lens, [60.0])                      # ep len at the best eval
        self.assertEqual(r.timesteps, 3000)
        r.rescore(tuning.METRIC_BEST)
        self.assertEqual((r.metric, r.values), (tuning.METRIC_BEST, [30.0]))
        r.rescore(tuning.METRIC_PER_MINUTE)
        self.assertEqual(r.values, [30.0])
        r.add_record({"evals": []})                               # unscored record: ignored
        self.assertEqual(len(r.records), 2)
        self.assertEqual(len(r.values), 1)

    def test_trial_config_and_compact_texts(self):
        base = {"game": "mario", "seed": 5, "ent_coef": 0.01}
        cfg = tuning.trial_config(base, {"ent_coef": 0.03}, 100_000, seed_offset=1)
        self.assertEqual((cfg["seed"], cfg["timesteps"], cfg["ent_coef"]), (6, 100_000, 0.03))
        self.assertEqual(set(cfg), set(tuning.PRESET_FIELDS))
        self.assertEqual(tuning.compact_overrides({"ent_coef": 0.03, "learning_rate": 3e-4}),
                         "ent 0.03 · lr 0.0003")
        self.assertTrue(tuning.compact_config(cfg).startswith("lr 0.00025 · ent 0.03 · steps 512"))
        self.assertEqual(tuning.config_diff(cfg, tuning.full_config(base, {})), {"ent_coef": 0.03})
        self.assertEqual(tuning.config_diff({"ent_coef": 0.010}, {"ent_coef": 0.01}), {})

    def test_explicit_candidate_lists(self):
        text = '[{"ent_coef": 0.02}, {"learning_rate": 0.0005, "n_epochs": 6}]'
        sweep, err = tuning.validate_sweep(text)
        self.assertIsNone(err)
        self.assertEqual(tuning.expand_grid(sweep), [{"ent_coef": 0.02},
                                                     {"learning_rate": 0.0005, "n_epochs": 6}])
        self.assertEqual(tuning.n_grid_configs(sweep), 2)
        self.assertEqual(len(tuning.sample_random(sweep, 1)), 1)
        self.assertEqual(tuning.sample_random(sweep, 5), tuning.expand_grid(sweep))
        self.assertIsNotNone(tuning.validate_sweep('[{"timesteps": 1}]')[1])
        self.assertIsNotNone(tuning.validate_sweep('[]')[1])
        self.assertIsNotNone(tuning.validate_sweep('[{"ent_coef": 0.02}, {"ent_coef": 0.02}]')[1])
        self.assertIsNotNone(tuning.validate_sweep('[1]')[1])

    def test_estimates_use_measured_throughput(self):
        cfg = {"n_envs": 10, "obs_type": "tiles"}
        self.assertEqual(tuning.estimate_fps(cfg), 3300.0)
        self.assertEqual(tuning.estimate_fps(cfg, 1234.0), 1234.0)
        self.assertEqual(tuning.estimate_fps(cfg, 0.0), 3300.0)
        self.assertAlmostEqual(tuning.estimate_seconds(2, 1000, cfg, 100.0),
                               2 * (10.0 + tuning.TRIAL_OVERHEAD))

    def test_results_roundtrip_and_ranking(self):
        a = tuning.ConfigResult(index=1, overrides={"ent_coef": 0.01}, config={}, metric="m",
                                values=[10.0, 20.0], ep_lens=[100.0], runs=["r1"])
        b = tuning.ConfigResult(index=2, overrides={"ent_coef": 0.05}, config={}, metric="m",
                                values=[40.0], ep_lens=[80.0], runs=["r2"])
        c = tuning.ConfigResult(index=3, overrides={}, config={}, metric="m")
        self.assertEqual(a.score, 15.0)
        self.assertEqual(a.score_text(), "15.0 ± 5.0")
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
