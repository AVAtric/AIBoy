import sys
import tempfile
import unittest
from pathlib import Path

from aiboy import runs


class RunDiscoveryTests(unittest.TestCase):
    def _make_run(self, root: Path, game: str, name: str, *, best=False, final=False, snaps=()):
        p = runs.run_paths(game, name, root)
        for d in ("checkpoints", "logs"):
            p[d].mkdir(parents=True, exist_ok=True)
        if best:
            (p["logs"] / "best_model.zip").write_bytes(b"x")
        if final:
            (p["checkpoints"] / "final.zip").write_bytes(b"x")
        for s in snaps:
            (p["checkpoints"] / f"ppo_{s}_steps.zip").write_bytes(b"x")
        return p

    def test_list_runs_hides_internal_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_run(root, "mario", "alpha")
            self._make_run(root, "mario", "_level_states")
            (root / "mario" / "_tune").mkdir()
            self.assertEqual(runs.list_runs("mario", root), ["alpha"])
            self.assertEqual(runs.list_runs("kirby", root), [])
            self.assertFalse(runs.is_run_name("_x"))
            self.assertFalse(runs.is_run_name(""))

    def test_best_model_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(runs.best_model_for_run("mario", "none", root))
            p = self._make_run(root, "mario", "r", snaps=(25000, 100000, 5000))
            self.assertEqual(runs.best_model_for_run("mario", "r", root).name, "ppo_100000_steps.zip")
            (p["checkpoints"] / "final.zip").write_bytes(b"x")
            self.assertEqual(runs.best_model_for_run("mario", "r", root).name, "final.zip")
            (p["logs"] / "best_model.zip").write_bytes(b"x")
            self.assertEqual(runs.best_model_for_run("mario", "r", root).name, "best_model.zip")
            with self.assertRaises(FileNotFoundError):
                runs.resolve_model_path(None, "mario", "none", root)
            with self.assertRaises(FileNotFoundError):
                runs.resolve_model_path(str(root / "nope.zip"), "mario", "r", root)

    def test_list_models_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_run(root, "mario", "a", best=True, final=True, snaps=(25000,))
            self._make_run(root, "mario", "_level_states", best=True)
            labels = [lbl for lbl, _ in runs.list_models("mario", root)]
            self.assertEqual(labels, ["mario/a — best", "mario/a — final", "mario/a — 25,000 steps"])
            self.assertEqual(runs.list_models("mario", root / "missing"), [])

    def test_run_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(runs.read_run_config("mario", "r", root))
            runs.write_run_config("mario", "r", {"obs_type": "tiles", "frame_stack": 4,
                                                 "junk": 1, "seed": 0}, root)
            cfg = runs.read_run_config("mario", "r", root)
            self.assertEqual(cfg["frame_stack"], 4)
            self.assertEqual(cfg["game"], "mario")
            self.assertNotIn("junk", cfg)
            model = runs.run_paths("mario", "r", root)["logs"] / "best_model.zip"
            self.assertEqual(runs.run_of_model(model), ("mario", "r"))

    def test_run_of_model_requires_run_layout(self):
        self.assertEqual(runs.run_of_model(Path("models/mario/r/checkpoints/final.zip")), ("mario", "r"))
        self.assertIsNone(runs.run_of_model(Path("/tmp/somewhere/final.zip")))
        self.assertIsNone(runs.run_of_model(Path("models/mario/_level_states/logs/x.zip")))

    def test_redundant_checkpoints_matches_compact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = self._make_run(root, "mario", "r", snaps=(3000, 1000, 2000, 5000, 4000, 7000, 6000))
            doomed = runs.redundant_checkpoints("mario", "r", root, keep=5)
            self.assertEqual([x.name for x in doomed], ["ppo_1000_steps.zip", "ppo_2000_steps.zip"])
            (p["checkpoints"] / "final.zip").write_bytes(b"x")
            self.assertEqual(len(runs.redundant_checkpoints("mario", "r", root)), 7)
            self.assertEqual(runs.redundant_checkpoints("mario", "missing", root), [])

    def test_build_train_cmd_accepts_sparse_normalized_preset(self):
        from aiboy import presets
        cmd = runs.build_train_cmd(presets.normalize({"game": "mario"}), "r")
        self.assertEqual(cmd[cmd.index("--n-envs") + 1], str(presets.DEFAULT_CONFIG["n_envs"]))

    def test_prune_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = self._make_run(root, "mario", "r", best=True, final=True,
                               snaps=(1000, 3000, 2000, 5000, 4000, 7000, 6000))
            deleted = runs.prune_checkpoints(p["checkpoints"], keep=5)
            self.assertEqual(sorted(x.name for x in deleted), ["ppo_1000_steps.zip", "ppo_2000_steps.zip"])
            left = sorted(x.name for x in p["checkpoints"].glob("*.zip"))
            self.assertEqual(left, ["final.zip", "ppo_3000_steps.zip", "ppo_4000_steps.zip",
                                    "ppo_5000_steps.zip", "ppo_6000_steps.zip", "ppo_7000_steps.zip"])
            self.assertEqual(runs.prune_checkpoints(p["checkpoints"], keep=5), [])
            self.assertEqual(runs.prune_checkpoints(p["checkpoints"], keep=None), [])   # disabled
            self.assertEqual(len(runs.prune_checkpoints(p["checkpoints"], keep=0)), 5)  # delete all
            self.assertTrue((p["checkpoints"] / "final.zip").exists())
            self.assertTrue((p["logs"] / "best_model.zip").exists())

    def test_compact_and_slim(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = self._make_run(root, "mario", "done", best=True, final=True, snaps=(1000, 2000, 3000))
            (p["tensorboard"]).mkdir(); (p["tensorboard"] / "events").write_bytes(b"e" * 10)
            freed = runs.compact_run("mario", "done", root)
            self.assertGreater(freed, 0)
            self.assertEqual(sorted(x.name for x in p["checkpoints"].glob("*.zip")), ["final.zip"])
            self.assertTrue((p["logs"] / "best_model.zip").exists())
            self.assertTrue((p["tensorboard"] / "events").exists())
            q = self._make_run(root, "mario", "stopped", best=True, snaps=tuple(range(1000, 9000, 1000)))
            runs.compact_run("mario", "stopped", root, keep=5)
            self.assertEqual(len(list(q["checkpoints"].glob("ppo_*"))), 5)   # no final.zip: keep 5
            t = self._make_run(root, "mario", "tune-001", best=True, final=True, snaps=(500,))
            t["tensorboard"].mkdir(); (t["tensorboard"] / "events").write_bytes(b"e")
            self.assertGreater(runs.slim_trial_run("mario", "tune-001", root), 0)
            self.assertFalse(t["checkpoints"].exists()); self.assertFalse(t["tensorboard"].exists())
            self.assertTrue((t["logs"] / "best_model.zip").exists())
            self.assertEqual(runs.slim_trial_run("mario", "tune-001", root), 0)

    def test_housekeeping(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("tune-001", "tune-002-s1", "tune-010", "tuner-001", "wizard-001", "keep"):
                self._make_run(root, "mario", name, best=True, snaps=(1000,))
            self._make_run(root, "mario", "_level_states")
            (root / "mario" / "_tune").mkdir()
            (root / "mario" / "_tune" / "tune.json").write_text("{}")
            self.assertEqual(runs.tune_trial_runs("mario", "tune", root),
                             ["tune-001", "tune-002-s1", "tune-010"])
            n, size = runs.tune_data_size("mario", "tune", root)
            self.assertEqual(n, 3)
            self.assertGreater(size, 0)
            n, freed = runs.delete_tune_data("mario", "tune", root)
            self.assertEqual(n, 3)
            self.assertEqual(freed, size)
            self.assertEqual(runs.list_runs("mario", root), ["keep", "tuner-001", "wizard-001"])
            self.assertFalse((root / "mario" / "_tune" / "tune.json").exists())
            self.assertEqual(runs.delete_tune_data("mario", "tune", root), (0, 0))
            self.assertGreater(runs.delete_run("mario", "keep", root), 0)
            self.assertEqual(runs.delete_run("mario", "keep", root), 0)
            with self.assertRaises(ValueError):
                runs.delete_run("mario", "_level_states", root)
            self.assertTrue((root / "mario" / "_level_states").exists())
        self.assertEqual(runs.format_size(512), "512 B")
        self.assertEqual(runs.format_size(3 * 1024 * 1024), "3.0 MB")

    def test_build_train_cmd(self):
        cfg = {"game": "mario", "n_envs": 2, "timesteps": 4000, "ent_coef": 0.01,
               "learning_rate": 2.5e-4, "n_steps": 128, "batch_size": 128}
        cmd = runs.build_train_cmd(cfg, "run1", timesteps=999, eval_freq=500, resume=True)
        self.assertEqual(cmd[:4], [sys.executable, "-u", "main.py", "train"])
        self.assertIn("--resume", cmd)
        self.assertEqual(cmd[cmd.index("--timesteps") + 1], "999")
        self.assertEqual(cmd[cmd.index("--eval-freq") + 1], "500")
        self.assertEqual(cmd[cmd.index("--gamma") + 1], "0.99")
        self.assertEqual(cmd[cmd.index("--time-budget") + 1], "250")
        self.assertEqual(cmd[cmd.index("--stall-steps") + 1], "0")
        self.assertEqual(cmd[cmd.index("--run-name") + 1], "run1")
        self.assertNotIn("--resume", runs.build_train_cmd(cfg, "run1"))
        self.assertNotIn("--source", cmd)
        tagged = runs.build_train_cmd(cfg, "run1", source="wizard")
        self.assertEqual(tagged[tagged.index("--source") + 1], "wizard")


if __name__ == "__main__":
    unittest.main()


class ResumeEvalHistoryTests(unittest.TestCase):
    def test_continued_run_keeps_its_evaluations_and_best_score(self):
        import tempfile
        import numpy as np
        from aiboy.cli import _continue_eval_history

        class FakeEvalCallback:
            evaluations_timesteps: list = []
            evaluations_results: list = []
            evaluations_length: list = []
            evaluations_successes: list = []
            best_mean_reward = -np.inf
            last_mean_reward = -np.inf

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "evaluations.npz"
            np.savez(f, timesteps=[300, 600], results=[[3.0, 5.0], [-450.0, -460.0]],
                     ep_lengths=[[71, 80], [70, 70]])
            cb = FakeEvalCallback()
            _continue_eval_history(cb, f)
            self.assertEqual(cb.evaluations_timesteps, [300, 600])
            self.assertEqual(cb.evaluations_results, [[3.0, 5.0], [-450.0, -460.0]])
            self.assertEqual(cb.evaluations_length, [[71, 80], [70, 70]])
            self.assertEqual(cb.best_mean_reward, 4.0)          # the run's best so far
            self.assertEqual(cb.last_mean_reward, -455.0)
            cb2 = FakeEvalCallback()
            _continue_eval_history(cb2, Path(tmp) / "missing.npz")   # a run without evals
            self.assertEqual(cb2.best_mean_reward, -np.inf)
