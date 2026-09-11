import sys
import tempfile
import unittest
from pathlib import Path

import runs


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

    def test_build_train_cmd_accepts_sparse_normalized_preset(self):
        import presets
        cmd = runs.build_train_cmd(presets.normalize({"game": "mario"}), "r")
        self.assertEqual(cmd[cmd.index("--n-envs") + 1], str(presets.DEFAULT_CONFIG["n_envs"]))

    def test_build_train_cmd(self):
        cfg = {"game": "mario", "n_envs": 2, "timesteps": 4000, "ent_coef": 0.01,
               "learning_rate": 2.5e-4, "n_steps": 128, "batch_size": 128}
        cmd = runs.build_train_cmd(cfg, "run1", timesteps=999, eval_freq=500, resume=True)
        self.assertEqual(cmd[:4], [sys.executable, "-u", "main.py", "train"])
        self.assertIn("--resume", cmd)
        self.assertEqual(cmd[cmd.index("--timesteps") + 1], "999")
        self.assertEqual(cmd[cmd.index("--eval-freq") + 1], "500")
        self.assertEqual(cmd[cmd.index("--gamma") + 1], "0.99")
        self.assertEqual(cmd[cmd.index("--run-name") + 1], "run1")
        self.assertNotIn("--resume", runs.build_train_cmd(cfg, "run1"))


if __name__ == "__main__":
    unittest.main()
