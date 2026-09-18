import sys
import unittest
from unittest import mock

from aiboy import paths
from aiboy import presets


class PathTests(unittest.TestCase):
    def test_source_tree_layout(self):
        self.assertFalse(paths.FROZEN)
        self.assertEqual(paths.BUNDLE_DIR, paths.DATA_DIR)
        self.assertEqual(paths.app_command("train", "--x"), [sys.executable, "-u", "main.py", "train", "--x"])
        self.assertTrue((paths.BUNDLE_DIR / "builtin_presets.json").exists())

    def test_recommended_n_envs_follows_profile_but_never_exceeds_cores(self):
        with mock.patch.object(paths, "system_profile", return_value={}):
            with mock.patch.object(paths, "cpu_count", return_value=16):
                self.assertEqual(paths.recommended_n_envs(), 12)
            with mock.patch.object(paths, "cpu_count", return_value=4):
                self.assertEqual(paths.recommended_n_envs(), 4)
        with mock.patch.object(paths, "system_profile", return_value={"n_envs": 6}):
            with mock.patch.object(paths, "cpu_count", return_value=8):
                self.assertEqual(paths.recommended_n_envs(), 6)
            with mock.patch.object(paths, "cpu_count", return_value=2):
                self.assertEqual(paths.recommended_n_envs(), 2)     # built on a bigger machine

    def test_builtin_presets_fit_the_machine(self):
        with mock.patch.object(presets, "recommended_n_envs", return_value=4):
            self.assertEqual(presets.fit_to_machine({"n_envs": 10})["n_envs"], 4)
            self.assertEqual(presets.fit_to_machine({"n_envs": 2})["n_envs"], 2)
            self.assertEqual(presets.fit_to_machine({}), {})
        for cfg in presets.BUILTIN_PRESETS.values():
            self.assertLessEqual(cfg["n_envs"], paths.recommended_n_envs())


if __name__ == "__main__":
    unittest.main()
