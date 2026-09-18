import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiboy import presets


class PresetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(presets, "PRESETS_FILE", Path(self._tmp.name) / "user.json")
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_builtins_are_complete_and_mario_only(self):
        for name, cfg in presets.BUILTIN_PRESETS.items():
            missing = set(presets.PRESET_FIELDS) - set(cfg) - set(presets.PRESET_DEFAULTS)
            self.assertFalse(missing, f"{name} lacks {missing}")
            self.assertEqual(cfg["game"], "mario", name)
        self.assertIn(presets.RECOMMENDED_PRESET, presets.BUILTIN_PRESETS)
        self.assertIn(presets.SMOKE_TEST_PRESET, presets.BUILTIN_PRESETS)

    def test_user_presets_roundtrip(self):
        self.assertEqual(presets.load_user(), {})
        cfg = {"game": "mario", "timesteps": 10, "junk": 1}
        presets.upsert("mine", cfg)
        saved = presets.load_user()["mine"]
        self.assertNotIn("junk", saved)
        self.assertEqual(saved["gamma"], 0.99)          # defaults filled in
        self.assertTrue(presets.is_user("mine"))
        self.assertFalse(presets.is_builtin("mine"))
        self.assertIn("mine", presets.load_all())
        presets.delete("mine")
        self.assertFalse(presets.is_user("mine"))

    def test_builtin_override_and_reset(self):
        name = presets.RECOMMENDED_PRESET
        shipped = presets.BUILTIN_PRESETS[name]
        self.assertEqual(presets.kind(name), "built-in")
        with self.assertRaises(ValueError):
            presets.delete(name)                      # nothing to delete yet
        presets.upsert(name, {**shipped, "ent_coef": 0.5})
        self.assertEqual(presets.kind(name), "modified")
        self.assertTrue(presets.is_overridden(name))
        self.assertEqual(presets.load_all()[name]["ent_coef"], 0.5)
        self.assertEqual(presets.BUILTIN_PRESETS[name]["ent_coef"], shipped["ent_coef"])
        with self.assertRaises(ValueError):
            presets.rename(name, "x")                 # built-ins keep their name
        presets.reset(name)
        self.assertEqual(presets.kind(name), "built-in")
        self.assertEqual(presets.load_all()[name]["ent_coef"], shipped["ent_coef"])
        with self.assertRaises(ValueError):
            presets.reset("not a builtin")
        presets.upsert("mine", {"game": "mario"})
        self.assertEqual(presets.kind("mine"), "user")

    def test_rename_rules(self):
        presets.upsert("a", {"game": "mario"})
        presets.upsert("b", {"game": "mario"})
        presets.rename("a", "c")
        self.assertEqual(set(presets.load_user()), {"b", "c"})
        with self.assertRaises(ValueError):
            presets.rename("c", "b")                              # collision
        with self.assertRaises(ValueError):
            presets.rename("missing", "x")
        with self.assertRaises(ValueError):
            presets.rename(presets.RECOMMENDED_PRESET, "x")       # built-in
        with self.assertRaises(ValueError):
            presets.rename("c", presets.RECOMMENDED_PRESET)

    def test_normalize_completes_missing_fields(self):
        cfg = presets.normalize({"game": "mario", "timesteps": 7, "junk": 1})
        self.assertEqual(set(cfg), set(presets.PRESET_FIELDS))
        self.assertEqual(cfg["timesteps"], 7)
        self.assertEqual(cfg["n_envs"], presets.DEFAULT_CONFIG["n_envs"])
        self.assertNotIn("junk", cfg)
        self.assertEqual(set(presets.DEFAULT_CONFIG), set(presets.PRESET_FIELDS))
        # load_all() delivers complete configs even for sparse user presets
        presets.PRESETS_FILE.write_text('{"sparse": {"game": "mario"}}')
        self.assertEqual(set(presets.load_all()["sparse"]), set(presets.PRESET_FIELDS))

    def test_kind_accepts_preloaded_user_dict(self):
        presets.upsert("u", {"game": "mario"})
        user = presets.load_user()
        self.assertEqual(presets.kind("u", user), "user")
        self.assertEqual(presets.kind(presets.RECOMMENDED_PRESET, user), "built-in")
        self.assertEqual(presets.kind(presets.RECOMMENDED_PRESET, {presets.RECOMMENDED_PRESET: {}}),
                         "modified")

    def test_sorted_names_puts_recommended_first(self):
        all_presets = dict(presets.BUILTIN_PRESETS)
        all_presets["zzz user"] = {"game": "mario"}
        all_presets["other game"] = {"game": "kirby"}
        names = presets.sorted_names(all_presets, "mario")
        self.assertEqual(names[0], presets.RECOMMENDED_PRESET)
        self.assertEqual(names[-1], "zzz user")
        self.assertNotIn("other game", names)

    def test_corrupt_user_file_is_ignored(self):
        presets.PRESETS_FILE.write_text("{ not json")
        self.assertEqual(presets.load_user(), {})
        presets.PRESETS_FILE.write_text(json.dumps([1, 2]))
        self.assertEqual(presets.load_user(), {})


if __name__ == "__main__":
    unittest.main()
