import unittest

import env


class LevelSpecTests(unittest.TestCase):
    def test_parse_start_level(self):
        self.assertIsNone(env.parse_start_level(None))
        self.assertIsNone(env.parse_start_level("default"))
        self.assertEqual(env.parse_start_level("random"), "random")
        self.assertEqual(env.parse_start_level("all_levels"), "marathon")
        self.assertEqual(env.parse_start_level("2-1"), (2, 1))
        self.assertEqual(env.parse_start_level((3, 2)), (3, 2))
        for bad in ("2-3", "4-3", "5-1", "1-2-3", "nope"):
            with self.assertRaises(ValueError, msg=bad):
                env.parse_start_level(bad)

    def test_level_choices_and_targets(self):
        choices = env.level_choices()
        self.assertEqual(choices[:4], list(env.LEVEL_MODES))
        self.assertEqual(len(choices), 4 + len(env.SML_ALL_LEVELS))
        self.assertNotIn("2-3", choices)
        self.assertEqual(env.level_targets("default"), [])
        self.assertEqual(env.level_targets("1-2"), [(1, 2)])
        self.assertEqual(env.level_targets("marathon"), list(env.SML_ALL_LEVELS))

    def test_mario_action_table_is_consistent(self):
        self.assertEqual(len(env.MarioEnv.ACTIONS), len(env.MarioEnv.RELEASES))
        self.assertEqual(len(env.MarioEnv.ACTIONS), len(env.MarioEnv.ACTION_NAMES))
        self.assertEqual(env.MarioEnv.ACTION_NAMES[0], "NOOP")

    def test_game_registry(self):
        self.assertEqual(env.SUPPORTED_GAMES, ("mario",))
        self.assertIn("kirby", env.GAMES)


if __name__ == "__main__":
    unittest.main()
