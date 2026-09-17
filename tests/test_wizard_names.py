import unittest

import wizard


class WizardNameTests(unittest.TestCase):
    def test_short_goal(self):
        self.assertEqual(wizard.short_goal("Mario — Campaign, recommended (~15 min)"),
                         "Campaign, recommended")
        self.assertEqual(wizard.short_goal("My preset"), "My preset")

    def test_preset_names(self):
        goal = "Mario — Campaign, recommended (~15 min)"
        self.assertEqual(wizard.default_preset_name(goal, {"ent_coef": 0.03, "learning_rate": 3e-4}),
                         "Campaign, recommended · ent 0.03 · lr 0.0003")
        self.assertEqual(wizard.default_preset_name(goal, {}), "Campaign, recommended (wizard)")
        self.assertEqual(wizard.compact_overrides({"n_steps": 512, "obs_type": "tiles"}),
                         "steps 512 · obs tiles")

    def test_run_slug(self):
        self.assertEqual(wizard.run_slug("Campaign, recommended · ent 0.03 · lr 0.0003"),
                         "campaign-recommended-ent0.03-lr0.0003")
        self.assertEqual(wizard.run_slug("Campaign, recommended (wizard)"), "campaign-recommended")
        self.assertEqual(wizard.run_slug("Marathon, all levels · steps 1024"),
                         "marathon-all-levels-steps1024")
        self.assertEqual(wizard.run_slug("···"), "wizard")
        self.assertNotIn(" ", wizard.run_slug("a b   c"))

    def test_plain_plan_and_goal_description(self):
        text = wizard.plain_plan(6, 12, 700.0, 900.0, True)
        self.assertTrue(text.startswith("Plan: try 6 variations plus the goal's own settings "
                                        "(12 short training runs"), text)
        self.assertIn("Total about 27 min", text)
        self.assertEqual(wizard.plain_plan(0, 0, 0.0, 900.0, False),
                         "Plan: train for about 15 min with the goal's own settings.")
        cfg = {"start_level": "marathon", "timesteps": 2_000_000, "n_envs": 10, "obs_type": "tiles"}
        desc = wizard.describe_preset(cfg)
        self.assertTrue(desc.startswith("It tries to beat every level in one go."), desc)
        self.assertIn("2.00M steps, 10 games in parallel", desc)
        self.assertIn("level 3-2 only", wizard.describe_preset({**cfg, "start_level": "3-2"}))

    def test_unique_run_name(self):
        from unittest import mock
        with mock.patch.object(wizard.runs, "list_runs", return_value=["x", "x-2"]):
            self.assertEqual(wizard.unique_run_name("mario", "y"), "y")
            self.assertEqual(wizard.unique_run_name("mario", "x"), "x-3")


if __name__ == "__main__":
    unittest.main()
