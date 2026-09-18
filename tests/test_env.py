import unittest

from aiboy import env
from aiboy import games


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
        self.assertIs(env.GAMES, games.GAMES)          # env re-exports games.py

    def test_light_modules_have_no_heavy_top_level_imports(self):
        """The GUI must open without loading PyBoy / SB3 / torch, and the
        helper sub-commands must start fast: only the trainer and env.py may
        import the heavy libraries, and only inside functions."""
        import ast
        from pathlib import Path
        heavy = {"torch", "pyboy", "stable_baselines3", "aiboy.env", "env"}
        light = [Path("aiboy") / f"{m}.py" for m in ("games", "presets", "runs", "tuning",
                                                     "experience", "cli", "paths")]
        light += sorted(Path("aiboy/gui").glob("*.py"))
        for path in light:
            tree = ast.parse(path.read_text())
            for node in tree.body:                      # module level only
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                bad = {n for n in names if n in heavy or n.split(".")[0] in heavy}
                self.assertFalse(bad, f"{path} imports {bad} at module level")


class ObservationTests(unittest.TestCase):
    def test_obs_types(self):
        self.assertEqual(games.OBS_TYPES, ("tiles", "pixels"))
        self.assertEqual(games.POWER_NAMES[games.POWER_SUPERBALL], "superball")

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_tiles_overlay_includes_power_up(self):
        import numpy as np
        from pyboy import PyBoy
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles")
            e.reset()
            obs = e._obs()
            self.assertEqual(obs.shape, (16, 20, 1))
            self.assertEqual(obs[0, 6, 0], 0.0)                    # small Mario reads 0
            self.assertGreater(obs[0, 2, 0], 0.9)                  # timer near full
            self.assertEqual(e.power_state(), games.POWER_SMALL)
            pb.set_memory_value(games.ADDR_POWERUP_STATE, 1)       # growing -> super
            for _ in range(30):
                pb.tick()
            self.assertEqual(e.power_state(), games.POWER_SUPER)
            self.assertEqual(e._obs()[0, 6, 0], 0.5)
            pb.set_memory_value(games.ADDR_SUPERBALL, 1)
            self.assertEqual(e.power_state(), games.POWER_SUPERBALL)
            self.assertEqual(e._obs()[0, 6, 0], 1.0)
            _, _, _, _, info = e.step(0)
            self.assertEqual(info["power"], "superball")
            self.assertTrue(np.isfinite(e._obs()).all())
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_marathon_always_starts_at_1_1(self):
        from pyboy import PyBoy
        games.prepare_level_states("marathon")
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles", start_level="marathon")
            e.reset(seed=3)
            for _ in range(4):
                e.reset()
                self.assertEqual(e._marathon_idx, 0)
                self.assertEqual(tuple(e.gw.world), (1, 1))      # training, eval and play alike
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_time_budget_truncates_and_stall_rule_is_off_by_default(self):
        from pyboy import PyBoy
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles", start_level="1-1", time_budget=3)
            self.assertEqual(e.stuck_steps, 0)                     # stall rule off by default
            e.reset()
            for step in range(600):                                 # stand still (NOOP)
                _, _, term, trunc, info = e.step(0)
                if term or trunc:
                    break
            self.assertTrue(trunc and info["time_budget_exceeded"], info)
            self.assertFalse(info["died"] or info["stalled"])
            self.assertGreaterEqual(info["time_used"], 3)
            self.assertLess(step, 600)
            # stall rule when enabled
            e2 = env.MarioEnv(pb, obs_type="tiles", start_level="1-1", time_budget=0, stuck_steps=20)
            e2.reset()
            for step in range(200):
                _, _, term, trunc, info = e2.step(0)
                if term or trunc:
                    break
            self.assertTrue(trunc and info["stalled"], info)
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_marathon_death_ends_the_run_and_the_next_one_restarts_at_1_1(self):
        from pyboy import PyBoy
        games.prepare_level_states("marathon")
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles", start_level="marathon", time_budget=0)
            e.reset()
            term = trunc = False
            for _ in range(1500):                          # run right into the first enemy
                _, _, term, trunc, info = e.step(5)
                self.assertNotIn("marathon_next_level", info)   # no level change without a clear
                if term or trunc:
                    break
            self.assertTrue(term and info["died"], info)       # the death ends the episode
            e.reset()
            self.assertEqual(tuple(e.gw.world), (1, 1))         # and the next one starts over
            self.assertEqual(e._marathon_idx, 0)
            # an exhausted time budget ends the run the same way (truncation)
            e = env.MarioEnv(pb, obs_type="tiles", start_level="marathon", time_budget=3)
            e.reset()
            for _ in range(600):
                _, _, term, trunc, info = e.step(0)
                if term or trunc:
                    break
            self.assertTrue(trunc and info["time_budget_exceeded"], info)
            e.reset()
            self.assertEqual(tuple(e.gw.world), (1, 1))
        finally:
            pb.stop(save=False)

    def test_invalid_obs_type_rejected(self):
        class FakePyBoy:
            def game_wrapper(self):
                return None
        with self.assertRaises(ValueError):
            env.MarioEnv(FakePyBoy(), obs_type="voxels")


class FormFieldTests(unittest.TestCase):
    def test_form_fields_match_preset_fields(self):
        from aiboy.gui import widgets
        from aiboy import presets
        keys = {f.key for f in widgets.FIELDS}
        self.assertEqual(keys, set(presets.PRESET_FIELDS) - {"game"})
        for f in widgets.FIELDS:
            default = presets.DEFAULT_CONFIG[f.key]
            if f.kind == "choice":
                self.assertIn(default, f.choices, f.key)
            elif f.kind == "int":
                self.assertIsInstance(default, int, f.key)
            if f.kind in ("int", "float") and not f.entry:
                self.assertTrue(f.lo <= default <= f.hi, f.key)


class RomDiscoveryTests(unittest.TestCase):
    def test_discover_roms_orders_known_games_first(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("zelda.gb", "kirby.gb", "mario.gb", "notes.txt", "wario.GB"):
                (root / name).write_bytes(b"x")
            names = [r.name for r in env.discover_roms(root)]
            self.assertEqual(names, ["mario", "kirby", "wario", "zelda"])
            self.assertEqual(env.discover_roms(root / "missing"), [])

    def test_status_levels(self):
        from pathlib import Path
        mario = env.RomInfo("mario", Path("ROMs/mario.gb"))
        self.assertEqual(mario.status()[0], "ok")            # supported before the probe
        self.assertTrue(mario.runnable)
        mario.probed, mario.title = True, "SOMETHING ELSE"
        self.assertEqual(mario.status()[0], "unsupported")   # wrong cartridge
        kirby = env.RomInfo("kirby", Path("ROMs/kirby.gb"))
        self.assertEqual(kirby.status()[0], "checking")
        kirby.probed, kirby.title, kirby.has_wrapper = True, "KIRBY DREAM LA", True
        self.assertEqual(kirby.status()[0], "experimental")
        self.assertFalse(kirby.runnable)
        unknown = env.RomInfo("zelda", Path("ROMs/zelda.gb"), probed=True, title="ZELDA",
                              has_wrapper=False)
        self.assertEqual(unknown.status()[0], "unsupported")
        broken = env.RomInfo("x", Path("x.gb"), probed=True, error="boom")
        self.assertEqual(broken.status()[0], "unsupported")

    @unittest.skipUnless((env.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_probe_real_rom(self):
        title, wrapper, error = env.probe_rom(env.ROM_DIR / "mario.gb")
        self.assertIsNone(error)
        self.assertEqual(title, "SUPER MARIOLAN")
        self.assertTrue(wrapper)

    def test_probe_bad_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".gb", delete=False) as f:
            f.write(b"not a rom")
        title, wrapper, error = env.probe_rom(Path(f.name), timeout_sec=60)
        self.assertIsNotNone(error)
        self.assertFalse(wrapper)


if __name__ == "__main__":
    unittest.main()
