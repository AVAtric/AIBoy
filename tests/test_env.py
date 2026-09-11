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
