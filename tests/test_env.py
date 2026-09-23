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
        self.assertEqual(env.parse_start_level("2-3"), (2, 3))      # the submarine
        self.assertEqual(env.parse_start_level("4-3"), (4, 3))      # the plane, Tatanga
        for bad in ("5-1", "1-4", "1-2-3", "nope"):
            with self.assertRaises(ValueError, msg=bad):
                env.parse_start_level(bad)

    def test_level_choices_and_targets(self):
        choices = env.level_choices()
        self.assertEqual(choices[:4], list(env.LEVEL_MODES))
        self.assertEqual(len(choices), 4 + len(env.SML_ALL_LEVELS))
        self.assertEqual(len(env.SML_ALL_LEVELS), 12)                # a marathon is the whole game
        self.assertEqual(env.SML_ALL_LEVELS[-1], (4, 3))
        self.assertIn("2-3", choices)
        self.assertEqual(env.level_targets("default"), [])
        self.assertEqual(env.level_targets("1-2"), [(1, 2)])
        self.assertEqual(env.level_targets("marathon"), list(env.SML_ALL_LEVELS))

    def test_action_tables_are_consistent(self):
        for game, cls in env.ENV_CLASSES.items():
            self.assertEqual(len(cls.ACTIONS), len(cls.ACTION_NAMES), game)
            self.assertEqual(cls.ACTION_NAMES[0], "NOOP")
            self.assertEqual(cls.ACTIONS[0], ())
            for i, presses in enumerate(cls.ACTIONS):
                releases = cls.releases(i)
                self.assertEqual(len(releases), len(presses), (game, i))
                for press, release in zip(presses, releases):
                    self.assertEqual(env.RELEASE_OF[press], release)
            self.assertTrue(issubclass(cls, env.GameBoyEnv))

    def test_game_registry(self):
        self.assertEqual(env.SUPPORTED_GAMES, ("mario", "kirby"))
        self.assertEqual(set(env.ENV_CLASSES), set(env.SUPPORTED_GAMES))
        self.assertIn("wario", env.GAMES)
        self.assertIs(env.GAMES, games.GAMES)          # env re-exports games.py
        self.assertTrue(games.GAMES["mario"].levels)
        self.assertFalse(games.GAMES["kirby"].levels)
        self.assertEqual(games.env_version("mario"), games.ENV_VERSION)
        self.assertNotEqual(games.env_version("kirby"), games.env_version("mario"))
        self.assertEqual(games.env_version("tetris"), "generic-1")

    def test_level_modes_are_mario_only(self):
        self.assertIsNone(games.check_start_level("mario", "marathon"))
        self.assertIsNone(games.check_start_level("kirby", "default"))
        self.assertIsNone(games.check_start_level("kirby", None))
        self.assertIn("Kirby", games.check_start_level("kirby", "1-1"))
        self.assertIn("level", games.check_start_level("wario", "random"))
        with self.assertRaises(ValueError):
            env.make_env("kirby", None, start_level="2-1")
        with self.assertRaises(ValueError):
            env.make_env("tetris", None)

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
    def test_tile_lut_shows_all_the_ground_and_no_decoration(self):
        # Measured with tools/mario_tile_survey.py (see games.SML_BACKGROUND_TILES).
        lut = env.MARIO_TILE_LUT
        for tile in (352, 353, 357, 358, 360, 361, 362, 142, 143, 383, 368, 239):  # stood on, lifts
            self.assertEqual(lut[tile], 0.5, tile)
        for tile in games.SML_BACKGROUND_TILES:                                   # walked through
            self.assertEqual(lut[tile], 0.0, tile)
        for tile in (300, 305, 306, 307, 311, 310, 350, 320, 325):               # sky, palms, hills, clouds
            self.assertEqual(lut[tile], 0.0, tile)
        self.assertEqual(lut[144], 1.0)                                          # goomba
        self.assertEqual(lut[0], -1.0)                                           # Mario
        for tile in (200, 201, 168, 169, 98, 216, 250, 251):                    # hazards PyBoy misses
            self.assertEqual(lut[tile], 1.0, tile)
        for tile in (88, 89, 145, 246, 254, 122, 110):                          # score numbers, torpedo
            self.assertEqual(lut[tile], 0.0, tile)
        self.assertEqual(lut[112], -1.0)                                         # the submarine
        self.assertEqual(games.env_version("mario"), "mario-5")

    def test_mario_has_up_actions_appended(self):
        names = env.MarioEnv.ACTION_NAMES
        self.assertEqual(len(names), 13)
        self.assertEqual(names[11:], ("UP", "UP+JUMP"))
        self.assertEqual(names[10], "DOWN")                  # the first 11 keep their meaning

    def test_old_models_play_through_a_narrowed_action_space(self):
        from gymnasium import spaces
        from stable_baselines3.common.vec_env import DummyVecEnv

        class Dummy(__import__("gymnasium").Env):
            observation_space = spaces.Box(-1, 1, (2,))
            action_space = spaces.Discrete(13)
            def reset(self, *, seed=None, options=None): return self.observation_space.sample() * 0, {}
            def step(self, a):
                assert 0 <= int(a) < 13
                return self.observation_space.sample() * 0, 0.0, False, False, {}
        vec = DummyVecEnv([Dummy])
        self.assertIs(env.fit_action_space(vec, spaces.Discrete(13)), vec)
        narrow = env.fit_action_space(vec, spaces.Discrete(11))
        self.assertEqual(narrow.action_space, spaces.Discrete(11))
        narrow.reset(); narrow.step(__import__("numpy").array([10]))
        with self.assertRaises(ValueError):
            env.fit_action_space(vec, spaces.Discrete(14))

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
    def test_time_budget_and_stall_end_the_attempt_like_a_death(self):
        """Standing still is not free: an exhausted time budget or a stall
        ends the episode (terminal, not truncated) with the death penalty,
        and the stall limit is on by default (games.DEFAULT_STALL_STEPS)."""
        from pyboy import PyBoy
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles", start_level="1-1", time_budget=3, stuck_steps=0)
            e.reset()
            for step in range(600):                                 # stand still (NOOP)
                _, r, term, trunc, info = e.step(0)
                if term or trunc:
                    break
            self.assertTrue(term and info["time_budget_exceeded"], info)
            self.assertFalse(trunc or info["died"] or info["stalled"])
            self.assertLessEqual(r, -e.death_penalty)
            self.assertGreaterEqual(info["time_used"], 3)
            self.assertLess(step, 600)
            # the stall rule: on by default, the same ending
            e2 = env.MarioEnv(pb, obs_type="tiles", start_level="1-1", time_budget=0)
            self.assertEqual(e2.stuck_steps, games.DEFAULT_STALL_STEPS)
            self.assertGreater(games.DEFAULT_STALL_STEPS, 0)
            e2.stuck_steps = 20
            e2.reset()
            for step in range(200):
                _, r, term, trunc, info = e2.step(0)
                if term or trunc:
                    break
            self.assertTrue(term and info["stalled"], info)
            self.assertFalse(trunc or info["time_budget_exceeded"])
            self.assertLessEqual(r, -e2.death_penalty)
            self.assertEqual(step, 19)                             # the 20th step (0-based)
            # walking keeps the stall counter at zero
            e2.reset()
            for _ in range(25):
                _, _, term, trunc, info = e2.step(e2.ACTION_NAMES.index("RIGHT"))
                self.assertFalse(term or trunc, info)
            self.assertLess(info["stuck"], 5)
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
            # an exhausted time budget ends the run the same way (a terminal
            # state with the death penalty, see test_time_budget_and_stall_...)
            e = env.MarioEnv(pb, obs_type="tiles", start_level="marathon", time_budget=3,
                             stuck_steps=0)
            e.reset()
            for _ in range(600):
                _, r, term, trunc, info = e.step(0)
                if term or trunc:
                    break
            self.assertTrue(term and info["time_budget_exceeded"], info)
            self.assertLessEqual(r, -e.death_penalty)
            e.reset()
            self.assertEqual(tuple(e.gw.world), (1, 1))
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_a_fixed_level_reports_won_only_when_the_clear_ends_the_episode(self):
        """info["won"] is what the exploration guard reads (through the
        Monitor) to tell a mastered level from a collapse: the clear that
        ends a fixed-level episode wins it, a death or a stall does not."""
        from pyboy import PyBoy
        games.prepare_level_states("1-1")
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles", start_level="1-1", time_budget=0, stuck_steps=5)
            e.reset()
            _, _, term, _, info = e.step(0)
            self.assertFalse(term or info["won"])
            pb.set_memory_value(games.ADDR_GAME_STATE, 0x07)          # the goal is touched
            _, _, term, _, info = e.step(0)
            self.assertTrue(term and info["level_cleared"] and info["won"], info)
            e.reset()
            for _ in range(20):                                       # stand still into the stall
                _, _, term, _, info = e.step(0)
                if term:
                    break
            self.assertTrue(term and info["stalled"], info)
            self.assertFalse(info["won"])
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_marathon_walks_all_twelve_levels_and_ends_after_tatanga(self):
        """Every clear (the game-state byte set to "goal touched") loads the
        next level's state in the same life, through the two vehicle levels,
        and the twelfth clear ends the run with the marathon bonus."""
        from pyboy import PyBoy
        games.prepare_level_states("marathon")
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.MarioEnv(pb, obs_type="tiles", start_level="marathon", time_budget=0)
            e.reset()
            for i, lvl in enumerate(games.SML_ALL_LEVELS):
                self.assertEqual(tuple(e.gw.world), lvl)
                for _ in range(3):
                    _, _, term, trunc, info = e.step(0)
                    self.assertFalse(term or trunc, (lvl, info))
                    self.assertIn(info["game_state"], games.SML_PLAY_STATES, lvl)
                pb.set_memory_value(games.ADDR_GAME_STATE, 0x07)      # the goal is touched
                _, reward, term, trunc, info = e.step(0)
                self.assertTrue(info["level_cleared"], lvl)
                self.assertGreaterEqual(reward, e.completion_bonus)
                if i < len(games.SML_ALL_LEVELS) - 1:
                    self.assertFalse(term or info["won"], lvl)
                    self.assertEqual(info["marathon_next_level"], games.SML_ALL_LEVELS[i + 1])
                    self.assertEqual(info["marathon_clears"], i + 1)
                    self.assertEqual(info["time_used"], 0)             # the timer starts afresh
                else:
                    self.assertTrue(term and info["marathon_done"] and info["won"], info)
                    self.assertEqual(info["marathon_clears"], 12)
                    self.assertGreaterEqual(reward, 4 * e.completion_bonus)
            e.reset()
            self.assertEqual(tuple(e.gw.world), (1, 1))
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "mario.gb").exists(), "needs ROMs/mario.gb")
    def test_vehicle_levels_play_through_the_env(self):
        """2-3 and 4-3: the game runs in its auto-scroll state, UP moves the
        vehicle up and DOWN down, a crash is a death, and the time budget
        counts there too."""
        import numpy as np
        from pyboy import PyBoy
        games.prepare_level_states("marathon")
        pb = PyBoy(str(games.ROM_DIR / "mario.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            for lvl in sorted(games.SML_VEHICLE_LEVELS):
                e = env.MarioEnv(pb, obs_type="tiles", start_level=lvl, time_budget=0)
                obs, _ = e.reset()
                self.assertEqual(tuple(e.gw.world), lvl)
                self.assertEqual(int((obs == -1.0).sum()), 4)          # the vehicle is Mario
                rows = []
                for action in [11] * 12 + [10] * 12:                  # UP, then DOWN
                    obs, _, term, trunc, info = e.step(action)
                    self.assertFalse(term or trunc, info)
                    rows.append(int(np.argwhere(obs[1:, :, 0] == -1.0)[:, 0].min()))
                self.assertLess(rows[11], rows[0])
                self.assertGreater(rows[-1], rows[11])
                self.assertEqual(info["game_state"], 0x0D)
                self.assertGreater(info["x"], 284)                    # the level scrolls by itself
                # time budget: counted while playing the vehicle level
                e = env.MarioEnv(pb, obs_type="tiles", start_level=lvl, time_budget=2)
                e.reset()
                for _ in range(300):
                    _, _, term, trunc, info = e.step(11 if _ % 2 else 10)
                    if term or trunc:
                        break
                self.assertTrue(info["time_budget_exceeded"] or info["died"], info)
        finally:
            pb.stop(save=False)

    def test_invalid_obs_type_rejected(self):
        class FakePyBoy:
            def game_wrapper(self):
                return None
        with self.assertRaises(ValueError):
            env.MarioEnv(FakePyBoy(), obs_type="voxels")
        with self.assertRaises(ValueError):                    # no wrapper at all
            env.KirbyEnv(FakePyBoy(), obs_type="tiles")

    @unittest.skipUnless((games.ROM_DIR / "kirby.gb").exists(), "needs ROMs/kirby.gb")
    def test_kirby_env_plays_and_rewards_progress(self):
        import numpy as np
        from pyboy import PyBoy
        pb = PyBoy(str(games.ROM_DIR / "kirby.gb"), window_type="null", game_wrapper=True,
                   disable_renderer=True)
        try:
            e = env.make_env("kirby", pb, obs_type="tiles", stall_steps=30)
            self.assertIsInstance(e, env.KirbyEnv)
            obs, _ = e.reset()
            self.assertEqual(obs.shape, (16, 20, 1))
            self.assertTrue(np.isfinite(obs).all() and obs.min() >= 0.0 and obs.max() <= 1.0)
            self.assertEqual(obs[0, 0, 0], 1.0)                    # full health
            total, furthest = 0.0, 0
            right = e.ACTION_NAMES.index("RIGHT")
            for _ in range(40):                                    # walk right: progress pays
                _, r, term, trunc, info = e.step(right)
                total += r
                furthest = max(furthest, info["x"])
                if term or trunc:
                    break
            self.assertGreater(furthest, 50, info)
            self.assertGreater(total, 0.0)
            self.assertIn("health", info)
            self.assertEqual(info["lives"], 4)
            # standing still trips the stall rule (no timer in this game),
            # which ends the attempt like a death
            for _ in range(200):
                _, r, term, trunc, info = e.step(0)
                if term or trunc:
                    break
            self.assertTrue(term and info["stalled"], info)
            self.assertFalse(trunc)
            self.assertLessEqual(r, -e.death_penalty)
            # the grid a person's game would show: the same tile numbers
            grid = e.view()
            self.assertEqual(grid.shape, (16, 20))
            self.assertEqual(grid[0, 2], min(e.progress(), 8192) / 8192.0)
            obs, _ = e.reset()                                     # back to the start
            self.assertEqual(e.progress(), int(obs[0, 2, 0] * 8192))
            self.assertLess(e.progress(), 100)
        finally:
            pb.stop(save=False)

    @unittest.skipUnless((games.ROM_DIR / "kirby.gb").exists(), "needs ROMs/kirby.gb")
    def test_make_pyboy_env_builds_kirby_with_pixels(self):
        e = env.make_pyboy_env("kirby", obs_type="pixels", action_repeat=2)
        try:
            obs, _ = e.reset()
            self.assertEqual(obs.shape, (144, 160, 3))
            obs, r, term, trunc, info = e.step(1)
            self.assertEqual(obs.shape, (144, 160, 3))
            self.assertIn("score", info)
        finally:
            e.close()


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
        self.assertEqual(mario.status()[0], "playable")      # wrong cartridge: play it, no training
        self.assertFalse(mario.runnable)
        self.assertTrue(mario.playable)
        kirby = env.RomInfo("kirby", Path("ROMs/kirby.gb"))
        self.assertEqual(kirby.status()[0], "ok")            # supported like Mario
        self.assertTrue(kirby.playable and kirby.runnable)
        kirby.probed, kirby.title, kirby.has_wrapper = True, "KIRBY DREAM LA", True
        self.assertEqual(kirby.status()[0], "ok")
        wario = env.RomInfo("wario", Path("ROMs/wario.gb"))
        self.assertEqual(wario.status()[0], "checking")
        self.assertTrue(wario.playable)
        wario.probed, wario.title, wario.has_wrapper = True, "SUPERMARIOLAND", True
        self.assertEqual(wario.status()[0], "playable")      # a wrapper alone is not an env
        other = env.RomInfo("tetris", Path("ROMs/tetris.gb"), probed=True, title="TETRIS",
                            has_wrapper=False)
        self.assertEqual(other.status()[0], "playable")      # any ROM that boots: play it
        self.assertTrue(other.playable and not other.runnable)
        broken = env.RomInfo("bad", Path("ROMs/bad.gb"), probed=True, error="boom")
        self.assertEqual(broken.status()[0], "unsupported")
        self.assertFalse(broken.playable)
        self.assertFalse(wario.runnable)

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
