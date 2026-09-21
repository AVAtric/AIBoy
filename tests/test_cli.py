import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

from aiboy import cli


class TensorboardCommandTests(unittest.TestCase):
    """The TensorBoard button runs the program's own `tensorboard` sub-command
    (the bundled copy in a release), never a `tensorboard` found on PATH."""

    def test_parser_accepts_the_sub_command(self):
        args = cli.build_parser().parse_args(["tensorboard", "--logdir", "models/mario", "--port", "6099"])
        self.assertEqual((args.mode, args.logdir, args.port), ("tensorboard", "models/mario", 6099))

    def test_the_port_defaults_to_tensorboards_own(self):
        self.assertEqual(cli.build_parser().parse_args(["tensorboard", "--logdir", "x"]).port, 6006)

    def test_missing_package_is_a_plain_message(self):
        args = cli.build_parser().parse_args(["tensorboard", "--logdir", "x"])
        with mock.patch.dict(sys.modules, {"tensorboard": None, "tensorboard.main": None}):
            with self.assertRaises(SystemExit) as cm:
                cli.cmd_tensorboard(args)
        self.assertIn("pip install tensorboard", str(cm.exception))

    def test_tensorboard_gets_its_flags_through_argv(self):
        args = cli.build_parser().parse_args(["tensorboard", "--logdir", "/x/models", "--port", "7007"])
        seen = {}
        fake = mock.MagicMock()
        fake.run_main = lambda: seen.setdefault("argv", list(sys.argv))
        with mock.patch.dict(sys.modules, {"tensorboard": mock.MagicMock(), "tensorboard.main": fake}):
            with mock.patch.object(sys, "argv", ["main.py"]):
                cli.cmd_tensorboard(args)
        self.assertEqual(seen["argv"], ["tensorboard", "--logdir=/x/models", "--port=7007",
                                        "--host=localhost", "--load_fast=false"])



class ExplorationGuardTests(unittest.TestCase):
    """The collapse detector and the repair it triggers (see cli.TARGET_KL)."""

    def test_is_collapsed_needs_identical_episodes_below_the_best(self):
        same = [(-128.3, 11)] * cli.COLLAPSE_EPISODES
        self.assertTrue(cli.is_collapsed(same, 10326.0))
        self.assertFalse(cli.is_collapsed(same, float("-inf")))          # no evaluation yet
        self.assertFalse(cli.is_collapsed(same, -128.3))                  # identical AND the best: solved
        self.assertFalse(cli.is_collapsed(same[:-1] + [(-100.0, 11)], 10326.0))
        self.assertFalse(cli.is_collapsed(same[:-1] + [(-128.3, 12)], 10326.0))
        self.assertFalse(cli.is_collapsed(same[:5], 10326.0))             # too few episodes
        # Only the newest episodes count: an old different one is history.
        self.assertTrue(cli.is_collapsed([(500.0, 300)] + same, 10326.0))

    @unittest.skipUnless(find_spec("stable_baselines3"), "needs Stable-Baselines3")
    def test_guard_restores_the_best_model_raises_curiosity_then_gives_up(self):
        import contextlib
        import io
        import tempfile
        import types
        import torch as th
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as tmp:
            best = Path(tmp) / "best_model.zip"
            model = PPO("MlpPolicy", "CartPole-v1", n_steps=32, batch_size=32, n_epochs=1,
                        ent_coef=0.005, seed=0, verbose=0, device="cpu")
            model.save(best)
            best_params = {k: v.clone() for k, v in model.policy.state_dict().items()}
            model.learn(64)                       # fills the rollout buffer, moves the weights
            self.assertFalse(all(th.equal(v, best_params[k])
                                 for k, v in model.policy.state_dict().items()))
            eval_cb = types.SimpleNamespace(best_mean_reward=500.0)
            guard = cli._exploration_guard(eval_cb, best)
            guard.init_callback(model)
            self.assertGreater(guard.policy_entropy(), 0.1)     # a real, exploring policy
            guard.policy_entropy = lambda: 0.0                  # now pretend it collapsed

            def collapsed_rollouts(n):
                model.ep_info_buffer.clear()
                model.ep_info_buffer.extend({"r": -128.3, "l": 11, "t": 0.0}
                                            for _ in range(cli.COLLAPSE_EPISODES))
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    for _ in range(n):
                        guard.on_rollout_end()
                return out.getvalue()

            self.assertEqual(collapsed_rollouts(cli.COLLAPSE_PATIENCE - 1), "")   # patience
            self.assertEqual(guard.repairs, 0)
            said = collapsed_rollouts(1)
            self.assertEqual(guard.repairs, 1)
            self.assertIn("[train] warning: the agent stopped exploring", said)
            self.assertIn("Restored the best model", said)
            self.assertEqual(model.ent_coef, 0.02)                           # 0.005 x4, at least 0.02
            self.assertTrue(all(th.equal(v, best_params[k])
                                for k, v in model.policy.state_dict().items()))
            self.assertEqual(len(model.ep_info_buffer), 0)                   # old episodes forgotten
            self.assertTrue(guard.on_step())                                 # training goes on
            # During the cooldown nothing happens, however bad it looks.
            self.assertEqual(collapsed_rollouts(cli.GUARD_COOLDOWN), "")
            self.assertEqual(guard.repairs, 1)
            # Every further collapse is repaired, with more curiosity, up to the limit.
            collapsed_rollouts(cli.COLLAPSE_PATIENCE)
            self.assertEqual(guard.repairs, 2)
            self.assertEqual(model.ent_coef, 0.08)
            collapsed_rollouts(cli.GUARD_COOLDOWN + cli.COLLAPSE_PATIENCE)
            self.assertEqual(guard.repairs, 3)
            self.assertEqual(model.ent_coef, 0.1)                            # capped
            said = collapsed_rollouts(cli.GUARD_COOLDOWN + cli.COLLAPSE_PATIENCE)
            self.assertEqual(guard.repairs, 3)
            self.assertIn("training stops here", said)
            self.assertFalse(guard.on_step())

    @unittest.skipUnless(find_spec("stable_baselines3"), "needs Stable-Baselines3")
    def test_resume_overrides_rebuild_the_learning_rate_schedule(self):
        import types
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", "CartPole-v1", n_steps=32, batch_size=32, learning_rate=1e-4,
                    ent_coef=0.005, n_epochs=4, verbose=0, device="cpu")
        self.assertEqual(model.lr_schedule(1.0), 1e-4)
        args = types.SimpleNamespace(ent_coef=0.02, learning_rate=3e-4, n_epochs=2)
        cli._apply_resume_overrides(model, args)
        self.assertEqual(model.lr_schedule(1.0), 3e-4)      # took effect now, not next resume
        self.assertEqual(model.learning_rate, 3e-4)
        self.assertEqual(model.ent_coef, 0.02)
        self.assertEqual(model.n_epochs, 2)
        self.assertEqual(model.target_kl, cli.TARGET_KL)


if __name__ == "__main__":
    unittest.main()
