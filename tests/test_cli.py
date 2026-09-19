import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
