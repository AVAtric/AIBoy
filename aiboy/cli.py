"""AIboy's command line: train or play an agent (PPO + PyBoy), or open the app.

Usage:
    python main.py                                                     # the app (wizard opens first)
    python main.py gui                                                 # same
    python main.py train --game mario --n-envs 10 --timesteps 500000   # train headless
    python main.py play  --game mario --episodes 3                     # watch in an SDL2 window

Every `train` records its settings, evaluation history and duration in the
experience file when it ends (see experience.py), which is how the app
learns which settings it has already tried and how fast this computer is.

Two internal sub-commands run PyBoy in a throw-away process for the GUI
(`probe-rom <file>`, `level-state <rom> <world> <level> <out>`); they are what
lets the GUI stay responsive even for a ROM the emulator cannot boot.

All paths (ROMs/, models/) are relative to the data directory (the project
folder, or the folder next to the built app; see paths.py), which `main()`
makes the working directory so the commands work from anywhere. The heavy
libraries (torch, Stable-Baselines3, PyBoy) are imported only by the
commands that need them, so the GUI and the helper sub-commands start fast.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from functools import partial
from pathlib import Path

from aiboy.games import (DEFAULT_STALL_STEPS, DEFAULT_TIME_BUDGET, GAMES, OBS_TYPES, SUPPORTED_GAMES,
                   level_choices, level_state_worker, prepare_level_states, probe_rom_worker)
from aiboy.paths import DATA_DIR, FROZEN, recommended_n_envs
from aiboy.runs import (KEEP_CHECKPOINTS, cpu_count, latest_checkpoint, prune_checkpoints,
                  read_run_config, resolve_model_path, run_paths, write_run_config)


# ------------------------- vec-env plumbing -------------------------

def build_vec_env(game, n_envs, seed, action_repeat, frame_stack, obs_type, start_level=None,
                  training=False, time_budget=DEFAULT_TIME_BUDGET, stall_steps=DEFAULT_STALL_STEPS):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from aiboy.env import env_factory, wrap_vec_env
    fns = [partial(env_factory, game, seed + i, "null", action_repeat, obs_type, start_level,
                   training, time_budget, stall_steps)
           for i in range(n_envs)]
    vec = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    return wrap_vec_env(vec, obs_type, frame_stack)


def build_play_env(game, action_repeat, frame_stack, emulation_speed, obs_type, start_level=None,
                   time_budget=DEFAULT_TIME_BUDGET, stall_steps=DEFAULT_STALL_STEPS):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from aiboy.env import make_pyboy_env, wrap_vec_env

    def _init():
        env = make_pyboy_env(
            game=game, window_type="SDL2",
            action_repeat=action_repeat, emulation_speed=emulation_speed,
            obs_type=obs_type, start_level=start_level,
            time_budget=time_budget, stall_steps=stall_steps,
        )
        return Monitor(env)

    return wrap_vec_env(DummyVecEnv([_init]), obs_type, frame_stack)


def _pruning_checkpoint_callback(keep: int | None, **kwargs):
    """A CheckpointCallback that keeps only the newest `keep` step snapshots.

    Long runs otherwise accumulate hundreds of multi-megabyte zips; the
    newest few are all that resume or inspection ever needs. `best_model.zip`
    and `final.zip` live elsewhere and are never touched. Defined inside a
    function so importing this module does not import torch.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback

    class PruningCheckpointCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                prune_checkpoints(Path(self.save_path), keep)    # None = keep every snapshot
            return result

    return PruningCheckpointCallback(**kwargs)


def pick_policy(obs_type: str) -> tuple[str, dict]:
    """(policy_name, policy_kwargs) for the observation type."""
    if obs_type == "pixels":
        return "CnnPolicy", {}
    # Tiles: 16x20 grid, so a moderately sized MLP is plenty.
    return "MlpPolicy", dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))


def _tune_torch_threads(obs_type: str, device: str) -> None:
    """Tile-obs training uses a tiny MLP (1280 -> 256 -> 256 -> 11). PyTorch's
    default multithreading costs more than it gains for a network this
    small AND its threads compete with the SubprocVecEnv workers for cores.
    One thread gives ~15% more fps in practice. Pixel obs (CnnPolicy) keeps
    the default."""
    if device == "cpu" and obs_type != "pixels":
        import torch
        torch.set_num_threads(1)


def _start_level(args: argparse.Namespace) -> str | None:
    """The env's start_level spec: None for the campaign. For Mario the
    save-states it needs are created here, once, so workers only load them."""
    start_level = args.start_level if args.start_level != "default" else None
    if args.game == "mario":
        failed = prepare_level_states(start_level)
        if failed:
            print(f"[{args.mode}] warning: could not bootstrap save-states for "
                  f"{[f'{w}-{l}' for w, l in failed]}; those levels fall back to "
                  f"set_world_level + start_game (slower)")
        elif start_level is not None:
            print(f"[{args.mode}] save-states ready")
    return start_level


def _eval_start_level(start_level: str | None) -> str | None:
    """Level mode for the evaluation env.

    A fixed level is mirrored so the eval reward is per-level. Marathon is
    evaluated as a marathon from 1-1 (any death ends it, so it terminates),
    which is exactly what a marathon agent is for. Random and sequential
    evaluate on the campaign: the reward stays comparable between evals
    and a campaign episode always terminates, whereas a random /
    sequential eval episode could run for a very long time.
    """
    if start_level is not None and ("-" in start_level or start_level == "marathon"):
        return start_level
    return None


# ------------------------- train -------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    started = time.time()
    _tune_torch_threads(args.obs_type, args.device)
    if args.n_envs > cpu_count():
        # More emulator processes than cores only adds scheduling overhead
        # and memory; total emulation throughput is bounded by the cores.
        print(f"[train] n_envs={args.n_envs} exceeds the {cpu_count()} CPU cores of this "
              f"machine; using n_envs={cpu_count()}")
        args.n_envs = cpu_count()
    run_name = args.run_name or "default"
    paths = run_paths(args.game, run_name)
    if args.resume:
        # A resumed model must keep seeing the observations it was trained
        # on; the run's manifest is the authority for those settings.
        recorded = read_run_config(args.game, run_name) or {}
        for key in ("obs_type", "frame_stack", "action_repeat"):
            if key in recorded and recorded[key] != getattr(args, key):
                print(f"[train] --resume: using {key}={recorded[key]} recorded for run "
                      f"'{run_name}' instead of {getattr(args, key)}")
                setattr(args, key, recorded[key])
    for d in ("checkpoints", "logs", "tensorboard"):
        paths[d].mkdir(parents=True, exist_ok=True)

    print(f"[train] game={args.game} run={run_name} obs_type={args.obs_type}")
    print(f"[train] n_envs={args.n_envs} device={args.device}")
    print(f"[train] ent_coef={args.ent_coef} n_steps={args.n_steps} "
          f"batch={args.batch_size} lr={args.learning_rate} n_epochs={args.n_epochs} "
          f"gamma={args.gamma} gae_lambda={args.gae_lambda} clip_range={args.clip_range}")
    print(f"[train] writing to {paths['base']}")
    write_run_config(args.game, run_name, vars(args))
    rollout = args.n_steps * args.n_envs
    if rollout % args.batch_size:
        print(f"[train] note: rollout size n_steps*n_envs={rollout} is not a multiple of "
              f"batch_size={args.batch_size}; the last minibatch of every epoch is smaller")

    start_level = _start_level(args)
    env = build_vec_env(args.game, args.n_envs, args.seed,
                        args.action_repeat, args.frame_stack, args.obs_type,
                        start_level=start_level, training=True,
                        time_budget=args.time_budget, stall_steps=args.stall_steps)
    print(f"[train] attempt limits: time budget {args.time_budget or 'full timer'} of 400 units, "
          f"stall limit {args.stall_steps or 'off'}")
    if start_level == "marathon":
        print("[train] marathon: training episodes start at a random level and run forward; "
              "evaluation and playback start at 1-1")
    eval_start = _eval_start_level(start_level)
    eval_env = build_vec_env(args.game, 1, args.seed + 10_000,
                             args.action_repeat, args.frame_stack, args.obs_type,
                             start_level=eval_start,
                             time_budget=args.time_budget, stall_steps=args.stall_steps)
    n_eval_episodes = args.n_eval_episodes
    if args.game == "mario" and n_eval_episodes > 1:
        # The eval env (campaign boot or a fixed save-state) and the greedy
        # policy are both deterministic, so every eval episode replays the
        # same trajectory. One episode carries all the information; the rest
        # would only burn wall-clock time between rollouts.
        print(f"[train] evaluation is deterministic ({eval_start or 'campaign'}); "
              f"running 1 eval episode instead of {n_eval_episodes}")
        n_eval_episodes = 1

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        tb_log = str(paths["tensorboard"])
    except ImportError:
        print("[train] tensorboard not installed; skipping TB logging (pip install tensorboard)")
        tb_log = None

    resumed_from = latest_checkpoint(paths["checkpoints"]) if args.resume else None
    if args.resume and resumed_from is None:
        print(f"[train] --resume requested but no checkpoint found in "
              f"{paths['checkpoints']}; starting fresh.")
    if resumed_from is not None:
        print(f"[train] resuming from {resumed_from}")
        print("[train] WARNING: --resume loads the saved model's hyperparameters. "
              "Mutable overrides (ent_coef, learning_rate, n_epochs) are applied, "
              "but architecture / n_steps / batch_size come from the saved model.")
        model = PPO.load(str(resumed_from), env=env, device=args.device, tensorboard_log=tb_log)
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.n_epochs = args.n_epochs
    else:
        policy, policy_kwargs = pick_policy(args.obs_type)
        print(f"[train] policy={policy} policy_kwargs={policy_kwargs}")
        model = PPO(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=args.device,
            tensorboard_log=tb_log,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
        )

    # SB3 counts callback frequency in vec-env steps; the flags are env steps.
    per_env = max(1, args.n_envs)
    checkpoint_cb = _pruning_checkpoint_callback(
        keep=args.keep_checkpoints or None,     # CLI: 0 = keep all
        save_freq=max(args.checkpoint_freq // per_env, 1),
        save_path=str(paths["checkpoints"]),
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["logs"]),
        log_path=str(paths["logs"]),
        eval_freq=max(args.eval_freq // per_env, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    completed = False
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=resumed_from is None,
            progress_bar=False,
        )
        completed = True
    finally:
        final_path = paths["checkpoints"] / "final.zip"
        model.save(str(final_path))
        print(f"[train] saved final model to {final_path}")
        if completed:
            # The run reached its target: final.zip is the last state, so
            # the step snapshots are redundant. Interrupted runs keep theirs
            # for --resume.
            removed = prune_checkpoints(paths["checkpoints"], keep=0)
            if removed:
                print(f"[train] removed {len(removed)} step snapshot(s); final.zip supersedes them")
        env.close()
        eval_env.close()
        _remember_run(args, run_name, paths, time.time() - started, completed,
                      resumed=resumed_from is not None)


def _remember_run(args: argparse.Namespace, run_name: str, paths: dict, duration: float,
                  completed: bool, *, resumed: bool) -> None:
    """Append this run to the experience file: what was trained, how it
    scored over time and how long it took. Never fails the run."""
    try:
        from aiboy import experience
        evals = experience.tuning.evals_from_npz(paths["logs"] / "evaluations.npz") or []
        record = experience.make_record(vars(args), evals, duration, completed=completed,
                                        source=args.source, run_name=run_name, resumed=resumed)
        experience.append_record(record)
        print(f"[train] remembered as {record.kind} {record.id} "
              f"({'complete' if record.complete else 'incomplete'}, "
              f"{len(evals)} evaluations, {duration:.0f} s)")
    except Exception as e:                                   # noqa: BLE001
        print(f"[train] could not record the run in the experience file: {e}")


# ------------------------- play -------------------------

def cmd_play(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    run_name = args.run_name or "default"
    model_path = resolve_model_path(args.model, args.game, run_name)
    print(f"[play] loading model from {model_path}")

    start_level = _start_level(args)
    env = build_play_env(args.game, args.action_repeat, args.frame_stack,
                         args.emulation_speed, args.obs_type,
                         start_level=start_level,
                         time_budget=args.time_budget, stall_steps=args.stall_steps)
    model = PPO.load(str(model_path), env=env, device=args.device)
    deterministic = not args.stochastic
    print(f"[play] {args.episodes} episode(s), deterministic={deterministic}")

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = [False]
            total = 0.0
            steps = 0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, _info = env.step(action)
                total += float(reward[0])
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
            print(f"[play] episode {ep + 1}: reward={total:.1f} steps={steps}")
    finally:
        env.close()


# ------------------------- CLI -------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--game", default="mario", choices=sorted(GAMES),
                   help=f"Supported: {', '.join(SUPPORTED_GAMES)}. Other titles are "
                        f"experimental (PyBoy's generic wrapper, pixels only, no level modes).")
    p.add_argument("--action-repeat", type=int, default=4, help="Frames each action is held")
    p.add_argument("--frame-stack", type=int, default=4, help="Consecutive frames stacked as obs")
    p.add_argument("--obs-type", default="tiles", choices=list(OBS_TYPES),
                   help="tiles = 16x20 tile grid + HUD + power-up (fast, MLP; default); "
                        "pixels = 144x160x3 RGB (slow, CNN)")
    p.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET,
                   help="Timer units (of 400) an attempt may use before it is truncated "
                        "(0 = the whole in-game timer)")
    p.add_argument("--stall-steps", type=int, default=DEFAULT_STALL_STEPS,
                   help="Steps without new progress before an attempt is truncated (0 = off)")
    p.add_argument("--start-level", default="default", choices=level_choices(),
                   help="Level mode: 'default' (campaign, respect lives), "
                        "'random' (new random level per episode), "
                        "'sequential' (advance level on clear, retry on death), "
                        "'marathon' (one episode = all 10 levels), "
                        "or a specific level 'W-L' (fixed).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiboy",
        description="AIboy: train or play a Game Boy agent (PPO + PyBoy), or open the app",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="Train a PPO agent (headless)")
    _add_common(t)
    t.add_argument("--timesteps", type=int, default=500_000)
    t.add_argument("--n-envs", type=int, default=recommended_n_envs(),
                   help="Parallel PyBoy instances (default: one per CPU core, max 12)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu",
                   help="cpu (recommended for tile obs), cuda, mps, or auto")
    t.add_argument("--resume", action="store_true", help="Resume from newest checkpoint")
    t.add_argument("--run-name", default=None, help="Sub-dir name (defaults to 'default')")
    t.add_argument("--checkpoint-freq", type=int, default=25_000)
    t.add_argument("--keep-checkpoints", type=int, default=KEEP_CHECKPOINTS,
                   help="Step snapshots to keep per run; older ones are deleted (0 = keep all)")
    t.add_argument("--eval-freq", type=int, default=10_000)
    t.add_argument("--n-eval-episodes", type=int, default=3)
    t.add_argument("--learning-rate", type=float, default=2.5e-4)
    t.add_argument("--n-steps", type=int, default=256, help="PPO rollout length per env")
    t.add_argument("--batch-size", type=int, default=128,
                   help="Bigger batches = fewer, larger PPO update passes (faster wall-clock)")
    t.add_argument("--n-epochs", type=int, default=4)
    t.add_argument("--ent-coef", type=float, default=0.01,
                   help="Entropy coefficient (raise for more exploration)")
    t.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    t.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    t.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    t.add_argument("--source", default="cli", choices=("cli", "train", "tune", "wizard"),
                   help="Who started the run; recorded in the experience file "
                        "(tune / wizard runs are short search trials)")

    sub.add_parser("gui", help="Open the AIboy app (wizard, train, tune, presets, experience)")

    # Internal helpers the GUI runs in a throw-away process (see games.py).
    pr = sub.add_parser("probe-rom", help=argparse.SUPPRESS)
    pr.add_argument("rom")
    ls = sub.add_parser("level-state", help=argparse.SUPPRESS)
    ls.add_argument("rom")
    ls.add_argument("world", type=int)
    ls.add_argument("level", type=int)
    ls.add_argument("out")

    p = sub.add_parser("play", help="Watch a trained agent in an SDL2 window")
    _add_common(p)
    p.add_argument("--model", default=None, help="Explicit path to a model .zip")
    p.add_argument("--run-name", default=None, help="Run name to auto-resolve model")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--emulation-speed", type=int, default=1, help="1=real time, 2=2x, 0=unlimited")
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    p.add_argument("--max-steps", type=int, default=0, help="Cap steps per episode (0=no cap)")
    p.add_argument("--device", default="cpu")

    return parser


def _prepare_process() -> None:
    """Make the process usable whichever way it was started: as a script,
    as the frozen app double-clicked in Finder (no terminal: stdout and
    stderr are None) or as a child the GUI reads line by line."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            setattr(sys, name, open(os.devnull, "w"))
        elif FROZEN:
            try:
                stream.reconfigure(line_buffering=True)   # the frozen app has no `-u`
            except (AttributeError, ValueError):
                pass
    os.chdir(DATA_DIR)


def main(argv: list[str] | None = None) -> None:
    # Must come first: in a frozen build the emulator workers are started
    # by re-running this executable, and this is what turns such a start
    # into a worker instead of another copy of the program.
    multiprocessing.freeze_support()
    _prepare_process()
    argv = sys.argv[1:] if argv is None else argv
    # `python main.py` with no arguments opens the GUI.
    if not argv:
        from aiboy.gui.app import run as run_gui
        run_gui()
        return
    args = build_parser().parse_args(argv)
    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "play":
        cmd_play(args)
    elif args.mode == "gui":
        from aiboy.gui.app import run as run_gui
        run_gui()
    elif args.mode == "probe-rom":
        probe_rom_worker(args.rom)
    elif args.mode == "level-state":
        level_state_worker(args.rom, args.world, args.level, args.out)


if __name__ == "__main__":
    main()
