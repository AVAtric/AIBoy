"""Train or play a Game Boy AI agent (PPO + PyBoy).

Usage:
    python main.py gui                                                 # open Tkinter GUI
    python main.py train --game mario --n-envs 8 --timesteps 500000    # train headless
    python main.py play  --game mario --episodes 3                     # play in SDL2 window
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from env import GAMES, SML_ALL_LEVELS, env_factory, ensure_level_states, make_pyboy_env


MODELS_ROOT = Path("models")


# ------------------------- paths -------------------------

def run_paths(game: str, run_name: str) -> dict[str, Path]:
    """models/<game>/<run>/{checkpoints,logs,tensorboard}/"""
    base = MODELS_ROOT / game / run_name
    return {
        "base": base,
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
        "tensorboard": base / "tensorboard",
    }


# ------------------------- vec-env plumbing -------------------------

def build_vec_env(game, n_envs, seed, action_repeat, frame_stack, obs_type, start_level=None):
    fns = [partial(env_factory, game, seed + i, "null", action_repeat, obs_type, start_level)
           for i in range(n_envs)]
    vec = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    if obs_type == "pixels":
        vec = VecTransposeImage(vec)
    if frame_stack > 1:
        # For pixels (transposed to CHW) use channels_order="first"
        # For tiles (float32 HWC), use "last" to stack along channel dim
        order = "first" if obs_type == "pixels" else "last"
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order=order)
    return vec


def build_play_env(game, action_repeat, frame_stack, emulation_speed, obs_type, start_level=None):
    def _init():
        env = make_pyboy_env(
            game=game, window_type="SDL2",
            action_repeat=action_repeat, emulation_speed=emulation_speed,
            obs_type=obs_type, start_level=start_level,
        )
        return Monitor(env)

    vec = DummyVecEnv([_init])
    if obs_type == "pixels":
        vec = VecTransposeImage(vec)
    if frame_stack > 1:
        order = "first" if obs_type == "pixels" else "last"
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order=order)
    return vec


# ------------------------- helpers -------------------------

def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.zip"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def resolve_model_path(model_arg: str | None, game: str, run_name: str) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    paths = run_paths(game, run_name)
    candidates = [
        paths["logs"] / "best_model.zip",
        paths["checkpoints"] / "final.zip",
    ]
    if paths["checkpoints"].exists():
        ckpts = sorted(paths["checkpoints"].glob("ppo_*.zip"))
        if ckpts:
            candidates.append(ckpts[-1])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No trained model for game='{game}' run='{run_name}'. "
        f"Looked in: {[str(c) for c in candidates]}"
    )


def pick_policy(obs_type: str) -> tuple[str, dict]:
    """Return (policy_name, policy_kwargs) for the given observation type."""
    if obs_type == "pixels":
        return "CnnPolicy", {}
    # Tiles → MLP, small state so a moderately-sized net is enough
    return "MlpPolicy", dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))


# ------------------------- train -------------------------

def _tune_torch_threads(obs_type: str, device: str) -> None:
    """For tile-obs training the policy is a tiny MLP (1280 → 256 → 256 → 11).
    PyTorch's default multithreading has more overhead than compute for a
    network this small AND its threads compete for CPU cores with the
    SubprocVecEnv workers. Setting it to 1 thread gives ~15% more fps in
    practice. For pixel obs (larger CnnPolicy) we keep the default.
    """
    if device != "cpu":
        return  # GPU/MPS compute doesn't touch CPU threads
    if obs_type == "tiles":
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass


def cmd_train(args: argparse.Namespace) -> None:
    _tune_torch_threads(args.obs_type, args.device)
    run_name = args.run_name or "default"
    paths = run_paths(args.game, run_name)
    for d in ("checkpoints", "logs", "tensorboard"):
        paths[d].mkdir(parents=True, exist_ok=True)

    print(f"[train] game={args.game} run={run_name} obs_type={args.obs_type}")
    print(f"[train] n_envs={args.n_envs} device={args.device}")
    print(f"[train] ent_coef={args.ent_coef} n_steps={args.n_steps} "
          f"batch={args.batch_size} lr={args.learning_rate}")
    print(f"[train] writing to {paths['base']}")

    start_level = args.start_level if args.start_level != "default" else None
    # Pre-bootstrap per-level save-states so SubprocVecEnv workers just load
    # them (instant, race-free) instead of each running set_world_level +
    # start_game, which is fragile and can hang.
    if args.game == "mario" and start_level is not None:
        rom = Path("ROMs") / "mario.gb"
        if start_level in ("random", "sequential", "marathon"):
            targets = SML_ALL_LEVELS
        else:
            targets = [tuple(int(x) for x in start_level.split("-"))]
        needed = [(w, l) for (w, l) in targets
                  if not (Path("models") / "mario" / "_level_states" / f"{w}-{l}.state").exists()]
        if needed:
            print(f"[train] bootstrapping save-states for {len(needed)} level(s): "
                  f"{[f'{w}-{l}' for w, l in needed]}...")
            ok = ensure_level_states(rom, needed)
            failed = [t for t in needed if t not in ok]
            if failed:
                print(f"[train] warning: could not bootstrap {failed} — falling back to old path")
            else:
                print(f"[train] all requested level states cached")
    env = build_vec_env(args.game, args.n_envs, args.seed,
                        args.action_repeat, args.frame_stack, args.obs_type,
                        start_level=start_level)
    # Eval env selection:
    #  - Fixed level ("W-L"): mirror it for consistent per-level eval.
    #  - Every other mode (default, random, sequential, marathon): use
    #    campaign (None). Reasons: eval reward is comparable across evals,
    #    and campaign guarantees the episode terminates on game_over —
    #    unlike random/sequential/marathon where an eval episode could
    #    run indefinitely and block `evaluate_policy` (was the source of
    #    the "hangs at ~100k timesteps" bug in marathon mode).
    if start_level is not None and isinstance(start_level, str) and "-" in start_level:
        eval_start = start_level
    else:
        eval_start = None
    eval_env = build_vec_env(args.game, 1, args.seed + 10_000,
                             args.action_repeat, args.frame_stack, args.obs_type,
                             start_level=eval_start)

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        tb_log = str(paths["tensorboard"])
    except ImportError:
        print("[train] tensorboard not installed; skipping TB logging (pip install tensorboard)")
        tb_log = None

    resumed_from = latest_checkpoint(paths["checkpoints"]) if args.resume else None
    if args.resume and resumed_from is None:
        print(f"[train] --resume requested but no checkpoint found in "
              f"{paths['checkpoints']} — starting fresh.")
    if resumed_from is not None:
        print(f"[train] resuming from {resumed_from}")
        print(f"[train] WARNING: --resume loads the saved model's hyperparameters. "
              f"Mutable overrides (ent_coef, learning_rate, n_epochs) will be applied, "
              f"but architecture / n_steps / batch_size come from the saved model.")
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
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
        )

    per_env = max(1, args.n_envs)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // per_env, 1),
        save_path=str(paths["checkpoints"]),
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["logs"]),
        log_path=str(paths["logs"]),
        eval_freq=max(args.eval_freq // per_env, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=resumed_from is None,
            progress_bar=False,
        )
    finally:
        final_path = paths["checkpoints"] / "final.zip"
        model.save(str(final_path))
        print(f"[train] saved final model to {final_path}")
        env.close()
        eval_env.close()


# ------------------------- play -------------------------

def cmd_play(args: argparse.Namespace) -> None:
    run_name = args.run_name or "default"
    model_path = resolve_model_path(args.model, args.game, run_name)
    print(f"[play] loading model from {model_path}")

    start_level = args.start_level if args.start_level != "default" else None
    if args.game == "mario" and start_level is not None:
        rom = Path("ROMs") / "mario.gb"
        if start_level in ("random", "sequential", "marathon"):
            ensure_level_states(rom)
        else:
            ensure_level_states(rom, [tuple(int(x) for x in start_level.split("-"))])
    env = build_play_env(args.game, args.action_repeat, args.frame_stack,
                         args.emulation_speed, args.obs_type,
                         start_level=start_level)
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
    p.add_argument("--game", default="mario", choices=sorted(GAMES.keys()))
    p.add_argument("--action-repeat", type=int, default=4, help="Frames each action is held")
    p.add_argument("--frame-stack", type=int, default=4, help="Consecutive frames stacked as obs")
    p.add_argument("--obs-type", default="tiles", choices=["pixels", "tiles"],
                   help="tiles = 16×20 game_area (fast, MLP); pixels = 144×160×3 RGB (slow, CNN)")
    level_choices = (["default", "random", "sequential", "marathon"]
                     + [f"{w}-{l}" for (w, l) in SML_ALL_LEVELS])
    p.add_argument("--start-level", default="default", choices=level_choices,
                   help="Level mode: 'default' (campaign, respect lives), "
                        "'random' (new random level per episode), "
                        "'sequential' (advance level on clear, retry on death), "
                        "'marathon' (one episode = all 10 levels), "
                        "or a specific level 'W-L' (fixed).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gameboy-ai",
        description="Train or play a Game Boy AI agent (PPO + PyBoy)",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="Train a PPO agent (headless)")
    _add_common(t)
    t.add_argument("--timesteps", type=int, default=500_000)
    t.add_argument("--n-envs", type=int, default=10,
                   help="Parallel PyBoy instances (10 uses all M1 cores)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu",
                   help="cpu (recommended for tile obs), cuda, mps, or auto")
    t.add_argument("--resume", action="store_true", help="Resume from newest checkpoint")
    t.add_argument("--run-name", default=None, help="Sub-dir name (defaults to 'default')")
    t.add_argument("--checkpoint-freq", type=int, default=25_000)
    t.add_argument("--eval-freq", type=int, default=10_000)
    t.add_argument("--n-eval-episodes", type=int, default=3)
    t.add_argument("--learning-rate", type=float, default=2.5e-4)
    t.add_argument("--n-steps", type=int, default=256, help="PPO rollout length per env")
    t.add_argument("--batch-size", type=int, default=128,
                   help="Bigger batches = fewer, larger PPO update passes (faster wall-clock)")
    t.add_argument("--n-epochs", type=int, default=4)
    t.add_argument("--ent-coef", type=float, default=0.01,
                   help="Entropy coefficient (raise for more exploration)")

    sub.add_parser("gui", help="Launch the Tkinter GUI (train + play in one window)")

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


def main() -> None:
    # `python main.py` with no args opens the GUI (same as `python main.py gui`).
    # Any subcommand is still parsed normally.
    if len(sys.argv) == 1:
        from gui import run as run_gui
        run_gui()
        return
    args = build_parser().parse_args()
    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "play":
        cmd_play(args)
    elif args.mode == "gui":
        from gui import run as run_gui
        run_gui()


if __name__ == "__main__":
    main()
