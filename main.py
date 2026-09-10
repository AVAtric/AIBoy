"""Train or play a Game Boy AI agent (PPO + PyBoy).

Usage:
    python main.py gui                                                 # open Tkinter GUI
    python main.py train --game mario --n-envs 4 --timesteps 1000000   # train headless
    python main.py play  --game mario --episodes 3                     # play in SDL2 window
"""
import argparse
import os
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

from env import GAMES, env_factory, make_pyboy_env


# ------------------------- vec-env plumbing -------------------------

def build_vec_env(game, n_envs, seed, action_repeat, frame_stack):
    fns = [partial(env_factory, game, seed + i, "null", action_repeat) for i in range(n_envs)]
    vec = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    vec = VecTransposeImage(vec)
    if frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="first")
    return vec


def build_play_env(game, action_repeat, frame_stack, emulation_speed):
    def _init():
        env = make_pyboy_env(
            game=game, window_type="SDL2",
            action_repeat=action_repeat, emulation_speed=emulation_speed,
        )
        return Monitor(env)

    vec = DummyVecEnv([_init])
    vec = VecTransposeImage(vec)
    if frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="first")
    return vec


# ------------------------- helpers -------------------------

def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.zip"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def resolve_model_path(model_arg: str | None, run_name: str) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    candidates = [
        Path("logs") / run_name / "best_model.zip",
        Path("checkpoints") / run_name / "final.zip",
    ]
    ckpt_dir = Path("checkpoints") / run_name
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("ppo_*.zip"))
        if ckpts:
            candidates.append(ckpts[-1])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No trained model for run '{run_name}'. Looked in: {[str(c) for c in candidates]}"
    )


# ------------------------- train -------------------------

def cmd_train(args: argparse.Namespace) -> None:
    run_name = args.run_name or args.game
    ckpt_dir = Path("checkpoints") / run_name
    log_dir = Path("logs") / run_name
    tb_dir = Path("tensorboard") / run_name
    for d in (ckpt_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[train] game={args.game} n_envs={args.n_envs} device={args.device}")
    print(f"[train] action_repeat={args.action_repeat} frame_stack={args.frame_stack} "
          f"ent_coef={args.ent_coef} n_steps={args.n_steps}")

    env = build_vec_env(args.game, args.n_envs, args.seed, args.action_repeat, args.frame_stack)
    eval_env = build_vec_env(args.game, 1, args.seed + 10_000, args.action_repeat, args.frame_stack)

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        tb_log = str(tb_dir)
    except ImportError:
        print("[train] tensorboard not installed; skipping TB logging (pip install tensorboard)")
        tb_log = None

    resumed_from = latest_checkpoint(ckpt_dir) if args.resume else None
    if resumed_from is not None:
        print(f"[train] resuming from {resumed_from}")
        model = PPO.load(str(resumed_from), env=env, device=args.device, tensorboard_log=tb_log)
    else:
        model = PPO(
            "CnnPolicy",
            env,
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
        save_path=str(ckpt_dir),
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
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
        final_path = ckpt_dir / "final.zip"
        model.save(str(final_path))
        print(f"[train] saved final model to {final_path}")
        env.close()
        eval_env.close()


# ------------------------- play -------------------------

def cmd_play(args: argparse.Namespace) -> None:
    run_name = args.run_name or args.game
    model_path = resolve_model_path(args.model, run_name)
    print(f"[play] loading model from {model_path}")

    env = build_play_env(args.game, args.action_repeat, args.frame_stack, args.emulation_speed)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gameboy-ai",
        description="Train or play a Game Boy AI agent (PPO + PyBoy)",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="Train a PPO agent (headless)")
    _add_common(t)
    t.add_argument("--timesteps", type=int, default=1_000_000)
    t.add_argument("--n-envs", type=int, default=4, help="Parallel PyBoy instances")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    t.add_argument("--resume", action="store_true", help="Resume from newest checkpoint")
    t.add_argument("--run-name", default=None, help="Sub-directory name (defaults to <game>)")
    t.add_argument("--checkpoint-freq", type=int, default=25_000)
    t.add_argument("--eval-freq", type=int, default=10_000)
    t.add_argument("--n-eval-episodes", type=int, default=3)
    t.add_argument("--learning-rate", type=float, default=2.5e-4)
    t.add_argument("--n-steps", type=int, default=256, help="PPO rollout length per env")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--n-epochs", type=int, default=4)
    t.add_argument("--ent-coef", type=float, default=0.05,
                   help="Entropy coefficient (higher = more exploration)")

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
