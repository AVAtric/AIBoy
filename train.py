"""Train a PPO agent on a Game Boy game using PyBoy + Stable-Baselines3."""
import argparse
import os
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from env import GAMES, env_factory


def build_vec_env(game: str, n_envs: int, seed: int, action_repeat: int, frame_stack: int):
    fns = [partial(env_factory, game, seed + i, "null", action_repeat) for i in range(n_envs)]
    vec = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    vec = VecTransposeImage(vec)
    if frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="first")
    return vec


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.zip"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on a Game Boy game")
    p.add_argument("--game", default="mario", choices=sorted(GAMES.keys()))
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=4, help="Frames each action lasts")
    p.add_argument("--frame-stack", type=int, default=4, help="Consecutive frames stacked as obs")
    p.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--run-name", default=None, help="Run name (defaults to <game>)")
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.game
    ckpt_dir = Path("checkpoints") / run_name
    log_dir = Path("logs") / run_name
    tb_dir = Path("tensorboard") / run_name
    for d in (ckpt_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[train] game={args.game} n_envs={args.n_envs} device={args.device}")
    print(f"[train] action_repeat={args.action_repeat} frame_stack={args.frame_stack}")

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
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
        )

    # eval_freq/save_freq in EvalCallback+CheckpointCallback count *policy calls*,
    # not env-steps. With n_envs, divide by n_envs to align with total env steps.
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


if __name__ == "__main__":
    main()
