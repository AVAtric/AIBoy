"""Watch a trained PPO agent play a Game Boy game in an SDL2 window."""
import argparse
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from env import GAMES, env_factory


def build_play_env(game: str, action_repeat: int, frame_stack: int, emulation_speed: int):
    def _init():
        from env import make_pyboy_env
        from stable_baselines3.common.monitor import Monitor

        env = make_pyboy_env(
            game=game,
            window_type="SDL2",
            action_repeat=action_repeat,
            emulation_speed=emulation_speed,
        )
        return Monitor(env)

    vec = DummyVecEnv([_init])
    vec = VecTransposeImage(vec)
    if frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="first")
    return vec


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained PPO agent play")
    p.add_argument("--game", default="mario", choices=sorted(GAMES.keys()))
    p.add_argument("--model", default=None, help="Explicit path to a model .zip")
    p.add_argument("--run-name", default=None, help="Run name to auto-resolve model")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--emulation-speed", type=int, default=1,
                   help="1=real time, 2=2x, 0=unlimited")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from policy instead of using argmax")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Cap steps per episode (0 = no cap, run until done)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
