# gameboyEnv

Train reinforcement-learning agents to play Game Boy games with
[PyBoy](https://github.com/Baekalfen/PyBoy) as the emulator and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) as the
learning algorithm.

Supported games (require a PyBoy game-wrapper):

| Name     | ROM file    | Cartridge title  |
|----------|-------------|------------------|
| `mario`  | `mario.gb`  | `SUPER MARIOLAN` |
| `kirby`  | `kirby.gb`  | `KIRBY DREAM LA` |
| `tetris` | `tetris.gb` | `TETRIS`         |

Super Mario Land uses a custom `MarioEnv` with:

- **Held-button actions** — buttons are held for the whole `frame_skip`
  window (fixing PyBoy's default that releases every frame, which prevented
  the agent from actually walking).
- **7 Mario-tuned discrete actions**: NOOP, RIGHT, LEFT, JUMP, RIGHT+JUMP,
  RIGHT+RUN, RIGHT+RUN+JUMP.
- **Shaped reward**: `dx + explore_bonus*(x-max_x) + score/50 − time
  − death_penalty (200 on death) + completion_bonus (1000 on level clear)`.
- **Episode termination** on death or level clear; truncation when Mario
  makes no forward progress for `stuck_steps` steps.

## Prerequisites

- Python 3.10 or newer.
- Game Boy ROMs in `ROMs/<name>.gb`. ROMs are copyrighted; you are
  responsible for obtaining them legally.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, PyTorch uses the `mps` backend via `--device auto`.

## Usage

```bash
python main.py gui                                                  # Tkinter GUI (recommended)
python main.py train --game mario --n-envs 4 --timesteps 1000000    # train headless (CLI)
python main.py play  --game mario --episodes 3                      # watch in SDL2 window (CLI)
```

The **GUI** is the easiest way to work with the project. It has two tabs:

- **Train** — start / stop a headless training run, see live `total_timesteps`,
  `ep_rew_mean`, `ep_len_mean`, `fps` and a scrolling training log. Training
  runs as a subprocess of the CLI so the GUI stays responsive.
- **Play** — load the best model for the selected run and watch the agent
  play, embedded in a 3× upscaled Canvas inside the GUI window. This avoids
  the macOS "hidden window" issue you may hit when running the SDL2 play
  mode from a nested shell.

### `train` flags

| Flag                | Default | Notes                                           |
|---------------------|---------|-------------------------------------------------|
| `--game`            | mario   | `mario`, `kirby`, `tetris`                      |
| `--n-envs`          | 4       | Parallel PyBoy instances (SubprocVecEnv)        |
| `--timesteps`       | 1000000 | Total env steps                                 |
| `--action-repeat`   | 4       | Frames each action is held                      |
| `--frame-stack`     | 4       | Consecutive frames stacked as CNN input         |
| `--device`          | auto    | `cpu`, `cuda`, `mps`, or `auto`                 |
| `--resume`          | off     | Continue from newest `checkpoints/<run>/*.zip`  |
| `--run-name`        | `<game>`| Sub-directory under checkpoints/logs/tensorboard|
| `--checkpoint-freq` | 25000   | Env-steps between checkpoints                   |
| `--eval-freq`       | 10000   | Env-steps between evaluations                   |
| `--n-eval-episodes` | 3       |                                                 |
| `--learning-rate`   | 2.5e-4  |                                                 |
| `--n-steps`         | 256     | PPO rollout length per env                      |
| `--batch-size`      | 256     |                                                 |
| `--n-epochs`        | 4       |                                                 |
| `--ent-coef`        | 0.05    | Entropy coefficient (raise for more exploration)|

### `play` flags

| Flag                 | Default | Notes                                     |
|----------------------|---------|-------------------------------------------|
| `--game`             | mario   |                                           |
| `--model`            | auto    | Explicit path to a model `.zip`           |
| `--run-name`         | `<game>`| Used to auto-locate best/final model      |
| `--episodes`         | 3       |                                           |
| `--emulation-speed`  | 1       | 1 = real time, 2 = 2x, 0 = unlimited      |
| `--stochastic`       | off     | Sample from policy instead of argmax      |
| `--max-steps`        | 0       | Cap steps per episode (0 = no cap)        |

Model resolution order for `play`:
1. `--model <path>` if given
2. `logs/<run-name>/best_model.zip` (best-eval-reward)
3. `checkpoints/<run-name>/final.zip`
4. Newest `checkpoints/<run-name>/ppo_*_steps.zip`

## Artifacts (per run)

```
checkpoints/<run>/ppo_*_steps.zip     # periodic snapshots
checkpoints/<run>/final.zip           # saved on exit
logs/<run>/best_model.zip             # best-eval model
logs/<run>/evaluations.npz            # eval history
tensorboard/<run>/                    # TB event files
```

View training curves:

```bash
tensorboard --logdir tensorboard
```

## Project layout

```
env.py            # MarioEnv + game registry + ActionRepeat wrapper
main.py           # CLI: `gui`, `train`, `play` subcommands
gui.py            # Tkinter GUI (embedded game view, subprocess-based training)
requirements.txt
README.md
ROMs/             # Your ROM files (gitignored)
checkpoints/, logs/, tensorboard/     # Training artifacts (gitignored)
```

## Notes on training Mario

Mario needs meaningful compute. Rough expectations:

- **~20k steps** — agent learns "RIGHT+RUN gives reward"; dies at the first
  enemy every time.
- **~150k–300k steps** — agent learns to jump some obstacles.
- **~1M–5M steps** — agent reliably clears world 1-1.

The default hyperparameters (`ent_coef=0.05`, `n_steps=256`, `batch_size=256`,
`clip_range=0.2`) are tuned for a smallish CNN + platformer reward shape.
If training plateaus, try lowering `--ent-coef` to 0.01 to reduce exploration
noise once the agent has a decent policy.
