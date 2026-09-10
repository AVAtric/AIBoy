# gameboyEnv

Train reinforcement-learning agents to play Game Boy games using
[PyBoy](https://github.com/Baekalfen/PyBoy) as the emulator/gym environment and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) as the RL library.

Supported games (require a game-wrapper in PyBoy):

| Name     | ROM file    | Cartridge title  |
|----------|-------------|------------------|
| `mario`  | `mario.gb`  | `SUPER MARIOLAN` |
| `kirby`  | `kirby.gb`  | `KIRBY DREAM LA` |
| `tetris` | `tetris.gb` | `TETRIS`         |

## Prerequisites

- Python 3.10 or newer
- A copy of the Game Boy ROM you want to train on, placed in `ROMs/<name>.gb`.
  ROMs are copyrighted; you are responsible for obtaining them legally.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

On Apple Silicon, PyTorch will use the `mps` backend automatically via `--device auto`.

## Train

```bash
python train.py --game mario --n-envs 4 --timesteps 1000000
```

Useful flags:

| Flag                | Default | Notes                                          |
|---------------------|---------|------------------------------------------------|
| `--game`            | mario   | `mario`, `kirby`, or `tetris`                  |
| `--n-envs`          | 4       | Parallel PyBoy instances (SubprocVecEnv)       |
| `--timesteps`       | 1000000 | Total env steps across all envs                |
| `--action-repeat`   | 4       | Each policy action is held for K frames        |
| `--frame-stack`     | 4       | Number of consecutive frames fed to the CNN   |
| `--device`          | auto    | `cpu`, `cuda`, `mps`, or `auto`                |
| `--resume`          | off     | Continue from the newest file in `checkpoints/<run>/` |
| `--run-name`        | `<game>`| Sub-directory under checkpoints/logs/tensorboard |
| `--checkpoint-freq` | 25000   | Env-steps between checkpoints                  |
| `--eval-freq`       | 10000   | Env-steps between evaluations                  |

Artifacts (per run):

```
checkpoints/<run>/ppo_*_steps.zip     # periodic snapshots
checkpoints/<run>/final.zip           # saved on exit
logs/<run>/best_model.zip             # best eval-reward model
logs/<run>/evaluations.npz            # eval history
tensorboard/<run>/                    # TB event files
```

Watch training curves:

```bash
tensorboard --logdir tensorboard
```

## Play

Watch a trained agent play in a real Game Boy window:

```bash
python play.py --game mario --episodes 3
```

By default `play.py` looks for `logs/<game>/best_model.zip`, then
`checkpoints/<game>/final.zip`, then the latest step-checkpoint. Override with
`--model path/to/model.zip` or `--run-name <name>`.

Useful flags:

| Flag                 | Default | Notes                                     |
|----------------------|---------|-------------------------------------------|
| `--episodes`         | 3       | Episodes to play                          |
| `--emulation-speed`  | 1       | 1 = real-time, 2 = 2x, 0 = unlimited      |
| `--stochastic`       | off     | Sample from policy instead of argmax      |

## Project layout

```
env.py            # PyBoy env factory + ActionRepeat wrapper
train.py          # Training CLI (PPO + CnnPolicy, frame stacking, TB, resume)
play.py           # Visualization CLI (SDL2 window)
requirements.txt
README.md
ROMs/             # Your ROM files (gitignored)
checkpoints/      # Training checkpoints (gitignored)
logs/             # Eval logs / best models (gitignored)
tensorboard/      # TB event files (gitignored)
```

## Notes on hyperparameters

Defaults follow the Atari-style PPO recipe: `n_steps=128`, `batch_size=256`,
`n_epochs=4`, `clip_range=0.1`, `ent_coef=0.01`, action-repeat 4, frame-stack 4.
These work reasonably as a starting point; expect to tune for each game.

Super Mario Land typically needs several million steps of experience before an
agent reliably clears the first level. Start small (~200k steps) to confirm the
pipeline works, then scale up.
