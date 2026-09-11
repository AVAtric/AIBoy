# gameboyEnv

Train reinforcement-learning agents to play Game Boy games with
[PyBoy](https://github.com/Baekalfen/PyBoy) as the emulator and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) as the
learning algorithm.

Supported games (require a PyBoy game-wrapper):

| Name    | ROM file   | Cartridge title  | Notes                                                    |
|---------|------------|------------------|----------------------------------------------------------|
| `mario` | `mario.gb` | `SUPER MARIOLAN` | Custom shaped-reward MarioEnv                            |
| `kirby` | `kirby.gb` | `KIRBY DREAM LA` | PyBoy's default openai_gym                               |
| `wario` | `wario.gb` | `WARIO`          | ⚠ No PyBoy game_wrapper — unsupported for training/play |

Super Mario Land uses a custom `MarioEnv` with:

- **Held-button actions** — buttons are held for the whole `frame_skip`
  window (fixing PyBoy's default that releases every frame, which prevented
  the agent from actually walking).
- **11 Mario-tuned discrete actions**: NOOP, RIGHT, LEFT, JUMP,
  RIGHT+JUMP, RIGHT+RUN, RIGHT+RUN+JUMP, LEFT+JUMP, LEFT+RUN,
  LEFT+RUN+JUMP, DOWN (crouch / pipe entry).
- **Shaped reward** (per env-step):
  `progress_weight × new_max_x_delta` (forward-only, backtracking gives 0)
  `+ coin_weight × Δcoins`
  `+ score_weight × Δscore` (enemy kills, pipe bonuses)
  `− time_penalty` (constant per-step tax)
  `− death_penalty` (500 on death)
  `+ completion_bonus` (1000 on level clear).
- **Two observation modes** (`--obs-type`):
  - `tiles` (default, recommended) — 16×20 semantic tile grid from PyBoy's
    `custom_minimal_enemy()`, values in {-1=Mario, 0=empty, 0.5=ground,
    0.6=enemy, 1.0=pipe/wall}. Six normalised scalar features (lives,
    coins, timer, current_x, world, level) are overlaid on the top-left
    tiles so the agent knows what world it's in and how much time it has.
    Trained with `MlpPolicy`. ~10× faster than pixels on CPU.
  - `pixels` — raw 144×160×3 RGB screen. Trained with `CnnPolicy`. Slower
    but strictly more information; use it if tile learning plateaus.

## Level modes (`--start-level`)

| Mode         | Episode boundary                       | Best for                                  |
|--------------|----------------------------------------|-------------------------------------------|
| `default`    | Game over (all lives lost)             | Learning to play through the game.        |
| `random`     | Any death or clear                     | Generalising across every usable level.   |
| `sequential` | Any death or clear (retry on death)    | Reliably beating levels in order.         |
| `marathon`   | Any death (or clearing the last level) | Speedrunning all 10 usable levels.        |
| `W-L`        | Any death or clear                     | Drilling one specific level.              |

`marathon` mode **skips the in-game level-end cutscene**: the instant Mario
touches the flagpole, the next level's save-state is force-loaded, so no
training steps are wasted on the ~15 s victory sequence.

Levels 2-3 and 4-3 cannot be booted through PyBoy's SML wrapper (upstream
bug); every mode that iterates levels transparently skips them.

## Apple Silicon note

For the tile-obs MLP, **CPU is faster than MPS** by ~4× (measured on M1
Max): Metal kernel-launch overhead exceeds the compute per call. The default
`--device cpu` is intentional. MPS would only start winning with much
larger networks.

## Install

- Python 3.10 or newer.
- Game Boy ROMs in `ROMs/<name>.gb`. ROMs are copyrighted; you are
  responsible for obtaining them legally.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py                                                     # Tkinter GUI (default)
python main.py gui                                                 # same
python main.py train --game mario --n-envs 10 --timesteps 500000   # train headless (CLI)
python main.py play  --game mario --episodes 3                     # watch in SDL2 window (CLI)
```

The **GUI** is the easiest way to work with the project. Three tabs, in
workflow order (Train opens focused by default):

- **Tune** — hyperparameter search. Pick a base preset, define a sweep as
  JSON (e.g. `{"ent_coef": [0.005, 0.01, 0.02], "learning_rate": [1e-4,
  5e-4]}`), choose grid- or random-search, run trials as isolated
  subprocesses, and get a best-first sorted table. Any row can be saved
  as a preset or loaded directly into the Train tab.
- **Train** — start / stop a headless training run, see live
  `total_timesteps`, `ep_rew_mean`, `ep_len_mean`, `fps`, `time_elapsed`
  and a scrolling training log. Live game stats (world, lives, coins,
  max_x) update from the preview thread when live preview is enabled.
  Training runs as a subprocess so the GUI stays responsive.
  A **Preset** bar lets you pick a built-in like
  `Mario — Campaign, recommended (~15 min)` or save your own tweaked
  config. User presets live in `training_presets.json`; built-ins are in
  `builtin_presets.json` and can be hand-edited.
- **Play** — model selector dropdown lists every
  `models/<game>/<run>/logs/best_model.zip`,
  `.../checkpoints/final.zip`, and step snapshot. Speed selector
  supports `0.5×`, `1× (real time)`, `2×`, `4×`, `Unlimited` — pacing is
  done manually via `time.sleep()` because PyBoy's built-in speed
  doesn't pace in null-window mode. Game plays inside a 3× upscaled
  Canvas at ~60 Hz.

The Train tab has a **live-preview checkbox** (off by default for max
speed). When enabled, a preview thread loads the newest `best_model.zip`
as it's saved and plays it in the Play-tab canvas — you watch the agent
improve during training. A **progress bar** shows `total_timesteps`.

While a training run is active, every config widget on the Train tab and
every input on the Play tab is disabled to prevent mid-run edits and
double-CPU contention.

### `train` flags

| Flag                | Default | Notes                                              |
|---------------------|---------|----------------------------------------------------|
| `--game`            | mario   | `mario`, `kirby`, `wario`                          |
| `--obs-type`        | tiles   | `tiles` (fast MLP) or `pixels` (CNN, slow)         |
| `--start-level`     | default | `default`/`random`/`sequential`/`marathon`/`W-L`   |
| `--n-envs`          | 10      | Parallel PyBoy instances (SubprocVecEnv)           |
| `--timesteps`       | 500000  | Total env steps                                    |
| `--action-repeat`   | 4       | Frames each action is held                         |
| `--frame-stack`     | 4       | Consecutive frames stacked as input                |
| `--device`          | cpu     | `cpu` (recommended for tiles), `cuda`, `mps`, `auto` |
| `--resume`          | off     | Continue from newest `models/<game>/<run>/checkpoints/*.zip` |
| `--run-name`        | default | Sub-dir under `models/<game>/`                     |
| `--checkpoint-freq` | 25000   | Env-steps between checkpoints                      |
| `--eval-freq`       | 10000   | Env-steps between evaluations                      |
| `--n-eval-episodes` | 3       |                                                    |
| `--learning-rate`   | 2.5e-4  |                                                    |
| `--n-steps`         | 256     | PPO rollout length per env                         |
| `--batch-size`      | 128     | Bigger batches = fewer, larger PPO update passes   |
| `--n-epochs`        | 4       |                                                    |
| `--ent-coef`        | 0.01    | Entropy coefficient (raise for more exploration)   |

### `play` flags

| Flag                 | Default | Notes                                     |
|----------------------|---------|-------------------------------------------|
| `--game`             | mario   |                                           |
| `--model`            | auto    | Explicit path to a model `.zip`           |
| `--run-name`         | default | Used to auto-locate best/final model      |
| `--episodes`         | 3       |                                           |
| `--start-level`      | default | Same options as `train`                   |
| `--emulation-speed`  | 1       | 1 = real time, 2 = 2×, 0 = unlimited      |
| `--stochastic`       | off     | Sample from policy instead of argmax      |
| `--max-steps`        | 0       | Cap steps per episode (0 = no cap)        |

Model resolution order for `play`:
1. `--model <path>` if given
2. `models/<game>/<run>/logs/best_model.zip` (best-eval-reward)
3. `models/<game>/<run>/checkpoints/final.zip`
4. Newest `models/<game>/<run>/checkpoints/ppo_*_steps.zip`

## Artifacts (per run, grouped by game)

```
models/
├── mario/
│   ├── _level_states/                # per-level PyBoy save-states (cache)
│   │   ├── 1-1.state
│   │   ├── 1-2.state
│   │   └── ...
│   └── <run-name>/                   # e.g. `default`, `experiment-1`
│       ├── checkpoints/
│       │   ├── ppo_25000_steps.zip   # periodic snapshots
│       │   └── final.zip             # saved on exit / Ctrl-C
│       ├── logs/
│       │   ├── best_model.zip        # best eval-reward model
│       │   └── evaluations.npz       # eval history (Tune tab reads this)
│       └── tensorboard/              # TB event files
└── kirby/
    └── <run-name>/
        └── ...
```

Run names starting with `_` are treated as internal caches and are hidden
from the GUI's run-name/model dropdowns.

View training curves:

```bash
tensorboard --logdir models/mario/<run-name>/tensorboard
```

## Project layout

```
env.py                  # MarioEnv + game registry + save-state bootstrap
main.py                 # CLI: `gui`, `train`, `play` subcommands
gui.py                  # Tkinter GUI (Tune, Train, Play tabs)
presets.py              # Built-in + user-saved training presets
builtin_presets.json    # Shipped built-in presets (hand-editable)
training_presets.json   # User-saved presets (created on first save)
requirements.txt
README.md
ROMs/                   # Your ROM files (gitignored)
models/                 # All training artifacts (gitignored)
  <game>/_level_states/{W-L}.state       # save-state cache
  <game>/<run>/{checkpoints,logs,tensorboard}/
```

## Rough training expectations for Mario

Rough compute expectations at ~2500–3000 fps (tile obs, `n_envs=10`, M1
Max):

- **~20 k steps** — agent learns "RIGHT+RUN gives reward"; dies at the
  first enemy every time.
- **~150–300 k steps** — agent jumps some obstacles.
- **~1–5 M steps** — agent reliably clears world 1-1.
- **~10 M+ steps** — starts making meaningful progress across levels
  (random / sequential / marathon).

If training plateaus, try one of:
- Raise `--ent-coef` from 0.01 → 0.02–0.05 to inject exploration.
- Switch `--obs-type` from `tiles` to `pixels` (slower but strictly more
  information — helps if the tile representation is missing something
  level-specific).
- Use the Tune tab to sweep `ent_coef` × `learning_rate` × `n_steps` on
  short trials before committing to a long run.
