# gameboyEnv — teach an AI to play Super Mario Land

Train a reinforcement-learning agent to play **Super Mario Land** on an
emulated Game Boy, then watch it play. [PyBoy](https://github.com/Baekalfen/PyBoy)
is the emulator, [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
PPO is the learner, and a Tkinter desktop app wraps the whole workflow:

```
Wizard:  1 Tune  →  2 Preset  →  3 Train  →  4 Watch
```

The window has two halves: **workflow tabs on the left, the Game Boy screen
on the right**. Whatever produces something to watch (a finished model, the
live preview of a training run, the wizard's final step) shows on that
screen without switching tabs.

- **Wizard** — a guided path: compare hyperparameters on short trials, save
  the winner as a preset, run the real training, watch the result. No
  knowledge of PPO needed.
- **Train** — one headless training run with live stats, a progress bar,
  the full log, TensorBoard, resume, and an optional live preview.
- **Tune** — hyperparameter sweeps (grid or random, multi-seed) with a
  best-first results table; any row becomes a preset.
- **Presets** — browse, create, edit, rename and delete training presets.
- **Screen panel** — plays any saved model at 0.5× to unlimited speed with
  the settings it was trained with, filled in automatically.
- **CLI** — `train` and `play` sub-commands for scripts and remote machines.

Only Super Mario Land is supported in this version; see [Roadmap](#roadmap).

---

## Install

Requirements: Python 3.10+, macOS / Linux / Windows, no GPU needed, a
display of at least 1366 × 960 for the GUI (the window is 1300 × 900). The
GUI follows the system appearance (light or dark).

```bash
git clone <this repo> gameboyEnv && cd gameboyEnv
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put your ROM at **`ROMs/mario.gb`**. ROMs are copyrighted and are not
included; you must own the game and dump the cartridge yourself. The file
must be the original *Super Mario Land* (cartridge title `SUPER MARIOLAN`).

Then start the app:

```bash
python main.py
```

## Quick start (the wizard)

The window opens on the **Wizard** tab.

1. **Tune.** Pick a *training goal* (a built-in preset such as
   `Mario — Campaign, recommended (~15 min)`) and what to *compare* (a sweep
   template such as `Quick start — entropy × learning rate`). The line next
   to the budget shows `6 configs = 6 trials · ≈ 4 min`. Click **Start
   search**. Each candidate trains briefly and is scored by its best
   evaluation reward; the table fills in best-first. Click **Continue with
   best**. Or click **Skip search** to use the preset as is.
2. **Preset.** Review the configuration (tuned values highlighted), give it
   a name, **Save preset & continue**. It now appears in the Train tab's
   preset list for future runs.
3. **Train.** Choose a run name and the total number of timesteps, optionally
   tick *live preview*, and **Start training**. Progress, reward, fps and
   elapsed time update live. **Stop** ends the run early and still saves
   the model.
4. **Watch.** When training finishes the wizard switches here and the agent
   plays on the screen panel with the exact observation setup it was trained
   with. **Play again**, or **Start over** with a new agent.

Everything the wizard does is visible on the expert tabs: the sweep on
**Tune**, the log on **Train**, the model on the screen panel.

### Choosing a game

The **Game** selector at the top lists every `.gb` / `.gbc` file in `ROMs/`.
Each ROM is booted once in the background to check whether PyBoy can run it,
and the label next to the selector shows the result:

- **green** — supported: Super Mario Land with the custom environment,
  presets and level modes. Everything works.
- **amber** — experimental: PyBoy has a game wrapper (e.g. Kirby's Dream
  Land) but this version has no environment, presets or level modes for it.
  It can only be tried from the CLI with pixel observations.
- **red** — cannot run: no PyBoy game wrapper for the cartridge, an unknown
  or corrupt file, or a `mario.gb` that is not the expected cartridge.

Start buttons are disabled unless the selected game is supported. Runs,
models and presets are listed per game. **Rescan ROMs** re-reads the folder.

### How long does it take?

Rough numbers on an Apple M1 Max with tile observations and 10 parallel
emulators (~3 000–3 500 env steps/s):

| Steps          | What you get                                             | Time     |
|----------------|----------------------------------------------------------|----------|
| 20 k           | learns "run right"; dies at the first enemy              | seconds  |
| 150–300 k      | jumps some obstacles                                     | ~2 min   |
| 2 M            | `Campaign, recommended` preset; usually clears 1-1       | ~12 min  |
| 8–10 M         | `extended` / `Random levels`; progress into world 2–3    | ~1 h     |
| 60 M           | `overnight` presets                                      | ~8 h     |

Tuning trials of 100 k steps (the default) are enough to separate good from
bad settings on the campaign; use 500 k+ for random / sequential sweeps.

---

## The GUI in detail

### Tune tab

Every trial is a `main.py train` subprocess of *Steps / trial* env steps,
scored from its `evaluations.npz` with the same eval cadence (*Evals /
trial*). Trials are named `<prefix>-<config>[-s<seed>]` under `models/mario/`.

- **Base preset** — the config every trial starts from; the sweep only
  overrides the listed fields.
- **Sweep template** — curated sweeps for Mario with tile observations, each
  with a one-line rationale. The JSON is editable and validated live:
  `{"param": [values, …]}` over `learning_rate, ent_coef, n_steps,
  batch_size, n_epochs, gamma, gae_lambda, clip_range, n_envs,
  action_repeat, frame_stack, obs_type, start_level, device`.
- **Grid or Random** — grid runs every combination; random draws N distinct
  ones. The random sample is seeded from the sweep text, so re-running the
  same sweep yields the same candidates (this is what makes resuming safe).
- **Seeds / config** — PPO varies a lot between seeds; with 2–3 seeds the
  table shows `mean ± std`.
- **Metric** — best, final or mean evaluation reward.
- **Reuse finished trials** — a trial whose run directory already reached
  its target *and* whose recorded config (`trial.json`) matches is reused
  instead of retrained. Stop a sweep and start it again to resume.

Results stream into the table and are saved to
`models/mario/_tune/<prefix>.json` after every config (**Load results…**
reopens them). **Save selected as preset** or **Load selected into Train
tab** promote a row.

### Train tab

Pick a **preset** or edit the fields directly (**Basic** and **Advanced**),
type a **run name** in the *Run* section (blank = `default`; pick an existing
run from the dropdown to resume it), **Start training**. The trainer runs as
a subprocess so the window stays responsive; its `ep_rew_mean`,
`ep_len_mean`, `fps`, `time_elapsed` and progress bar update live and the
raw log scrolls on the right. The status bar at the bottom of the window
always shows what is running.

- **Resume from newest checkpoint** continues the selected run. The saved
  model's architecture, `n_steps` and `batch_size` are kept; `ent_coef`,
  `learning_rate` and `n_epochs` are taken from the fields. The hint under
  the run name shows the target folder and whether it already exists.
- A **modified** marker appears under the preset selector as soon as a
  field differs from the selected preset.
- **Show live preview** loads each new `best_model.zip` as it is saved and
  plays it on the screen panel while training continues (costs some fps).
  When a run ends, its best model is selected on the screen panel, ready
  to play.
- **Stop** sends the trainer an interrupt: it saves `checkpoints/final.zip`,
  closes its emulator workers and exits. `logs/best_model.zip` (best
  evaluation reward so far) is always kept.
- **TensorBoard** serves `models/mario/` (every run, including tune trials)
  on port 6006 and opens the browser.
- **Save as…** stores the current fields as a user preset; **Manage…** opens
  the Presets tab.

While training or tuning runs, every input on every tab is locked so a run
cannot be edited mid-flight and two CPU-hungry sessions cannot collide.

### Housekeeping

Trial runs and wizard runs accumulate under `models/mario/`. Delete them
from where they were made: **Delete trial runs…** on the Tune tab removes
every run of the current prefix plus its results file, **Clear previous
search data…** on the wizard's first step does the same for the wizard's
trials, and **Delete run…** on the Train tab removes the run named in the
run field. Each asks first and shows the size it will free; saved presets
are never touched. **Open folder** reveals a run in the file manager, and
**Clear** on the screen panel blanks the emulator view.

### Screen panel (right)

The emulator view with the live episode (episode, reward, world, position,
steps, lives, coins, action) and the controls to **play a model**: the
model dropdown lists every `best`, `final` and step snapshot of every run,
plus episodes and speed (`0.5×` … `4×`, `Unlimited`), and **Clear** to blank
the view. Speed is paced per
emulator frame, so real time is real time even though a jump holds the
button for 10 frames and a walk step for 4. Selecting a model
applies the observation settings recorded in its `run.json`; **Advanced…**
opens max steps, stochastic actions and the observation setup for models
from older runs without that file.

### Presets tab

Every preset in one table (type, level mode, steps, envs, obs) with the
selected one shown in the same Basic / Advanced form as the Train tab.
Every preset is editable:

- **built-in** presets keep their shipped values behind your edits. Saving
  stores your version (the row turns *modified*); **Reset to default**
  brings the shipped values back.
- **user** presets (created here, on the Train tab or by the wizard) can be
  edited, renamed and deleted.

**New…** creates a preset from the selected one, **New from Train tab…**
from the Train tab's current fields, **Duplicate…** copies. Double-click a
row (or **Load into Train tab**) to train with it; **Use in Wizard** makes
it the wizard's goal. User presets and overrides live in
`training_presets.json`, built-ins in `builtin_presets.json`.

---

## Command line

```bash
python main.py train --n-envs 10 --timesteps 2000000 --run-name campaign1
python main.py play  --run-name campaign1 --episodes 3          # SDL2 window
python main.py gui
```

Paths are relative to the project directory regardless of where you run
the command from.

### `train` flags

| Flag                | Default | Notes                                                        |
|---------------------|---------|--------------------------------------------------------------|
| `--game`            | mario   | Only `mario` is supported (see Roadmap)                      |
| `--obs-type`        | tiles   | `tiles` (fast MLP) or `pixels` (CNN, ~40× slower)            |
| `--start-level`     | default | `default` / `random` / `sequential` / `marathon` / `W-L`     |
| `--n-envs`          | cores   | Parallel emulators, one per CPU core (max 12), clamped to cores |
| `--timesteps`       | 500000  | Total env steps                                              |
| `--action-repeat`   | 4       | Frames each action is held                                   |
| `--frame-stack`     | 4       | Consecutive observations stacked as input                    |
| `--device`          | cpu     | `cpu` (recommended for tiles), `cuda`, `mps`, `auto`         |
| `--resume`          | off     | Continue from newest `models/mario/<run>/checkpoints/*.zip`  |
| `--run-name`        | default | Sub-directory under `models/mario/`                          |
| `--checkpoint-freq` | 25000   | Env steps between checkpoints                                |
| `--eval-freq`       | 10000   | Env steps between evaluations                                |
| `--n-eval-episodes` | 3       | Mario evals are deterministic; 1 episode is run (see Performance) |
| `--learning-rate`   | 2.5e-4  |                                                              |
| `--n-steps`         | 256     | PPO rollout length per env                                   |
| `--batch-size`      | 128     | Shipped presets use 256 (see Performance)                    |
| `--n-epochs`        | 4       |                                                              |
| `--ent-coef`        | 0.01    | Entropy coefficient (raise for more exploration)             |
| `--gamma`           | 0.99    | Discount factor                                              |
| `--gae-lambda`      | 0.95    | GAE lambda                                                   |
| `--clip-range`      | 0.2     | PPO clip range                                               |
| `--seed`            | 0       |                                                              |

### `play` flags

| Flag                | Default | Notes                                          |
|---------------------|---------|------------------------------------------------|
| `--model`           | auto    | Explicit path to a model `.zip`                |
| `--run-name`        | default | Auto-resolves best → final → newest snapshot   |
| `--episodes`        | 3       |                                                |
| `--start-level`     | default | Same options as `train`                        |
| `--emulation-speed` | 1       | 1 = real time, 2 = 2×, 0 = unlimited           |
| `--stochastic`      | off     | Sample from the policy instead of argmax       |
| `--max-steps`       | 0       | Cap steps per episode (0 = none)               |

`--obs-type`, `--action-repeat` and `--frame-stack` must match the values
the model was trained with.

---

## How it works

### The Mario environment (`env.py`)

`MarioEnv` wraps PyBoy's Super Mario Land game wrapper as a Gymnasium env.

- **Held-button actions.** Buttons stay pressed for the whole
  `action_repeat` window (PyBoy's default releases every frame, which
  prevents Mario from walking). Jump actions hold A for extra frames so
  Mario reaches full jump height.
- **11 discrete actions:** NOOP, RIGHT, LEFT, JUMP, RIGHT+JUMP, RIGHT+RUN,
  RIGHT+RUN+JUMP, LEFT+JUMP, LEFT+RUN, LEFT+RUN+JUMP, DOWN.
- **Two observation types.** `tiles` (default): a 16×20 semantic grid from
  PyBoy (`-1` Mario, `0` empty, `0.5` ground, `0.6` enemy, `1` pipe/wall)
  with six normalised scalars (lives, coins, timer, x position, world,
  level) written into the top-left cells; trained with an MLP, ~10× faster
  than pixels on CPU. `pixels`: the raw 144×160 RGB screen with a CNN.
- **Shaped reward per step:**
  `3 × new forward distance` (only new territory counts, so backtracking
  cannot farm) `+ 5 × coins collected` `+ 0.05 × score gained` (enemies,
  items; the game also adds 100 score per coin, so a coin is worth 10 in
  total) `− 0.03` per step. A death gives `− 500` and a level clear
  `+ 1000` (`+ 3000` more for finishing a marathon); those steps carry no
  other terms.
- **Instant event detection.** The game-state byte at `0xFFB3` credits a
  clear the step Mario touches the goal and a death the step he dies,
  instead of 20–70 steps later when PyBoy's counters catch up. Episodes that
  end on those events end immediately; no samples are wasted in cutscenes.
- **Stuck timeout.** 200 steps without a new maximum x truncate the episode.

### Level modes (`--start-level`)

| Mode         | Episode ends on                       | Use it for                              |
|--------------|---------------------------------------|-----------------------------------------|
| `default`    | game over (all lives lost)            | Playing through the game (campaign).    |
| `random`     | any death or clear; new random level  | Generalising across every level.        |
| `sequential` | any death or clear; retry on death    | Reliably beating levels in order.       |
| `marathon`   | any death, or clearing the last level | Speedrunning all 10 levels in one go.   |
| `W-L`        | any death or clear                    | Drilling one level.                     |

Levels other than 1-1 start from cached PyBoy save-states in
`models/mario/_level_states/`, created automatically on first use. Levels
2-3 and 4-3 cannot be booted through PyBoy's wrapper (upstream bug) and are
skipped by every mode. Evaluation during training runs on the campaign
(or the fixed level for `W-L`) so eval rewards stay comparable and every
eval episode terminates.

### Training (`main.py`)

PPO from Stable-Baselines3 with `MlpPolicy` (two 256-unit layers) for tiles
or `CnnPolicy` for pixels, over a `SubprocVecEnv` of `n_envs` emulators
with `VecFrameStack`. A `CheckpointCallback` writes snapshots and an
`EvalCallback` keeps the best model. On CPU with tile observations PyTorch
is limited to one thread, which measurably speeds things up because the
network is tiny and the emulator workers need the cores.

**Apple Silicon:** CPU beats MPS by ~4× for the tile MLP (kernel-launch
overhead dominates), so `--device cpu` is the default on purpose.

### Performance

Where the time goes for one rollout of 10 envs × 512 steps (M1 Max, tile
observations), measured:

| Phase                         | Time    | Notes                                              |
|-------------------------------|---------|----------------------------------------------------|
| emulation + inference         | ~1.1 s  | 80% is PyBoy itself (0.06 ms per frame; jumps hold 10 frames) |
| PPO update, batch 128         | 0.83 s  | emulator processes idle meanwhile                  |
| PPO update, batch 256         | 0.49 s  | shipped default                                    |
| PPO update, batch 512         | 0.33 s  |                                                    |
| evaluation (every 25 k steps) | ~0.35 s | 1 episode; more would replay the identical episode |

What the software does about it:

- **Batch size 256** in every shipped preset (was 128): about 20% faster
  wall-clock at 80 gradient steps per rollout. Larger batches are faster
  still; the `Rollout shape` sweep on the Tune tab compares 256 and 512.
- **One eval episode.** Emulator and greedy policy are deterministic, so
  repeated eval episodes are identical; the trainer runs one regardless of
  the setting and says so in the log.
- **Emulators = cores.** `--n-envs` defaults to the number of CPU cores (max
  12) and is clamped to it, because more emulator processes than cores only
  adds scheduling overhead. The Train tab shows your core count next to
  the field. Torch runs single-threaded for the tile MLP; more threads
  measured no faster for a network this small.
- **Tile observations** are ~40× cheaper than pixels; use `pixels` only if
  tiles plateau.

On other machines expect roughly `350 × cores` env steps/s for tiles (the
Tune tab's time estimates use that), less on machines without performance
cores.

### Artefacts

```
models/mario/
├── _level_states/            per-level save-states (cache)
├── _tune/<prefix>.json       sweep results (Tune tab, wizard)
└── <run-name>/
    ├── checkpoints/ppo_<N>_steps.zip   periodic snapshots
    ├── checkpoints/final.zip           saved when the run ends or is stopped
    ├── logs/best_model.zip             best evaluation reward
    ├── logs/evaluations.npz            eval history
    ├── tensorboard/                    TensorBoard events
    ├── run.json                        training settings (the screen panel reads it)
    └── trial.json                      (tune trials only) config of this trial
```

`models/`, `ROMs/` and `training_presets.json` are git-ignored.

## Project layout

```
main.py               CLI entry point: gui | train | play
gui.py                Tkinter app: Train / Tune tabs, screen panel, status bar, event pump
wizard.py             Wizard tab (guided Tune → Preset → Train → Watch)
presets_tab.py        Presets tab (browse / edit / organise presets)
widgets.py            Shared Tk pieces: parameter form (ConfigForm), uniform tables
player.py             Embedded playback / live preview engine (background thread)
games.py              Game registry, ROM discovery / probe, level facts (no emulator imports)
env.py                MarioEnv, save-state bootstrap, VecEnv wrapping
runs.py               Run directories, model discovery, trainer command line
tuning.py             Sweep templates, grid/random expansion, trial scoring, results files
presets.py            Built-in and user presets
builtin_presets.json  Shipped presets (hand-editable)
tests/                Unit tests (python -m unittest discover -s tests)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

33 tests cover the emulator-free modules: sweep expansion and the
trial-reuse rules, run discovery and the `run.json` manifest, preset
defaults / overrides / rename rules, level parsing, ROM discovery and the
ROM probe (the probe test is skipped without `ROMs/mario.gb`), and a guard
that no GUI module imports PyBoy or Stable-Baselines3 at module level so the
window keeps opening instantly. The GUI itself is exercised by starting it
and running the `Mario — Quick smoke test (30 s)` preset through the wizard.

## Troubleshooting

- **`ROM not found: ROMs/mario.gb`** — place your ROM there (see Install).
- **Game shows red or amber** — only Super Mario Land is supported in this
  version; see Roadmap. Check that `mario.gb` is the original cartridge.
- **Training plateaus** — raise `ent_coef` to 0.02–0.05, sweep
  `ent_coef × learning_rate` on the Tune tab, or switch to `pixels`.
- **Slow training** — keep `obs_type=tiles`, `device=cpu`, and `n_envs`
  close to your core count. Close the live preview.
- **TensorBoard button does nothing** — `pip install tensorboard`; the
  button falls back to `python -m tensorboard.main`.
- **Playback looks wrong** — obs type, action repeat and frame stack must
  match the training run; apply the run's preset (or use the wizard) so
  they are filled in for you.

## Roadmap

- **Kirby's Dream Land and Wario Land.** The code still has a generic path
  (PyBoy's default wrapper, pixel observations, CLI only, `--game kirby`),
  but it is untested in this version and has no level modes or reward
  shaping. Proper support is planned for the next version.
- Curriculum training across level modes from inside the wizard.

## License

MIT — see [LICENSE](LICENSE). Super Mario Land is a trademark of Nintendo;
this project is not affiliated with or endorsed by Nintendo and does not
distribute any game data.
