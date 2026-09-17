# gameboyEnv — teach an AI to play Super Mario Land

Train a reinforcement-learning agent to play **Super Mario Land** on an
emulated Game Boy, then watch it play. [PyBoy](https://github.com/Baekalfen/PyBoy)
is the emulator, [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
PPO is the learner, and a Tkinter desktop app wraps the whole workflow:

```
Wizard:  1 Set up  →  2 Save  →  3 Train  →  4 Watch
```

The window has two halves: **workflow tabs on the left, the Game Boy screen
on the right**. Whatever produces something to watch (a finished model, the
live preview of a training run, the wizard's final step) shows on that
screen without switching tabs.

- **Wizard** — a guided path in plain language: pick what the agent should
  learn, let the app try a few variations of the settings and keep the
  best, train, watch. One Start button runs the whole pipeline. No
  knowledge of PPO needed; the expert vocabulary stays on the other tabs.
- **Train** — one headless training run with the full log, TensorBoard,
  resume, and an optional live preview; status, progress and ETA are
  always visible under the screen.
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

The window opens on the **Wizard** tab. It asks three plain questions and
then does the work:

1. **Set up.**
   - *What should it learn?* A **goal** such as `Mario — Campaign,
     recommended (~15 min)`. The line under it says in plain words what
     the agent will do and how long training takes on this computer.
   - *Look for better settings first?* **Yes** (recommended) tries a few
     variations of the goal's settings and keeps the best one. *Vary*
     picks what to vary, described in plain terms (`Curiosity × learning
     speed — the two that matter most`); *Effort* is Quick / Normal /
     Thorough (how many times and how long each variation is trained;
     Normal trains every variation twice for 5% of the goal's length).
     **Advanced…** exposes the raw steps and seeds. The goal's own settings
     always take part, and a variation wins only if it clearly beats them.
     **No** trains the goal's settings as they are.
   - *Then:* **run everything by itself** (on by default) chains search →
     save → train → watch without further clicks; **show it playing while
     it trains** turns on the live preview.

   The **Plan** line sums it up ("try 6 variations plus the goal's own
   settings (14 short training runs, about 9 min), then train for about
   10 min with the winner"). Press **▶ Start**. With the automatic run
   the next thing to do is come back to a playing agent; otherwise the
   result appears in the table and **Continue with the best →** moves on.
2. **Save.** Review the settings (values found by the search are
   highlighted, fields carry their plain labels). The suggested name is
   the goal plus the tuned values, e.g. `Campaign, recommended · ent 0.03
   · lr 0.0003`. **Save & continue** puts it in the Train tab's preset
   list for future runs.
3. **Train.** The run name is derived from the preset name
   (`campaign-recommended-ent0.03-lr0.0003`, numbered if it already
   exists); change it if you like, set the training length, and **Start
   training**. Progress, average score, speed, ETA and elapsed time
   update live. **Stop** ends the run early and still keeps the best model.
4. **Watch.** When training finishes the wizard switches here and the agent
   plays on the screen panel with the exact settings it was trained with.
   **Play again**, or **Train another agent**.

Everything the wizard does is visible on the expert tabs: the sweep on
**Tune** (with the expert names: the wizard's "curiosity" is `ent_coef`,
"learning speed" is `learning_rate`, and so on), the log on **Train**, the
model on the screen panel. Training status, progress, ETA, reward and fps
are always shown under the Game Boy screen, whatever tab is open.

### Running overnight

1. Goal `Mario — Marathon, overnight (~8 h)`, keep *Yes* and the default
   variation set, leave **run everything by itself** ticked (and tick the
   live preview if you want to watch), press **Start**. Expect about 1.5 h
   of search (6 variations plus the goal's settings × 2 seeds × 1 M steps)
   followed by the 60 M-step training; the ETA is under the screen.
2. On a Mac the app prevents idle sleep while training or tuning runs
   (`caffeinate`); the display may still turn off. Do not close the lid.
3. In the morning the wizard is on **Watch** and the agent is playing. If
   you restart the app instead, pick the run in the model list under the
   screen (its settings are applied from `run.json`) and click **▶ Play**.
   With the **marathon demo** on (default), a death, an exhausted time
   budget or a stall moves the demo on to the next level, so it plays
   through the whole game; the status line reports each skip.

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
  overrides the listed fields. **Include the base preset** (default on)
  adds it unchanged as candidate 1, so a sweep can only "win" by beating
  what you already had.
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
- **Metric** — **late eval reward** (average of the last third of the
  evaluations, at least two; the most reliable predictor of a long run and
  the wizard's choice), best, final or mean evaluation reward, or **reward per
  compute-minute**: best reward divided by the trial's measured wall-clock
  minutes, so candidates that learn slightly better but train slower (more
  epochs, smaller batches, pixel observations) are ranked by what each
  minute of compute actually bought. Negative rewards count as zero.
  Switching the metric re-ranks the results already in the table; the time
  of each trial is recorded in its `trial.json` (trials from before this
  was recorded use the throughput estimate).
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
  model's architecture, `n_steps` and `batch_size` are kept, and so are the
  observation settings recorded in the run's `run.json`; `ent_coef`,
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

### Disk space and housekeeping

Training artefacts are kept small automatically:

- **Step snapshots**: a run keeps only its newest 5 `ppo_<N>_steps.zip`
  while it trains (CLI: `--keep-checkpoints`, 0 = keep all), and drops
  them all when it finishes, because `final.zip` holds the last state.
  Interrupted runs keep theirs for **Resume**. `best_model.zip` and
  `final.zip` are always kept. **Compact run…** on the Train tab applies
  the same rule to older runs.
- **Tuning trials** keep only `logs/` (eval history, best model) and
  `trial.json`; checkpoints and TensorBoard events go as soon as the trial
  is scored.
- The status bar shows the size of `models/` and the number of runs.
- **Sweeps**: while a sweep runs, only the run folders of the best N
  candidates are kept (Tune tab **Keep best N trial runs**, default 9; the
  wizard uses 9). Their scores stay in the table and in the results file,
  marked *(deleted)*.
- **Wizard**: **Continue with best** and auto-complete delete all search
  runs and the results file, because the saved preset carries everything
  the training needs.

Manual clean-up lives where the data was made: **Delete trial runs…** on
the Tune tab removes every run of the current prefix plus its results
file, **Clear previous search data…** on the wizard's first step does the
same for the wizard's trials, and **Delete run…** on the Train tab removes
the run named in the run field. Each asks first and shows the size it will
free; saved presets are never touched. **Open folder** reveals a run in the
file manager, and **Clear** on the screen panel blanks the emulator view.

### Screen panel (right)

When the app starts, the Game Boy screen plays the boot video from
`assets/` with its sound, and then rests on the video's final logo frame
with a "No video" line under it until a preview or a playback takes over.
(The shipped clip shows an "AIboy" logo with a chime of our own; it is
generated by `python tools/make_intro.py`. If files named
`assets/orig_gb_intro.npz` / `.wav` are present they are used instead of
`assets/gb_intro.*`, but the repo does not contain or provide such files and
`.gitignore` keeps them out of it. To use a clip of your own put it at
`assets/gb_intro.mp4` and run `python tools/build_intro.py`, which needs
ffmpeg, to regenerate the frame and sound files the app actually uses. Set
`GAMEBOY_NO_INTRO=1` to start silently.) Below the screen: the **live episode** (episode, world, power-up,
reward, position, lives, steps, coins, action) with a status line that
says how each episode ended (died in 1-2, time budget used up, step cap,
completed every level); the **training** block with status, progress bar,
ETA, reward, episode length, fps and elapsed time, visible from every tab;
and the controls to **play a model**: the
model dropdown lists every `best`, `final` and step snapshot of every run,
plus episodes and speed (`0.5×` … `4×`, `Unlimited`), and **Clear** to blank
the view. Speed is paced per
emulator frame, so real time is real time even though a jump holds the
button for 10 frames and a walk step for 4. Selecting a model
applies the observation settings recorded in its `run.json`; **Advanced…**
opens max steps, stochastic actions, the **marathon demo** switch (after a
death, an exhausted time budget or a stall the demo continues with the next
level so it shows the whole game; off gives the strict one-life marathon)
and the observation setup for
models from older runs without that file.

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
| `--keep-checkpoints`| 5       | Newest step snapshots kept per run (0 = all)                 |
| `--eval-freq`       | 10000   | Env steps between evaluations                                |
| `--n-eval-episodes` | 3       | Mario evals are deterministic; 1 episode is run (see Performance) |
| `--time-budget`     | 250     | Timer units (of 400) an attempt may use; 0 = whole timer     |
| `--stall-steps`     | 0       | Steps without progress before truncation; 0 = off            |
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
- **Two observation types.** `tiles` (default): a 16×20 semantic grid
  from PyBoy (`-1` Mario, `0` empty, `0.5` ground, `0.6` enemy, `1`
  pipe/wall) with seven normalised scalars written into the top-left cells:
  lives, coins, timer, x position, world, level, and Mario's **power-up**
  (0 small, 0.5 super, 1 superball, read from the game's RAM at `FF99` /
  `FFB5`). Trained with an MLP, ~10× faster than pixels on CPU. `pixels`:
  the raw 144×160 RGB screen with a CNN. The score is deliberately not part
  of the observation: it only grows and nothing the agent does depends on
  it.
- **Shaped reward per step:**
  `3 × new forward distance` (only new territory counts, so backtracking
  cannot farm) `+ 5 × coins collected` `+ 0.05 × score gained` `− 0.03`
  per step. A death gives `− 500` and a level clear `+ 1000` (`+ 3000`
  more for finishing a marathon); those steps carry no other terms.
  Measured in the emulator, the score term pays: **5 per enemy kill**
  (100 points), **5 extra per coin** (coins also score 100, so a coin is
  worth 10 in total), **50 for a mushroom or flower** (1000 points), and in
  campaign mode 5 per step of the level-end timer countdown (finishing
  with more time left pays more).
- **Instant event detection.** The game-state byte at `0xFFB3` credits a
  clear the step Mario touches the goal and a death the step he dies,
  instead of 20–70 steps later when PyBoy's counters catch up. Episodes that
  end on those events end immediately; no samples are wasted in cutscenes.
- **Time per attempt.** The game gives 400 timer units per level (one unit
  is 0.65 s, so about 4 min 18 s). An attempt may use a **time budget** of
  250 of them (about 2 min 42 s, ≈ 2,400 steps); after that it is
  truncated without a death penalty, so a stuck agent does not burn the
  whole timer. The budget is checked only while playing, never during the
  level-end countdown. An optional **stall limit** (steps without a new
  furthest x) is off by default; set it, e.g. to 200, for the old
  fast-fail behaviour. Both are preset fields and recorded per run.

### Level modes (`--start-level`)

| Mode         | Episode ends on                       | Use it for                              |
|--------------|---------------------------------------|-----------------------------------------|
| `default`    | game over (all lives lost)            | Playing through the game (campaign).    |
| `random`     | any death or clear; new random level  | Generalising across every level.        |
| `sequential` | any death or clear; retry on death    | Reliably beating levels in order.       |
| `marathon`   | any death, or clearing the last level | Playing all 10 levels in one life.      |
| `W-L`        | any death or clear                    | Drilling one level.                     |

Levels other than 1-1 start from cached PyBoy save-states in
`models/mario/_level_states/`, created automatically on first use. Levels
2-3 and 4-3 cannot be booted through PyBoy's wrapper (upstream bug) and are
skipped by every mode.

**A marathon is one continuous run, not one run per level.** The episode
starts in a level; the moment Mario touches the goal the next level's
save-state is loaded (the level-end cutscene is skipped) and play continues
in the same episode. The episode ends at the first death, when the attempt's time budget is
used up, or after the last usable level (with a large bonus). There are no
extra lives in a marathon.

**Marathon training starts at a random level.** If every training episode
started at 1-1 the agent would reach level 5 only after clearing 1–4
flawlessly, and later levels would get almost no training signal. Training
episodes therefore start at a random level and run forward from there (the
level is part of the observation, so the policy knows where it is);
evaluation and playback always start at 1-1 and run the real marathon.
Completing all ten levels in one life is a long project: the `Marathon,
all levels (~1 h)` preset is a start, the `overnight` preset is realistic.

Evaluation during training mirrors the mode where that terminates cleanly
(fixed level, marathon from 1-1) and uses the campaign otherwise (random,
sequential) so eval rewards stay comparable and every eval episode ends.

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
    ├── checkpoints/ppo_<N>_steps.zip   periodic snapshots (newest 5 kept)
    ├── checkpoints/final.zip           saved when the run ends or is stopped
    ├── logs/best_model.zip             best evaluation reward
    ├── logs/evaluations.npz            eval history
    ├── tensorboard/                    TensorBoard events
    ├── run.json                        training settings (the screen panel reads it)
    └── trial.json                      (tune trials only) config of this trial
```

`models/`, `ROMs/` and `training_presets.json` are git-ignored.

## Building a standalone app (release)

`build_release.py` packages the program with PyInstaller for the computer
it runs on, so it can be used without Python:

```bash
pip install pyinstaller
python build_release.py            # -> release/GameBoyAI/ and release/GameBoyAI-<platform>.zip
python build_release.py --no-roms  # leave the ROMs folder empty (for handing the build on)
```

The script first profiles the machine (cores, performance cores, memory,
chip) and writes `system_profile.json` into the bundle. The number of
emulators the release runs in parallel comes from that profile (one per
core, at most 12, and never more than half the memory can hold), and every
built-in preset is capped to it, so a build uses the cores it has and no
more. The release folder holds the app (`GameBoyAI.app` on macOS), a
`ROMs/` folder (your ROM files are copied in unless `--no-roms`; the app
never ships a game), an empty `models/` folder and a `README.txt`. The
user's files (ROMs, models, saved presets, `gui_errors.log`) live next to
the app, never inside it, so replacing the app keeps them. After building,
the script runs the built app once (probes a ROM and trains a few hundred
steps with two emulator processes) to prove that process spawning works
in the frozen app; `--no-selftest` skips that.

Build on each kind of machine you want to ship to: a build is for one
platform and CPU type (here macOS on Apple Silicon). On macOS the app is
ad-hoc signed; on another Mac, right-click → Open the first time.

## Project layout

```
main.py               Entry point: gui | train | play (+ probe-rom / level-state helpers)
paths.py              Where the program and its data live (source tree vs frozen release)
build_release.py      Packages a standalone app for this machine (PyInstaller)
gui.py                Tkinter app: Train / Tune tabs, screen panel, status bar, event pump
wizard.py             Wizard tab (guided Set up → Save → Train → Watch, plain language)
presets_tab.py        Presets tab (browse / edit / organise presets)
widgets.py            Shared Tk pieces: parameter form (ConfigForm), uniform tables
player.py             Embedded playback / live preview engine (background thread)
games.py              Game registry, ROM discovery / probe, level facts (no emulator imports)
env.py                MarioEnv, save-state bootstrap, VecEnv wrapping
runs.py               Run directories, model discovery, trainer command line
tuning.py             Sweep templates, grid/random expansion, trial scoring, results files
presets.py            Built-in and user presets
builtin_presets.json  Shipped presets (hand-editable)
assets/               Boot video (mp4 source, generated frames .npz and .wav)
tools/make_intro.py   Generates the shipped AIboy boot-video assets (npz, wav, mp4)
tools/build_intro.py  Regenerates the boot-video assets from the mp4 (needs ffmpeg)
tests/                Unit tests (python -m unittest discover -s tests); tests/gui/ GUI checks
```

## Tests

```bash
python -m unittest discover -s tests -v          # unit tests, a few seconds
python tests/gui/check_layout.py                 # GUI checks: run one at a time, app closed
python tests/gui/check_games.py
python tests/gui/check_smoke.py
python tests/gui/check_e2e.py                    # search -> preset -> train -> watch (~30 s)
python tests/gui/check_e2e_auto.py               # the same through Auto-complete
python tests/gui/check_marathon.py               # marathon: train via Train tab, play in demo mode
python tests/gui/check_intro.py                  # start-up boot video and sound (plays once)
```

The unit tests cover the emulator-free modules: sweep expansion and the
trial-reuse rules, run discovery, manifests and clean-up rules, preset
defaults / overrides / rename rules, level parsing, ROM discovery and the
ROM probe, the power-up observation cell (both skipped without
`ROMs/mario.gb`), and a guard that no GUI module imports PyBoy or
Stable-Baselines3 at module level so the window keeps opening instantly.

The GUI checks open the real window, answer their own dialogs, and use a
separate trial prefix (`e2etest`) so they never touch your runs. Each
prints an `…_OK` line and removes everything it created.

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
- **Something went wrong in the window** — unexpected errors are shown in a
  dialog and appended to `gui_errors.log` in the project folder; a running
  training or sweep is not affected.
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
