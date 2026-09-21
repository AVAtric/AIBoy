"""AIboy's command line: train or play an agent (PPO + PyBoy), or open the app.

Usage:
    python main.py                                                     # the app (wizard opens first)
    python main.py gui                                                 # same
    python main.py train --game mario --n-envs 10 --timesteps 500000   # train headless
    python main.py play  --game mario --episodes 3                     # watch in an SDL2 window
    python main.py controller-test                                     # what a game controller sends

Every `train` records its settings, evaluation history and duration in the
experience file when it ends (see experience.py), which is how the app
learns which settings it has already tried and how fast this computer is.

Two internal sub-commands run PyBoy in a throw-away process for the GUI
(`probe-rom <file>`, `level-state <rom> <world> <level> <out>`); they are what
lets the GUI stay responsive even for a ROM the emulator cannot boot. A third,
`tensorboard --logdir <dir> --port <n>`, serves the training curves with the
TensorBoard that ships inside the program, so the built app needs no
`tensorboard` command installed on the computer.

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
                   check_start_level, level_choices, level_state_worker, prepare_level_states,
                   probe_rom_worker)
from aiboy.paths import DATA_DIR, FROZEN, recommended_n_envs
from aiboy.runs import (KEEP_CHECKPOINTS, cpu_count, latest_checkpoint, prune_checkpoints,
                  read_run_config, resolve_model_path, run_paths, write_run_config)

# PPO stability. A policy that has become nearly deterministic can be pushed
# by one large update into a state it never leaves: every emulator then
# plays the very same short episode, the policy's entropy is ~0 and so is
# its gradient (a marathon run collapsed that way at 70 M steps, to "jump
# into the first goomba", and stayed there for 80 M more while the eval
# score read -128 every time). Two safeguards:
#   - `target_kl` ends a PPO update early once it has moved the policy too
#     far (SB3 stops at 1.5x the value);
#   - the exploration guard (`_exploration_guard`) measures the policy's
#     entropy after every rollout; when it has been below ENTROPY_FLOOR for
#     COLLAPSE_PATIENCE rollouts and the last COLLAPSE_EPISODES finished
#     episodes are all identical and below the run's best evaluation, the
#     run is repaired: weights and optimizer go back to `best_model.zip`
#     and ent_coef (curiosity) is raised, ENT_COEF_AFTER_REPAIR. At most
#     MAX_REPAIRS times per run, then the run stops with a message.
TARGET_KL = 0.03
ENTROPY_FLOOR = 0.01            # nats; a uniform Discrete(11) policy has ln(11) = 2.4
COLLAPSE_PATIENCE = 3           # rollouts below the floor before a collapse is declared
COLLAPSE_EPISODES = 20          # identical finished episodes that make a collapse
GUARD_COOLDOWN = 20             # rollouts after a repair before the guard looks again
MAX_REPAIRS = 3
ENT_COEF_AFTER_REPAIR = (4.0, 0.02, 0.1)    # ent_coef x factor, at least, at most


# ------------------------- vec-env plumbing -------------------------

def build_vec_env(game, n_envs, seed, action_repeat, frame_stack, obs_type, start_level=None,
                  time_budget=DEFAULT_TIME_BUDGET, stall_steps=DEFAULT_STALL_STEPS):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from aiboy.env import env_factory, wrap_vec_env
    fns = [partial(env_factory, game, seed + i, "null", action_repeat, obs_type, start_level,
                   time_budget, stall_steps)
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


def is_collapsed(episodes, best_reward: float, min_episodes: int = COLLAPSE_EPISODES) -> bool:
    """True when the last `min_episodes` finished episodes (score, length)
    are all the same, one deterministic trajectory in a deterministic game,
    and that score is below `best_reward`, the run's best evaluation.
    Identical episodes at the best score are a solved task, not a collapse;
    without an evaluation yet (`best_reward` -inf) nothing is decided."""
    import math
    recent = list(episodes)[-min_episodes:]
    if len(recent) < min_episodes or not math.isfinite(best_reward):
        return False
    if len({(round(float(r), 1), int(l)) for r, l in recent}) > 1:
        return False
    return float(recent[-1][0]) < best_reward


def _exploration_guard(eval_cb, best_path: Path):
    """The exploration guard (see TARGET_KL above). `eval_cb` is the run's
    EvalCallback (its `best_mean_reward` says what the run once reached);
    `best_path` the model that scored it. Defined inside a function so
    importing this module does not import torch."""
    import numpy as np
    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback

    class ExplorationGuard(BaseCallback):
        def __init__(self):
            super().__init__()
            self.below_floor = 0        # consecutive rollouts with entropy under the floor
            self.cooldown = 0
            self.repairs = 0
            self.stop = False

        def policy_entropy(self) -> float:
            """Mean entropy of the current policy over (a sample of) the
            observations of the rollout just collected."""
            # (n_steps, n_envs, *obs) before the update, flattened after it.
            obs = np.asarray(self.model.rollout_buffer.observations)
            obs = obs.reshape(-1, *self.model.observation_space.shape)
            if len(obs) > 512:
                pick = np.random.default_rng(self.num_timesteps).choice(len(obs), 512, replace=False)
                obs = obs[pick]
            with th.no_grad():
                obs_t, _ = self.model.policy.obs_to_tensor(obs)
                return float(self.model.policy.get_distribution(obs_t).entropy().mean())

        def _on_rollout_end(self) -> None:
            entropy = self.policy_entropy()
            self.logger.record("train/policy_entropy", entropy)
            self.below_floor = self.below_floor + 1 if entropy < ENTROPY_FLOOR else 0
            if self.cooldown > 0:
                self.cooldown -= 1
                return
            episodes = [(float(e["r"]), int(e["l"])) for e in self.model.ep_info_buffer]
            best = float(eval_cb.best_mean_reward)
            if self.below_floor < COLLAPSE_PATIENCE or not is_collapsed(episodes, best):
                return
            reward, length = episodes[-1]
            what = (f"the agent stopped exploring: every attempt ends the same way "
                    f"({length} steps, score {reward:.0f}) while its best evaluation "
                    f"scored {best:.0f}")
            if self.repairs >= MAX_REPAIRS or not best_path.exists():
                why = ("there is no best model to go back to" if not best_path.exists()
                       else f"it was repaired {self.repairs} times already")
                print(f"[train] warning: {what}; {why}, so training stops here. Start a new "
                      f"run with a higher ent_coef (curiosity).", flush=True)
                self.stop = True
                return
            try:
                self.model.set_parameters(str(best_path), exact_match=True,
                                          device=self.model.device)
            except (RuntimeError, ValueError):
                # The best model is from before the game gained actions:
                # widen a copy and take its weights and optimizer state.
                best_model = load_for_training(best_path, self.model.get_env(),
                                               device=self.model.device)
                self.model.set_parameters(best_model.get_parameters(), exact_match=True,
                                          device=self.model.device)
            factor, at_least, at_most = ENT_COEF_AFTER_REPAIR
            old = float(self.model.ent_coef)
            self.model.ent_coef = min(max(old * factor, at_least), at_most)
            self.model.ep_info_buffer.clear()
            self.repairs += 1
            self.below_floor = 0
            self.cooldown = GUARD_COOLDOWN
            print(f"[train] warning: {what}. Restored the best model and raised curiosity "
                  f"(ent_coef {old:g} -> {self.model.ent_coef:g}) so it explores again "
                  f"(repair {self.repairs} of {MAX_REPAIRS}).", flush=True)

        def _on_step(self) -> bool:
            return not self.stop

    return ExplorationGuard()


def widen_actions(model, n_actions: int) -> bool:
    """Give a loaded PPO model `n_actions` outputs instead of fewer.

    Actions are only ever appended to a game's table (Mario gained UP and
    UP+JUMP after 11 others), so a model from before keeps what it learned:
    the old output rows are copied, the new rows start with zero weights
    and a bias 3 nats below the old actions' lowest, so they are rarely
    picked until training finds them useful. The optimizer is rebuilt
    (its moments have the old shape). False when there was nothing to do.
    Defined here so importing this module does not import torch."""
    import torch as th
    from gymnasium import spaces
    from stable_baselines3.common.distributions import make_proba_distribution
    have = int(model.action_space.n)
    if n_actions <= have:
        return False
    policy = model.policy
    old = policy.action_net
    new = th.nn.Linear(old.in_features, n_actions).to(old.weight.device)
    with th.no_grad():
        new.weight.zero_()
        new.weight[:have] = old.weight
        new.bias[:have] = old.bias
        new.bias[have:] = old.bias.min() - 3.0
    policy.action_net = new
    space = spaces.Discrete(n_actions)
    policy.action_space = model.action_space = space
    policy.action_dist = make_proba_distribution(space)
    policy.optimizer = policy.optimizer_class(policy.parameters(), lr=model.lr_schedule(1),
                                              **policy.optimizer_kwargs)
    return True


def load_for_training(path, env, *, device: str, tensorboard_log=None):
    """PPO.load of a checkpoint to continue training on `env`, widened when
    the game has gained actions since the checkpoint was saved."""
    import tempfile
    from stable_baselines3 import PPO
    from stable_baselines3.common.save_util import load_from_zip_file
    n = int(env.action_space.n)
    data, _, _ = load_from_zip_file(str(path), load_data=True, device=device)
    if int(data["action_space"].n) >= n:
        return PPO.load(str(path), env=env, device=device, tensorboard_log=tensorboard_log)
    # Widen a copy and load that one with the env, so SB3 sizes the rollout
    # buffer and the env count the usual way (set_env insists on the saved
    # number of envs; load does not).
    model = PPO.load(str(path), device=device)
    widen_actions(model, n)
    print(f"[train] the game has {n} actions now; the model had fewer. It keeps what it "
          f"learned and starts to try the new ones (UP for the submarine and the plane).",
          flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        widened = Path(tmp) / "widened.zip"
        model.save(str(widened))
        return PPO.load(str(widened), env=env, device=device, tensorboard_log=tensorboard_log)


def _apply_resume_overrides(model, args: argparse.Namespace) -> None:
    """The mutable settings of a continued run come from the flags. A loaded
    model built its learning-rate schedule from the saved value, so the
    schedule is rebuilt as well (before, a new learning rate only took
    effect at the resume after the next one)."""
    from stable_baselines3.common.utils import get_schedule_fn
    model.ent_coef = args.ent_coef
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)
    model.n_epochs = args.n_epochs
    model.target_kl = TARGET_KL


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
    why = check_start_level(args.game, start_level)
    if why:
        sys.exit(f"[{args.mode}] {why}")
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
                        start_level=start_level,
                        time_budget=args.time_budget, stall_steps=args.stall_steps)
    if GAMES[args.game].levels:
        print(f"[train] attempt limits: time budget {args.time_budget or 'full timer'} of 400 "
              f"units, stall limit {args.stall_steps or 'off'}")
    else:
        print(f"[train] attempt limits: stall limit {args.stall_steps or 'off'} (this game has "
              f"no timer; the time budget does not apply)")
    if start_level == "marathon":
        print("[train] marathon: every episode is one life from 1-1 through all levels; "
              "a death starts the next episode at 1-1 again")
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
        print(f"[train] --resume: no checkpoint in {paths['checkpoints']} yet; starting fresh.")
    if resumed_from is not None:
        print(f"[train] resuming from {resumed_from}")
        print("[train] WARNING: --resume loads the saved model's hyperparameters. "
              "Mutable overrides (ent_coef, learning_rate, n_epochs) are applied, "
              "but architecture / n_steps / batch_size come from the saved model.")
        model = load_for_training(resumed_from, env, device=args.device, tensorboard_log=tb_log)
        _apply_resume_overrides(model, args)
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
            target_kl=TARGET_KL,
            seed=args.seed,
        )
    print(f"[train] safety: an update stops once it moves the policy more than KL {TARGET_KL}; "
          f"a run whose agent stops exploring goes back to its best model with more curiosity "
          f"(up to {MAX_REPAIRS} times)")

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
    if resumed_from is not None:
        _continue_eval_history(eval_cb, paths["logs"] / "evaluations.npz")
    guard_cb = _exploration_guard(eval_cb, paths["logs"] / "best_model.zip")

    completed = False
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb, guard_cb],
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


def _continue_eval_history(eval_cb, eval_file: Path) -> None:
    """On --resume, give the new EvalCallback the run's earlier evaluations:
    a fresh callback starts with no best score, so the first evaluation of
    the continued run would replace `best_model.zip` even when it is worse
    than what the run had reached, and it rewrites evaluations.npz from
    scratch, losing the history the experience file and the Tune tab read.
    """
    import numpy as np
    if not eval_file.exists():
        return
    try:
        data = np.load(eval_file)
        eval_cb.evaluations_timesteps = [int(t) for t in data["timesteps"]]
        eval_cb.evaluations_results = [list(map(float, r)) for r in data["results"]]
        eval_cb.evaluations_length = [list(map(int, l)) for l in data["ep_lengths"]]
        if "successes" in data:
            eval_cb.evaluations_successes = [list(map(bool, s)) for s in data["successes"]]
        means = [float(np.mean(r)) for r in eval_cb.evaluations_results]
        if means:
            eval_cb.best_mean_reward = max(means)
            eval_cb.last_mean_reward = means[-1]
        print(f"[train] --resume: keeping {len(means)} earlier evaluation(s); best_model.zip is "
              f"only replaced by a score above {eval_cb.best_mean_reward:.2f}")
    except Exception as e:                                   # noqa: BLE001
        print(f"[train] --resume: could not read the earlier evaluations ({e}); the eval "
              f"history starts over")


def _remember_run(args: argparse.Namespace, run_name: str, paths: dict, duration: float,
                  completed: bool, *, resumed: bool) -> None:
    """Append this run to the experience file: what was trained, how it
    scored over time and how long it took. Never fails the run."""
    try:
        from aiboy import experience, tuning
        evals = tuning.evals_from_npz(paths["logs"] / "evaluations.npz") or []
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
    from aiboy.env import load_model
    run_name = args.run_name or "default"
    model_path = resolve_model_path(args.model, args.game, run_name)
    print(f"[play] loading model from {model_path}")

    start_level = _start_level(args)
    env = build_play_env(args.game, args.action_repeat, args.frame_stack,
                         args.emulation_speed, args.obs_type,
                         start_level=start_level,
                         time_budget=args.time_budget, stall_steps=args.stall_steps)
    model, env = load_model(model_path, env, device=args.device)
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
    p.add_argument("--game", default="mario", choices=sorted(SUPPORTED_GAMES),
                   help=f"A game with an AIboy environment: {', '.join(SUPPORTED_GAMES)}")
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
                        "or a specific level 'W-L' (fixed). Super Mario Land only; other "
                        "games start from the beginning.")


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
    t.add_argument("--resume", action="store_true",
                   help="Continue the run from its newest checkpoint if one exists "
                        "(a run without checkpoints starts fresh)")
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
    ct = sub.add_parser("controller-test",
                        help="Show what the app sees from a game controller (which pads, "
                             "which buttons) for a few seconds")
    ct.add_argument("--seconds", type=float, default=20.0)

    # Internal helpers the GUI runs in a throw-away process (see games.py).
    pr = sub.add_parser("probe-rom", help=argparse.SUPPRESS)
    pr.add_argument("rom")
    ls = sub.add_parser("level-state", help=argparse.SUPPRESS)
    ls.add_argument("rom")
    ls.add_argument("world", type=int)
    ls.add_argument("level", type=int)
    ls.add_argument("out")
    tb = sub.add_parser("tensorboard", help=argparse.SUPPRESS)
    tb.add_argument("--logdir", required=True)
    tb.add_argument("--port", type=int, default=6006)

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


def cmd_tensorboard(args: argparse.Namespace) -> None:
    """Serve `args.logdir` on localhost:`args.port` with the TensorBoard this
    program was installed (or built) with. Exits with a plain message when the
    package is missing, which the GUI shows in a dialog."""
    try:
        from tensorboard.main import run_main
    except ImportError:
        sys.exit("TensorBoard is not installed (pip install tensorboard)")
    # absl reads the command line from sys.argv; TensorBoard's own flag
    # syntax is `--flag=value`. The plain Python event loader is enough for
    # a handful of runs and needs no extra data-server binary.
    sys.argv = ["tensorboard", f"--logdir={args.logdir}", f"--port={args.port}",
                "--host=localhost", "--load_fast=false"]
    run_main()


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
    elif args.mode == "controller-test":
        from aiboy.gui.controls import watch
        sys.exit(0 if watch(args.seconds) else 1)
    elif args.mode == "probe-rom":
        probe_rom_worker(args.rom)
    elif args.mode == "level-state":
        level_state_worker(args.rom, args.world, args.level, args.out)
    elif args.mode == "tensorboard":
        cmd_tensorboard(args)


if __name__ == "__main__":
    main()
