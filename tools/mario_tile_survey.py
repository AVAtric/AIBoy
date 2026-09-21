"""Which Super Mario Land tiles are solid, and which sprites are hazards,
measured in the game itself.

PyBoy's Super Mario Land wrapper ships hand-made tile lists (blocks, pipes,
enemies, ...) that MarioEnv's tile observation is built on (env.mario_tile_lut).
This tool checks them against the game: it plays every usable level from
its save-state (the run's best model, sampling, and random button mashing,
so that it also gets off the beaten path) and records, step by step,

  - the background tile directly under Mario whenever he stands still for
    three steps and no lift sprite is under him  ->  "stood on" (solid)
  - the background tiles his body overlaps while alive  ->  "walked
    through" (not solid)

and prints, per tile id, what the observation shows for it, what PyBoy
calls it, and both counts per level. A tile that is stood on but reads 0.0
is ground the agent cannot see; a tile that is walked through but reads
0.5 is a wall that is not there (games.SML_BACKGROUND_TILES holds those).

With `--sprites` it lists instead every tile a sprite on screen used (the
enemies, items, bombs, projectiles and the score numbers are all sprites),
per level with where it was seen and how it moved. A sprite tile that is
not Mario, not an item and reads 0.0 is something the agent cannot see;
if it moves on its own and kills, it belongs in games.SML_HAZARD_TILES.
The vehicle levels (2-3, 4-3) are played with UP / DOWN and fire.

    python tools/mario_tile_survey.py [--model models/mario/<run>/logs/best_model.zip]
                                      [--episodes 10] [--steps 1500] [--sprites]

Needs ROMs/mario.gb and the level save-states (made on first use). The
model only helps as far as it gets; the random play gets a bit further
into every level each run, so the counts grow with `--episodes`.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from aiboy import games  # noqa: E402

LIFTS = {230, 238, 239}          # moving platforms are sprites; standing on one says nothing about the background


def background_tiles(sm) -> np.ndarray:
    """The 16x20 game area of background tiles only (what PyBoy's game_area
    shows before it draws the sprites in)."""
    pos = sm.screen().tilemap_position_list()
    tm = sm.tilemap_background()
    out = np.zeros((16, 20), dtype=np.int32)
    for y in range(16):
        scx = pos[(2 + y) * 8][0] // 8
        scy = pos[(2 + y) * 8][1] // 8
        for x in range(20):
            out[y, x] = tm.tile_identifier((x + scx) % 32, (2 + y + scy) % 32)
    return out


def sprite_survey(model_path: Path | None, episodes: int, steps: int):
    """Every sprite tile seen per level: count, x range on the level, y
    range on the screen, and whether it moves with Mario (part of him or
    his shot) or on its own."""
    from pyboy import PyBoy
    from pyboy.plugins import game_wrapper_super_mario_land as sml
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from aiboy.env import MARIO_TILE_LUT, MarioEnv, wrap_vec_env

    known = {}
    for name in PYBOY_LISTS:
        for t in getattr(sml, name):
            known[t] = name
    model = None
    if model_path is not None:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
    rng = np.random.default_rng(1)
    count = defaultdict(Counter)
    xs = defaultdict(lambda: defaultdict(list))
    ys = defaultdict(lambda: defaultdict(list))
    with_mario = defaultdict(lambda: defaultdict(list))     # (dx, dy) to Mario, per sighting
    reached = {}
    games.prepare_level_states("marathon")
    for lvl in games.SML_ALL_LEVELS:
        pyboy = PyBoy(str(games.ROM_DIR / games.GAMES["mario"].rom_file), window_type="null",
                      game_wrapper=True, disable_renderer=True)
        pyboy.set_emulation_speed(0)
        env = MarioEnv(pyboy, frame_skip=4, obs_type="tiles", start_level=lvl, stuck_steps=0)
        vec = wrap_vec_env(DummyVecEnv([lambda: Monitor(env)]), "tiles", 4)
        sm = pyboy.botsupport_manager()
        vehicle = lvl in games.SML_VEHICLE_LEVELS
        best = 0
        for ep in range(episodes):
            obs = vec.reset()
            done = [False]
            n = 0
            while not done[0] and n < steps:
                if vehicle:
                    action = np.array([rng.choice([11, 12, 10, 1, 5, 3, 0, 11, 12])])
                elif model is not None and ep % 3 == 0:
                    action, _ = model.predict(obs, deterministic=False)
                elif ep % 3 == 1:
                    action = np.array([rng.choice([1, 4, 5, 6, 6, 3, 0, 2, 7])])
                else:
                    action = np.array([rng.choice([5, 6, 6, 6, 4])])
                obs, _, done, info = vec.step(action)
                n += 1
                i = info[0]
                best = max(best, i["x"])
                if i["game_state"] not in games.SML_PLAY_STATES:
                    continue
                sprites = [sm.sprite(k) for k in range(40)]
                mario = [s for s in sprites if s.on_screen and MARIO_TILE_LUT[s.tile_identifier] == -1.0]
                if not mario:
                    continue
                mx = min(s.x for s in mario)
                my = min(s.y for s in mario)
                for s in sprites:
                    if not s.on_screen:
                        continue
                    t = int(s.tile_identifier)
                    count[t][lvl] += 1
                    xs[t][lvl].append(int(i["x"]))
                    ys[t][lvl].append(int(s.y))
                    with_mario[t][lvl].append((int(s.x) - mx, int(s.y) - my))
        reached[lvl] = best
        vec.close()

    print("furthest x per level:", {f"{w}-{l}": x for (w, l), x in reached.items()})
    print(f"{'tile':>5} {'obs':>5} {'pyboy calls it':>16}  where seen: level:count x-range on the level, "
          f"y on the screen, how it moves")
    for t in sorted(count):
        parts = []
        for (w, l) in games.SML_ALL_LEVELS:
            if not count[t][(w, l)]:
                continue
            rel = with_mario[t][(w, l)]
            dxs = {d[0] for d in rel}
            dys = {d[1] for d in rel}
            how = ("with Mario" if len(dxs) <= 3 and len(dys) <= 3 and len(rel) > 5 else
                   "on its own")
            parts.append(f"{w}-{l}:{count[t][(w, l)]} x{min(xs[t][(w, l)])}-{max(xs[t][(w, l)])} "
                         f"y{min(ys[t][(w, l)])}-{max(ys[t][(w, l)])} {how}")
        flag = ""
        if MARIO_TILE_LUT[t] == 0.0 and t not in games.SML_HARMLESS_SPRITES:
            flag = "  <- the agent cannot see this"
        print(f"{t:>5} {MARIO_TILE_LUT[t]:>5.2f} {known.get(t, '-'):>16}  {'; '.join(parts)}{flag}")


PYBOY_LISTS = ("base_scripts", "plane", "submarine", "coin", "mushroom", "heart", "star", "lever",
               "neutral_blocks", "moving_blocks", "pushable_blokcs", "question_block", "pipes",
               "goomba", "koopa", "moth", "flying_moth", "sphinx", "big_sphinx", "fist", "bill",
               "projectiles", "shell", "explosion", "spike", "plant")


def survey(model_path: Path | None, episodes: int, steps: int):
    from pyboy import PyBoy
    from pyboy.plugins import game_wrapper_super_mario_land as sml
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from aiboy.env import MARIO_TILE_LUT, MarioEnv, wrap_vec_env

    known = {}
    for name in PYBOY_LISTS:
        for t in getattr(sml, name):
            known[t] = name
    model = None
    if model_path is not None:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
    rng = np.random.default_rng(1)
    stood = defaultdict(Counter)
    walked = defaultdict(Counter)
    reached = {}
    games.prepare_level_states("marathon")
    for lvl in games.SML_ALL_LEVELS:
        pyboy = PyBoy(str(games.ROM_DIR / games.GAMES["mario"].rom_file), window_type="null",
                      game_wrapper=True, disable_renderer=True)
        pyboy.set_emulation_speed(0)
        env = MarioEnv(pyboy, frame_skip=4, obs_type="tiles", start_level=lvl, stuck_steps=0)
        vec = wrap_vec_env(DummyVecEnv([lambda: Monitor(env)]), "tiles", 4)
        sm = pyboy.botsupport_manager()
        best = 0
        for ep in range(episodes):
            obs = vec.reset()
            done = [False]
            n = 0
            recent = []
            while not done[0] and n < steps:
                if model is not None and ep % 3 == 0:
                    action, _ = model.predict(obs, deterministic=False)
                elif ep % 3 == 1:
                    action = np.array([rng.choice([1, 4, 5, 6, 6, 3, 0, 2, 7])])
                else:
                    action = np.array([rng.choice([5, 6, 6, 6, 4])])
                obs, _, done, info = vec.step(action)
                n += 1
                i = info[0]
                best = max(best, i["x"])
                if i["game_state"] != 0:
                    recent = []
                    continue
                bg = background_tiles(sm)
                sprites = [sm.sprite(k) for k in range(40)]
                mario = [s for s in sprites if s.on_screen and s.tile_identifier <= 80]
                if not mario:
                    recent = []
                    continue
                top = min(s.y for s in mario)
                bottom = max(s.y for s in mario) + 8
                cols = sorted({s.x // 8 for s in mario})
                recent.append((top, tuple(cols)))
                for rr in range(top // 8 - 2, bottom // 8 - 2):
                    for cc in cols:
                        if 0 <= rr < 16 and 0 <= cc < 20:
                            walked[int(bg[rr, cc])][lvl] += 1
                if len(recent) >= 3 and recent[-1] == recent[-2] == recent[-3]:
                    below = bottom // 8 - 2
                    on_lift = any(s.on_screen and s.tile_identifier in LIFTS and abs(s.y - bottom) <= 8
                                  and any(abs(s.x // 8 - c) <= 1 for c in cols) for s in sprites)
                    if on_lift:
                        continue
                    for cc in cols:
                        if 0 <= below < 16 and 0 <= cc < 20:
                            stood[int(bg[below, cc])][lvl] += 1
        reached[lvl] = best
        vec.close()

    print("furthest x per level:", {f"{w}-{l}": x for (w, l), x in reached.items()})
    print(f"{'tile':>5} {'obs':>5} {'pyboy calls it':>16} {'stood':>7} {'walked':>7}  per level stood/walked")
    for t in sorted(set(stood) | set(walked)):
        s, w = sum(stood[t].values()), sum(walked[t].values())
        detail = " ".join(f"{a}-{b}:{stood[t][(a, b)]}/{walked[t][(a, b)]}" for (a, b) in games.SML_ALL_LEVELS
                          if stood[t][(a, b)] or walked[t][(a, b)])
        flag = ""
        if s > 10 and MARIO_TILE_LUT[t] == 0.0:
            flag = "  <- ground the agent cannot see"
        elif w > 10 and s == 0 and MARIO_TILE_LUT[t] == 0.5:
            flag = "  <- a wall that is not there"
        print(f"{t:>5} {MARIO_TILE_LUT[t]:>5.2f} {known.get(t, '-'):>16} {s:>7} {w:>7}  {detail}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=None,
                    help="a trained Mario model to play with (sampling); random play otherwise")
    ap.add_argument("--episodes", type=int, default=10, help="attempts per level")
    ap.add_argument("--steps", type=int, default=1500, help="steps per attempt")
    ap.add_argument("--sprites", action="store_true",
                    help="list the sprite tiles (enemies, items, shots) instead of the ground")
    args = ap.parse_args(argv)
    if args.sprites:
        sprite_survey(args.model, args.episodes, args.steps)
    else:
        survey(args.model, args.episodes, args.steps)


if __name__ == "__main__":
    main()
