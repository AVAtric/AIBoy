"""What the agent sees: its tile observation as a small table of numbers.

A tile model does not get the picture on the Game Boy's screen. It gets a
16 x 20 grid of numbers, one per 8 x 8 tile of the play field, that says what
is there (see env.MarioEnv._tiles), plus a few HUD numbers written into the
top-left cells. `AgentView` draws the newest such grid under Tracking,
each cell tinted by what its number means, so a person can check that the
ground, the enemies and Mario are where the screen shows them.

Only the newest of the stacked frames is drawn: the model gets the last
four, this one is "now". A pixel model sees the picture itself, so there is
no table for it.
"""
from __future__ import annotations

import tkinter as tk

import numpy as np

from aiboy.gui.widgets import MONO, THEME

ROWS, COLS = 16, 20
CELL_W, CELL_H = 16, 10

# Super Mario Land's tile meanings (env.mario_tile_lut) and the seven HUD
# cells of row 0 (env.MarioEnv._tiles).
MARIO_MEANINGS = (
    (-1.0, "mario", "Mario (or his submarine / plane)"),
    (0.5, "ground", "ground, pipes and lifts: solid"),
    (0.6, "block", "question and breakable blocks"),
    (0.8, "coin", "coin"),
    (0.85, "item", "mushroom, flower, heart, star"),
    (1.0, "enemy", "enemy, bomb or projectile: dangerous"),
)
MARIO_HUD = ("lives / 9", "coins / 99", "time left / 400", "x in the level / 4096",
             "world (1-4 as 0 .. 1)", "level (1-3 as 0 .. 1)", "power-up (small 0, super .5, "
             "superball 1)")
KIRBY_HUD = ("health / 6", "lives / 9", "progress / 8192")


def format_cell(v: float) -> str:
    """A cell's number in at most three characters: 0, 1, -1, .5, .85."""
    if v == 0.0:
        return "0"
    if v == 1.0:
        return "1"
    if v == -1.0:
        return "-1"
    text = f"{v:.2f}".rstrip("0")
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def legend(game: str) -> str:
    """The hover text: how to read the table for `game`."""
    head = ("The numbers this agent gets instead of the picture: one per 8 x 8 tile of the "
            "play field, the newest of the frames it sees. ")
    if game == "mario":
        kinds = "; ".join(f"{format_cell(v)} {text}" for v, _, text in MARIO_MEANINGS)
        hud = ", ".join(f"cell {i + 1}: {t}" for i, t in enumerate(MARIO_HUD))
        return (head + f"0 is empty (sky, decoration); {kinds}. The first seven cells of the "
                f"top row are not tiles but numbers from the HUD: {hud}.")
    if game == "kirby":
        hud = ", ".join(f"cell {i + 1}: {t}" for i, t in enumerate(KIRBY_HUD))
        return (head + "Kirby's table is PyBoy's tile number / 400 (0 is empty), so a "
                f"number names a tile, not a kind of thing. The first three cells of the top "
                f"row: {hud}.")
    return head


class AgentView(tk.Canvas):
    """A 16 x 20 table of numbers; `show(grid)` redraws the cells that changed."""

    def __init__(self, parent: tk.Misc, game: str = "mario"):
        super().__init__(parent, width=COLS * CELL_W, height=ROWS * CELL_H,
                         highlightthickness=0, borderwidth=0)
        self.game = game
        self._rects: list[list[int]] = []
        self._texts: list[list[int]] = []
        self._shown = np.full((ROWS, COLS), np.nan, dtype=np.float32)
        self._build()

    def _palette(self) -> dict[str, tuple[str, str]]:
        """(fill, text) per kind for the current appearance."""
        if THEME.dark:
            return {"empty": ("#1e1e1e", "#555555"), "mario": ("#1f4f8f", "#ffffff"),
                    "ground": ("#4a4036", "#e8e0d0"), "block": ("#6b4a1a", "#ffe8c0"),
                    "coin": ("#6b5f10", "#fff6b0"), "item": ("#245a2c", "#d8ffd8"),
                    "enemy": ("#7a2323", "#ffd6d6"), "hud": ("#3a3a52", "#e0e0ff"),
                    "other": ("#333333", "#dddddd")}
        return {"empty": ("#ffffff", "#c4c4c4"), "mario": ("#cfe0ff", "#0b3a8a"),
                "ground": ("#e6ddd0", "#4a3a20"), "block": ("#ffe1b0", "#6b3f00"),
                "coin": ("#fff3a8", "#5a4a00"), "item": ("#d4f2d6", "#185a20"),
                "enemy": ("#ffc9c9", "#8a0e0e"), "hud": ("#e4e4f4", "#2a2a6a"),
                "other": ("#eeeeee", "#222222")}

    def _kind(self, r: int, c: int, v: float) -> str:
        if r == 0 and c < (len(MARIO_HUD) if self.game == "mario" else
                           len(KIRBY_HUD) if self.game == "kirby" else 0):
            return "hud"
        if v == 0.0:
            return "empty"
        if self.game == "mario":
            for value, kind, _ in MARIO_MEANINGS:
                if abs(v - value) < 1e-3:
                    return kind
        return "other"

    def _build(self) -> None:
        pal = self._palette()
        font = (MONO[0], 7)         # the app's monospace family, set up for this platform
        self.configure(background=pal["empty"][0])
        for r in range(ROWS):
            rects, texts = [], []
            for c in range(COLS):
                x, y = c * CELL_W, r * CELL_H
                rects.append(self.create_rectangle(x, y, x + CELL_W, y + CELL_H,
                                                   fill=pal["empty"][0], outline=""))
                texts.append(self.create_text(x + CELL_W / 2, y + CELL_H / 2, text="",
                                              font=font, fill=pal["empty"][1]))
            self._rects.append(rects)
            self._texts.append(texts)

    def set_game(self, game: str) -> None:
        if game != self.game:
            self.game = game
            self.clear()

    def clear(self) -> None:
        pal = self._palette()
        for r in range(ROWS):
            for c in range(COLS):
                self.itemconfigure(self._rects[r][c], fill=pal["empty"][0])
                self.itemconfigure(self._texts[r][c], text="")
        self._shown[:] = np.nan

    def show(self, grid) -> int:
        """Draw `grid` (16 x 20 numbers). Returns how many cells changed."""
        grid = np.asarray(grid, dtype=np.float32)
        if grid.shape != (ROWS, COLS):
            return 0
        pal = self._palette()
        changed = 0
        for r, c in zip(*np.nonzero(~(grid == self._shown))):
            v = float(grid[r, c])
            fill, ink = pal[self._kind(int(r), int(c), v)]
            self.itemconfigure(self._rects[r][c], fill=fill)
            self.itemconfigure(self._texts[r][c], text=format_cell(v), fill=ink)
            changed += 1
        self._shown = grid.copy()
        return changed
