"""The Game Boy in the Preview panel: a photo of a real one with the
emulator's picture inside its LCD, its buttons lit up as the agent presses
them, and the battery LED on while something is playing.

`assets/gb_interface.png` is the device photo (the repository's copy, with
AIboy lettering, made by tools/make_interface.py; a local
`assets/orig_gb_interface.png` is used instead when present, see
aiboy.paths.local_or_shipped). Its LCD is a 10:9 window of
242x218 px, so at the photo's native size it holds the 160x144 emulator
frame at 1.5x; the photo is scaled so the frame lands on the LCD at the
largest of LCD_SCALES the display has room for (`Geometry`; the window
picks the scale, see app.choose_lcd_scale). `GameBoyView` is one canvas
with three image items: the photo, the screen (on a drawn, exactly even
dark rim: the photo's own shadow around the window is uneven, so that area
is repainted in the bezel colour) and the controls region, which is
swapped for a version with a glow over the pressed buttons (composed with
PIL once per action and cached). Emulator frames arrive in PyBoy's four
greys and are shown in the original Game Boy's four green shades
(`dmg_tint`). Hovering a button tells
what it does; when a person plays, pressing a button with the mouse holds
it (`on_buttons`) and a click gives the canvas the keyboard focus.

Without the photo (a bundle built without the asset) a plain drawn device
with the same geometry is used.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from aiboy.gui.widgets import Tooltip
from aiboy.paths import local_or_shipped

PHOTO = local_or_shipped("gb_interface.png")
PHOTO_W, PHOTO_H = 446, 737

GAME_W, GAME_H = 160, 144
LCD_SCALES = (2.0, 1.75, 1.5)                  # emulator pixel scales, largest first

# Everything below is in the photo's own pixels (measured from the image).
LCD = (109, 99, 351, 317)                      # the window behind the glass (blue in the photo)
LCD_PATCH = (104, 95, 354, 321)                # the window plus the photo's uneven shadow round it
BEZEL = (86, 78, 89)                           # the glass bezel's colour next to the window
LCD_OFF = (12, 14, 24)                         # the drawn rim between bezel and screen
RIM = 2.5                                      # its width, in photo pixels

# The DMG's four shades, lightest first, for PyBoy's four greys.
DMG_SHADES = ((155, 188, 15), (139, 172, 15), (48, 98, 48), (15, 56, 15))
PYBOY_GREYS = (255, 153, 85, 0)


def _dmg_lut() -> np.ndarray:
    """Grey level -> DMG shade; a level in between gets the nearest shade."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        nearest = min(range(4), key=lambda i: abs(PYBOY_GREYS[i] - v))
        lut[v] = DMG_SHADES[nearest]
    return lut


_DMG_LUT = _dmg_lut()


def dmg_tint(frame: np.ndarray) -> np.ndarray:
    """A grey emulator frame in the Game Boy's green shades; a coloured
    frame (the boot video) unchanged."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        return frame
    sample = frame[::8, ::8]
    if not (np.array_equal(sample[..., 0], sample[..., 1])
            and np.array_equal(sample[..., 1], sample[..., 2])):
        return frame
    return _DMG_LUT[frame[..., 0]]
LED = (64, 178)                                # battery LED centre
PAD_BOX = (28, 440, 420, 650)                  # region that holds every button

# Button shapes: (kind, (x0, y0, x1, y1)); "rect" = D-pad arm, "oval" = button.
REGIONS: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "up": ("rect", (72, 447, 113, 484)),
    "down": ("rect", (72, 520, 113, 557)),
    "left": ("rect", (38, 482, 74, 522)),
    "right": ("rect", (111, 482, 147, 522)),
    "b": ("oval", (286, 487, 338, 539)),
    "a": ("oval", (358, 455, 410, 507)),
    "select": ("oval", (136, 596, 204, 621)),
    "start": ("oval", (212, 596, 270, 621)),
}
LABELS = {
    "up": "D-pad up (not used by the agent)",
    "down": "D-pad down: duck / enter a pipe",
    "left": "D-pad left: walk left",
    "right": "D-pad right: walk right",
    "a": "A: jump (held longer = higher)",
    "b": "B: run, and fire when Mario has a flower",
    "select": "SELECT (not used by the agent)",
    "start": "START: pauses the game (not used by the agent)",
}
# Action-name words -> buttons. Anything else (NOOP, an unknown name) lights nothing.
WORD_BUTTONS = {"RIGHT": "right", "LEFT": "left", "DOWN": "down", "UP": "up",
                "JUMP": "a", "RUN": "b", "A": "a", "B": "b", "START": "start", "SELECT": "select"}
GLOW = (255, 214, 60)             # the highlight colour (alpha added per layer)


def photo_scale(lcd_scale: float) -> float:
    """Photo scale that puts GAME_W * lcd_scale pixels across the LCD."""
    return lcd_scale * GAME_W / (LCD[2] - LCD[0])


def photo_size(lcd_scale: float) -> tuple[int, int]:
    f = photo_scale(lcd_scale)
    return round(PHOTO_W * f), round(PHOTO_H * f)


class Geometry:
    """Every position of the view for one LCD scale, in canvas pixels."""

    def __init__(self, lcd_scale: float):
        f = self.factor = photo_scale(lcd_scale)
        self.lcd_scale = lcd_scale
        self.width, self.height = photo_size(lcd_scale)
        self.screen_w, self.screen_h = round(GAME_W * lcd_scale), round(GAME_H * lcd_scale)
        patch = self.box(LCD_PATCH)
        self.screen_x = (patch[0] + patch[2] - self.screen_w) // 2
        self.screen_y = (patch[1] + patch[3] - self.screen_h) // 2
        self.rim = max(2, round(RIM * f))
        self.rim_box = (self.screen_x - self.rim, self.screen_y - self.rim,
                        self.screen_x + self.screen_w + self.rim,
                        self.screen_y + self.screen_h + self.rim)
        self.led = (round(LED[0] * f), round(LED[1] * f))
        self.pad_box = self.box(PAD_BOX)
        self.regions = {key: (kind, self.box(b)) for key, (kind, b) in REGIONS.items()}

    def box(self, b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        f = self.factor
        return round(b[0] * f), round(b[1] * f), round(b[2] * f), round(b[3] * f)

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.screen_w, self.screen_h


def buttons_for_action(name: str | None) -> frozenset[str]:
    """'RIGHT+RUN+JUMP' -> {'right', 'b', 'a'}; None / 'NOOP' / '—' -> {}."""
    if not name:
        return frozenset()
    found = set()
    for word in str(name).upper().split("+"):
        button = WORD_BUTTONS.get(word.strip())
        if button:
            found.add(button)
    return frozenset(found)


def _draw_fallback() -> Image.Image:
    """A plain device with the photo's geometry, for bundles without it."""
    img = Image.new("RGB", (PHOTO_W, PHOTO_H), (205, 203, 196))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((6, 6, PHOTO_W - 7, PHOTO_H - 7), radius=24, outline=(120, 120, 116),
                        width=2)
    d.rounded_rectangle((40, 62, 410, 350), radius=10, fill=(96, 92, 104))
    d.rectangle(LCD_PATCH, fill=BEZEL)
    d.ellipse((LED[0] - 3, LED[1] - 3, LED[0] + 3, LED[1] + 3), fill=(90, 20, 20))
    d.text((40, 372), "GAME BOY", fill=(40, 40, 140))
    dark = (40, 40, 44)
    d.rectangle(REGIONS["up"][1][:2] + REGIONS["down"][1][2:], fill=dark)
    d.rectangle(REGIONS["left"][1][:2] + REGIONS["right"][1][2:], fill=dark)
    for key, colour in (("a", (150, 40, 110)), ("b", (150, 40, 110)),
                        ("select", (120, 120, 120)), ("start", (120, 120, 120))):
        d.ellipse(REGIONS[key][1], fill=colour)
    d.text((150, 626), "SELECT", fill=(40, 40, 140))
    d.text((228, 626), "START", fill=(40, 40, 140))
    d.text((305, 545), "B", fill=(40, 40, 140))
    d.text((378, 512), "A", fill=(40, 40, 140))
    return img


def device_mask(img: Image.Image) -> Image.Image:
    """Where the device is (255) and where the white backdrop is (0): the
    backdrop is whatever a flood fill reaches from the corners. The edge is
    eroded by a pixel and softened so the photo's anti-aliased white rim
    disappears instead of showing as a bright outline."""
    probe = img.copy()
    sentinel = (255, 0, 255)
    w, h = probe.size
    for corner in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if min(probe.getpixel(corner)) > 200:                # white-ish: outside the device
            ImageDraw.floodfill(probe, corner, sentinel, thresh=70)
    px = probe.load()
    mask = Image.new("L", probe.size, 255)
    mp = mask.load()
    rim = 10                                                  # white specks the fill missed
    for y in range(h):
        for x in range(w):
            if px[x, y] == sentinel:
                mp[x, y] = 0
            elif (x < rim or y < rim or x >= w - rim or y >= h - rim) and min(px[x, y]) > 225:
                mp[x, y] = 0
    return mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))


def load_photo(path: Path = PHOTO, background: tuple[int, int, int] = (34, 34, 34),
               size: tuple[int, int] = (PHOTO_W, PHOTO_H)) -> Image.Image:
    """The device photo at `size` (or the drawn fallback), laid on
    `background` (the window colour) with the photo's white backdrop
    removed, so the device sits on the window and not on a white card."""
    try:
        img = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        img = _draw_fallback()
    if img.size != (PHOTO_W, PHOTO_H):
        img = img.resize((PHOTO_W, PHOTO_H), Image.LANCZOS)
    # The window and the uneven shadow the photo has around it become plain
    # bezel; the view draws its own, even rim under the screen.
    ImageDraw.Draw(img).rectangle(LCD_PATCH, fill=BEZEL)
    img = Image.composite(img, Image.new("RGB", img.size, background), device_mask(img))
    if img.size != tuple(size):
        img = img.resize(size, Image.LANCZOS)
    return img


def window_rgb(widget: tk.Misc) -> tuple[int, int, int]:
    """The window background as 8-bit RGB (Tk names it, e.g.
    'systemWindowBackgroundColor' on macOS, so ask Tk for the value)."""
    try:
        name = ttk.Style(widget).lookup("TFrame", "background") or widget.cget("background")
        r, g, b = widget.winfo_rgb(name)
        return r // 256, g // 256, b // 256
    except tk.TclError:
        return 34, 34, 34


def glow_image(pad: Image.Image, pressed: frozenset[str], geo: Geometry) -> Image.Image:
    """The controls region (`pad`, the pad_box crop) with a glow over every
    button in `pressed`."""
    if not pressed:
        return pad
    img = pad.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    ox, oy = geo.pad_box[0], geo.pad_box[1]
    f = geo.factor
    for key in pressed:
        kind, (x0, y0, x1, y1) = geo.regions[key]
        x0, y0, x1, y1 = x0 - ox, y0 - oy, x1 - ox, y1 - oy
        # A soft halo around the button, then the bright press on top.
        for grow, alpha in ((7 * f, 60), (3 * f, 100), (0, 150)):
            box = (x0 - grow, y0 - grow, x1 + grow, y1 + grow)
            if kind == "oval":
                d.ellipse(box, fill=GLOW + (alpha,))
            else:
                d.rounded_rectangle(box, radius=6 * f + grow, fill=GLOW + (alpha,))
    return Image.alpha_composite(img, overlay).convert("RGB")


class GameBoyView(tk.Canvas):
    """The device on a canvas, at `lcd_scale` emulator pixels (see
    `Geometry`). `set_screen(photo_image)` puts a `screen_size` PhotoImage
    on the LCD; `show(action_name)` lights that action's buttons (None or
    "NOOP" clears them); `set_power(on)` drives the battery LED; hovering
    a button shows its label (plus the keys that press it, see
    `set_key_hints`); `on_buttons(callback)` reports the buttons held down
    with the mouse."""

    def __init__(self, parent: tk.Misc, lcd_scale: float = 1.5, **kw):
        geo = self.geo = Geometry(lcd_scale)
        rgb = window_rgb(parent)
        self.photo = load_photo(background=rgb, size=(geo.width, geo.height))
        super().__init__(parent, width=geo.width, height=geo.height, highlightthickness=0,
                         bg="#%02x%02x%02x" % rgb, **kw)
        self._photo_img = ImageTk.PhotoImage(self.photo)
        self.create_image(0, 0, anchor="nw", image=self._photo_img)
        self.create_rectangle(*geo.rim_box, fill="#%02x%02x%02x" % LCD_OFF, outline="")
        self._pad = self.photo.crop(geo.pad_box)
        self._cache: dict[frozenset[str], ImageTk.PhotoImage] = {}
        self._pressed: frozenset[str] = frozenset()
        self._pad_id = self.create_image(geo.pad_box[0], geo.pad_box[1], anchor="nw",
                                         image=self._pad_image(frozenset()))
        self._screen_id = self.create_image(geo.screen_x, geo.screen_y, anchor="nw")
        self._screen_img: ImageTk.PhotoImage | None = None
        x, y = geo.led
        r = max(4, round(4 * geo.factor))
        self._led_halo = self.create_oval(x - r - 3, y - r - 3, x + r + 3, y + r + 3,
                                          fill="#7a1010", outline="", state="hidden")
        self._led = self.create_oval(x - r, y - r, x + r, y + r, fill="#ff3030", outline="",
                                     state="hidden")
        self._power = False
        self._hover: str | None = None
        self._key_hints: dict[str, str] = {}
        self._tip = Tooltip(self, self._hover_text)
        self._on_buttons = None
        self._mouse_held: frozenset[str] = frozenset()
        self.bind("<Motion>", self._on_motion, add="+")
        self.bind("<ButtonPress-1>", self._on_mouse_down, add="+")
        self.bind("<ButtonRelease-1>", self._on_mouse_up, add="+")
        self.bind("<Leave>", self._on_mouse_up, add="+")

    # ----- screen -----

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.geo.screen_size

    def set_screen(self, image: ImageTk.PhotoImage) -> None:
        """Show `image` (`screen_size`) on the LCD; keeps a reference."""
        self._screen_img = image
        self.itemconfig(self._screen_id, image=image)

    # ----- battery LED -----

    @property
    def power(self) -> bool:
        return self._power

    def set_power(self, on: bool) -> None:
        if on == self._power:
            return
        self._power = on
        state = "normal" if on else "hidden"
        self.itemconfig(self._led_halo, state=state)
        self.itemconfig(self._led, state=state)

    # ----- buttons -----

    def _pad_image(self, pressed: frozenset[str]) -> ImageTk.PhotoImage:
        img = self._cache.get(pressed)
        if img is None:
            img = ImageTk.PhotoImage(glow_image(self._pad, pressed, self.geo))
            self._cache[pressed] = img
        return img

    @property
    def pressed(self) -> frozenset[str]:
        return self._pressed

    def show(self, action_name: str | None) -> None:
        pressed = buttons_for_action(action_name)
        if pressed == self._pressed:
            return
        self._pressed = pressed
        self.itemconfig(self._pad_id, image=self._pad_image(pressed))

    def button_at(self, x: int, y: int) -> str | None:
        """The button under canvas position (x, y), if any."""
        for key, (_kind, (x0, y0, x1, y1)) in self.geo.regions.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return key
        return None

    def _on_motion(self, event) -> None:
        key = self.button_at(event.x, event.y)
        if key == self._hover:
            return
        self._hover = key
        self.config(cursor="hand2" if key else "")
        self._tip.hide()
        if key:
            self._tip.schedule()

    def set_key_hints(self, hints: dict[str, str]) -> None:
        """Button -> the keys that press it ("↑ or W"), added to the hover labels."""
        self._key_hints = dict(hints)

    def _hover_text(self) -> str:
        key = self._hover or ""
        text = LABELS.get(key, "")
        hint = self._key_hints.get(key)
        if text and hint and hint != "—":
            text += f" — key {hint}"
        return text

    # ----- mouse input -----

    def on_buttons(self, callback) -> None:
        """`callback(frozenset)` is called with the buttons held with the
        mouse whenever that changes (a press on a button, then the release)."""
        self._on_buttons = callback

    def _report_mouse(self, held: frozenset[str]) -> None:
        if held == self._mouse_held:
            return
        self._mouse_held = held
        if self._on_buttons is not None:
            self._on_buttons(held)

    def _on_mouse_down(self, event) -> None:
        self.focus_set()                        # so the keyboard reaches the game
        key = self.button_at(event.x, event.y)
        self._report_mouse(frozenset({key}) if key else frozenset())

    def _on_mouse_up(self, _event=None) -> None:
        self._report_mouse(frozenset())
