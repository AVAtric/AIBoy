"""The Game Boy in the Preview panel: a picture of the device with the
emulator's picture inside its LCD, its buttons lit up as the agent presses
them, and its ON light lit while something is playing.

The artwork is a pair of pictures, `assets/interface/aiboy_off.png` and
`aiboy_on.png` (the AIboy design, transparent around the case; the two
differ only at the ON light). A developer's originals, `gameboy_off.png`
and `gameboy_on.png` in the same folder (or in the assets folder next to
a built app), are shown instead when present and are never distributed
(see aiboy.paths.local_or_shipped). tools/make_interface.py brings new
pictures to PHOTO_W x PHOTO_H, the size every measurement below is in.

The LCD window (LCD) is 568 x 512 px, 10:9 like the 160 x 144 emulator
frame. The picture is scaled so that the frame plus a RIM-wide dark rim
lands exactly on that window, at the largest of LCD_SCALES the display has
room for (`Geometry`; the window picks the scale, see
app.choose_lcd_scale). `GameBoyView` is a frame of the picture's size
holding three canvases: the device (the picture in bands, the rim, the ON
light as a swap between the two pictures' crops), the pad (the controls
region, swapped for a version with a glow over the pressed buttons,
composed with PIL once per action and cached) and the screen, each placed
where it belongs.

On macOS each of the three canvases lives in a borderless child window
of its own (`own_windows`): Tk there redraws a whole window, every image
in it, whenever anything in it changes, and the picture is big. With
everything in the main window a new frame or a lit button cost ~35 ms
(the display fell to ~14 fps) and every live-panel number did the same.
A child window redraws only itself and hides and shows with the main
window; `_follow` polls where the frame is every FOLLOW_MS and moves the
three along (macOS sends no event while the window is dragged, so
event-driven placement left them behind), and re-stacks them whenever the
main window is shown, focused or moved. Tk gives such a window a level
above every application; `_place_windows` puts it back to the normal
level each time it shows one, so another app's window covers the Game
Boy like any other. Elsewhere the canvases are simply placed in the
frame. Either way
`screen_image` is the PhotoImage to paste frames into. Emulator frames
arrive in PyBoy's four greys and are shown in the pale green of the boot
video's idle screen, dark on light (`dmg_tint`). Hovering a button tells
what it does; when a person plays, pressing a button with the mouse holds
it (`on_buttons`) and a click gives the canvas the keyboard focus.

Without the pictures (a bundle built without them) a plain drawn device
with the same geometry is used.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from aiboy.gui.widgets import Tooltip
from aiboy.paths import local_or_shipped

PHOTO_OFF = local_or_shipped("interface/aiboy_off.png")
PHOTO_ON = local_or_shipped("interface/aiboy_on.png")
PHOTO_W, PHOTO_H = 1000, 1659

GAME_W, GAME_H = 160, 144
LCD_SCALES = (2.0, 1.75, 1.5)                  # emulator pixel scales, largest first
FOLLOW_MS = 16                                 # how often the child windows check where the frame is

# Everything below is in the picture's own pixels (measured from aiboy_off.png).
LCD = (228, 208, 796, 720)                     # the dark window behind the glass (568 x 512)
LCD_OFF = (16, 21, 33)                         # its colour: the rim the view draws round the screen
RIM = 4                                        # the rim's width, in window pixels at every scale
LED_BOX = (96, 353, 172, 429)                  # the ON light with its glow; the "on" picture differs only here
BEZEL_BOX = (18, 114, 929, 834)                # the glass around the LCD (the drawn fallback)
PAD_BOX = (40, 980, 965, 1435)                 # region that holds every button

# The screen's four shades, lightest first, for PyBoy's four greys: the
# boot video's idle screen (pale green paper, near-black ink, sampled from
# the video) with two shades in between.
DMG_SHADES = ((219, 243, 204), (133, 155, 134), (76, 96, 88), (4, 22, 30))
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


# Button shapes: (kind, (x0, y0, x1, y1)); "rect" = D-pad arm, "oval" = button.
REGIONS: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "up": ("rect", (152, 1014, 247, 1097)),
    "down": ("rect", (152, 1175, 247, 1264)),
    "left": ("rect", (72, 1093, 156, 1177)),
    "right": ("rect", (244, 1093, 330, 1177)),
    "b": ("oval", (642, 1101, 761, 1221)),
    "a": ("oval", (809, 1023, 928, 1144)),
    "select": ("oval", (304, 1338, 437, 1405)),
    "start": ("oval", (480, 1342, 611, 1409)),
}
LABELS = {
    "up": "D-pad up: the submarine and the plane rise (levels 2-3 and 4-3); Kirby flies",
    "down": "D-pad down: duck / enter a pipe",
    "left": "D-pad left: walk left",
    "right": "D-pad right: walk right",
    "a": "A: jump (held longer = higher)",
    "b": "B: run, and fire when Mario has a flower; Kirby inhales",
    "select": "SELECT (not used by the agent)",
    "start": "START: pauses the game (not used by the agent)",
}
# Action-name words -> buttons. Anything else (NOOP, an unknown name) lights nothing.
WORD_BUTTONS = {"RIGHT": "right", "LEFT": "left", "DOWN": "down", "UP": "up",
                "JUMP": "a", "RUN": "b", "A": "a", "B": "b", "START": "start", "SELECT": "select"}
GLOW = (255, 214, 60)             # the highlight colour (alpha added per layer)


def photo_scale(lcd_scale: float) -> float:
    """Picture scale that puts GAME_W * lcd_scale pixels plus the rim on
    both sides exactly across the LCD window."""
    return (lcd_scale * GAME_W + 2 * RIM) / (LCD[2] - LCD[0])


def photo_size(lcd_scale: float) -> tuple[int, int]:
    f = photo_scale(lcd_scale)
    return round(PHOTO_W * f), round(PHOTO_H * f)


class Geometry:
    """Every position of the view for one LCD scale, in canvas pixels."""

    def __init__(self, lcd_scale: float):
        self.factor = photo_scale(lcd_scale)
        self.lcd_scale = lcd_scale
        self.width, self.height = photo_size(lcd_scale)
        self.screen_w, self.screen_h = round(GAME_W * lcd_scale), round(GAME_H * lcd_scale)
        lcd = self.box(LCD)
        self.screen_x = (lcd[0] + lcd[2] - self.screen_w) // 2
        self.screen_y = (lcd[1] + lcd[3] - self.screen_h) // 2
        self.rim = RIM
        self.rim_box = (self.screen_x - RIM, self.screen_y - RIM,
                        self.screen_x + self.screen_w + RIM,
                        self.screen_y + self.screen_h + RIM)
        self.led_box = self.box(LED_BOX)
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


def tiles(width: int, height: int, holes: list[tuple[int, int, int, int]]
          ) -> list[tuple[int, int, int, int]]:
    """Rectangles that cover a width x height area except the `holes`
    (x0, y0, x1, y1; x1 / y1 exclusive), none of them overlapping a hole:
    rows are cut at every hole edge, and each row is cut around the holes
    that span it."""
    ys = sorted({0, height, *(y for h in holes for y in (h[1], h[3]) if 0 < y < height)})
    out = []
    for y0, y1 in zip(ys, ys[1:]):
        spanning = sorted((h[0], h[2]) for h in holes if h[1] <= y0 and h[3] >= y1)
        x = 0
        for hx0, hx1 in spanning:
            if hx0 > x:
                out.append((x, y0, hx0, y1))
            x = max(x, hx1)
        if x < width:
            out.append((x, y0, width, y1))
    return out


def _draw_fallback(lit: bool) -> Image.Image:
    """A plain device with the pictures' geometry, for bundles without
    them: a black case, the glass, the LCD window, the ON light (`lit`),
    the buttons and their labels. RGBA, transparent around the case."""
    img = Image.new("RGBA", (PHOTO_W, PHOTO_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, PHOTO_W - 9, PHOTO_H - 9), radius=60, fill=(18, 22, 28))
    d.rounded_rectangle(BEZEL_BOX, radius=24, fill=(54, 62, 78))
    d.rectangle(LCD, fill=LCD_OFF)
    x0, y0, x1, y1 = LED_BOX
    cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, 14
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(40, 190, 120) if lit else (20, 40, 34))
    d.text((cx - 10, cy + 24), "ON", fill=(170, 176, 190))
    d.text((40, 870), "AI BOY", fill=(235, 235, 240))
    pad = (22, 24, 30)
    d.rectangle(REGIONS["up"][1][:2] + REGIONS["down"][1][2:], fill=pad)
    d.rectangle(REGIONS["left"][1][:2] + REGIONS["right"][1][2:], fill=pad)
    for key, colour in (("a", (40, 60, 170)), ("b", (40, 60, 170)),
                        ("select", (120, 120, 128)), ("start", (120, 120, 128))):
        d.ellipse(REGIONS[key][1], fill=colour)
    grey = (170, 176, 190)
    d.text((330, 1420), "SELECT", fill=grey)
    d.text((510, 1424), "START", fill=grey)
    d.text((740, 1235), "B", fill=grey)
    d.text((905, 1158), "A", fill=grey)
    return img


def load_photo(path: Path = PHOTO_OFF, background: tuple[int, int, int] = (34, 34, 34),
               size: tuple[int, int] = (PHOTO_W, PHOTO_H), lit: bool = False) -> Image.Image:
    """The device picture at `size` (or the drawn fallback, with its ON
    light `lit`), laid on `background` (the window colour) through its
    transparency, so the case sits on the window and not on a card. The
    LCD window is flattened to the rim's colour: the view draws its own,
    exactly even rim under the screen and no sliver of the glass' reflection
    may show beside it."""
    try:
        img = Image.open(path).convert("RGBA")
        if img.size != (PHOTO_W, PHOTO_H):
            img = img.convert("RGBa").resize((PHOTO_W, PHOTO_H), Image.LANCZOS).convert("RGBA")
    except (OSError, ValueError):
        img = _draw_fallback(lit)
    inset = 3                                                 # keep the window's soft edge
    ImageDraw.Draw(img).rectangle((LCD[0] + inset, LCD[1] + inset, LCD[2] - inset, LCD[3] - inset),
                                  fill=LCD_OFF + (255,))
    img = Image.alpha_composite(Image.new("RGBA", img.size, background + (255,)), img).convert("RGB")
    if img.size != tuple(size):
        img = img.resize(size, Image.LANCZOS)
    return img


# On macOS a ttk LabelFrame's inside is drawn one shade lighter than the
# window (Tk's name for that shade), so a device composed on the window
# colour showed as a darker rectangle around the rounded corners.
AQUA_GROUP_BOX_COLOR = "systemWindowBackgroundColor1"


def window_rgb(widget: tk.Misc) -> tuple[int, int, int]:
    """The colour behind `widget` as 8-bit RGB: the group-box shade when it
    sits inside a ttk LabelFrame on macOS, else the frame background (Tk
    names them, e.g. 'systemWindowBackgroundColor', so ask Tk for the
    value)."""
    try:
        name = ttk.Style(widget).lookup("TFrame", "background") or widget.cget("background")
        if widget.tk.call("tk", "windowingsystem") == "aqua":
            parent = widget
            while parent is not None:
                if isinstance(parent, ttk.LabelFrame):
                    name = AQUA_GROUP_BOX_COLOR
                    break
                parent = parent.master
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


class GameBoyView(tk.Frame):
    """The device, at `lcd_scale` emulator pixels (see `Geometry`).
    `screen_image` is the `screen_size` PhotoImage on the LCD: paste each
    new frame into it (`set_screen`). `show(action_name)` lights that
    action's buttons (None or "NOOP" clears them); `set_power(on)` drives
    the ON light; hovering a button shows its label (plus the keys that
    press it, see `set_key_hints`); `on_buttons(callback)` reports the
    buttons held down with the mouse. Click anywhere on it to give it the
    keyboard focus."""

    def __init__(self, parent: tk.Misc, lcd_scale: float = 1.5, **kw):
        geo = self.geo = Geometry(lcd_scale)
        rgb = window_rgb(parent)
        bg = "#%02x%02x%02x" % rgb
        lcd_bg = "#%02x%02x%02x" % LCD_OFF
        super().__init__(parent, width=geo.width, height=geo.height, bg=bg, **kw)
        size = (geo.width, geo.height)
        self.photo = load_photo(PHOTO_OFF, background=rgb, size=size)
        photo_on = load_photo(PHOTO_ON, background=rgb, size=size, lit=True)
        self.own_windows = self.tk.call("tk", "windowingsystem") == "aqua"
        self._windows: list[tuple[tk.Toplevel, int, int, int, int]] = []
        self._placed: dict[tk.Toplevel, str] = {}

        px, py, px1, py1 = geo.pad_box
        self.device = self._canvas(0, 0, geo.width, geo.height, bg)
        # The pad and the screen lie over the device: on macOS their windows
        # are children of the device's window (a sibling shown later would
        # stay underneath, whatever `lift` says).
        over = self.device.winfo_toplevel() if self.own_windows else None
        self.pad = self._canvas(px, py, px1 - px, py1 - py, bg, over)
        self.screen = self._canvas(geo.screen_x, geo.screen_y, geo.screen_w, geo.screen_h, lcd_bg,
                                   over)

        # The device: the picture in bands that leave out the rim box, the
        # pad box and the ON light (so nothing large sits under what
        # changes), the rim, and the light: the same spot of the "off" and
        # the "on" picture, swapped by `set_power`.
        self._tiles: list[ImageTk.PhotoImage] = []
        for box in tiles(geo.width, geo.height, [geo.rim_box, geo.pad_box, geo.led_box]):
            img = ImageTk.PhotoImage(self.photo.crop(box))
            self._tiles.append(img)
            self.device.create_image(box[0], box[1], anchor="nw", image=img)
        self.device.create_rectangle(*geo.rim_box, fill=lcd_bg, outline="")
        self._led_off = ImageTk.PhotoImage(self.photo.crop(geo.led_box))
        self._led_on = ImageTk.PhotoImage(photo_on.crop(geo.led_box))
        self._led = self.device.create_image(geo.led_box[0], geo.led_box[1], anchor="nw",
                                             image=self._led_off)
        self._power = False

        # The pad: the controls region, with a glow over the pressed buttons.
        self._pad = self.photo.crop(geo.pad_box)
        self._cache: dict[frozenset[str], ImageTk.PhotoImage] = {}
        self._pressed: frozenset[str] = frozenset()
        self._pad_id = self.pad.create_image(0, 0, anchor="nw", image=self._pad_image(frozenset()))
        self._hover: str | None = None
        self._key_hints: dict[str, str] = {}
        self._tip = Tooltip(self.pad, self._hover_text)
        self._on_buttons = None
        self._mouse_held: frozenset[str] = frozenset()
        self.pad.bind("<Motion>", self._on_motion, add="+")
        self.pad.bind("<ButtonPress-1>", self._on_mouse_down, add="+")
        self.pad.bind("<ButtonRelease-1>", self._on_mouse_up, add="+")
        self.pad.bind("<Leave>", self._on_mouse_up, add="+")

        # The screen.
        self.screen_image = ImageTk.PhotoImage("RGB", geo.screen_size)
        self.screen.create_image(0, 0, anchor="nw", image=self.screen_image)
        for widget in (self, self.device, self.screen):
            widget.bind("<ButtonPress-1>", lambda e: self.focus_set(), add="+")

        if self.own_windows:
            # The child windows follow this frame by polling its place on the
            # screen (`_follow`): macOS delivers no <Configure> while the
            # main window is being dragged, so an event-driven placement left
            # them behind. Hide, show and focus changes of the main window
            # re-place and re-stack them at once.
            top = self.winfo_toplevel()
            for event in ("<Map>", "<Unmap>", "<Visibility>", "<FocusIn>", "<Activate>"):
                top.bind(event, self._on_main_window_event, add="+")
            # A click brings the main window to the front on its own; the
            # children must come along or the LCD vanishes behind the photo.
            top.bind("<ButtonPress>", lambda e: self._place_windows(lift=True), add="+")
            self.bind("<Destroy>", self._stop_following, add="+")
            self._follow_id: str | None = None
            self._follow()

    # ----- the three canvases -----

    def _canvas(self, x: int, y: int, w: int, h: int, bg: str,
                over: tk.Toplevel | None = None) -> tk.Canvas:
        """A canvas at (x, y) of this frame: in a child window of its own
        on macOS (see the module docstring; a child of `over`, or of the
        main window), placed in the frame elsewhere."""
        if self.own_windows:
            win = tk.Toplevel(self, bg=bg)
            win.withdraw()
            win.overrideredirect(True)
            win.transient(over if over is not None else self.winfo_toplevel())
            # No drop shadow: it draws a dark edge that shows as a seam on
            # the photo. Tk sets the window's style each time it maps it,
            # so ours goes on top of that, on every <Map>.
            win.bind("<Map>", lambda e, w=win: self._no_shadow(w), add="+")
            canvas = tk.Canvas(win, width=w, height=h, highlightthickness=0, bd=0, bg=bg)
            canvas.pack()
            self._windows.append((win, x, y, w, h))
            return canvas
        canvas = tk.Canvas(self, width=w, height=h, highlightthickness=0, bd=0, bg=bg)
        canvas.place(x=x, y=y)
        return canvas

    @staticmethod
    def _no_shadow(win: tk.Toplevel) -> None:
        try:
            win.tk.call("::tk::unsupported::MacWindowStyle", "style", win._w, "simple",
                        ["noActivates", "noShadow"])
        except tk.TclError:
            pass

    def _follow(self) -> None:
        """Poll: keep the child windows on this frame, every frame."""
        self._follow_id = None
        if not self._place_windows():
            return                                  # gone
        try:
            self._follow_id = self.after(FOLLOW_MS, self._follow)
        except tk.TclError:
            pass

    def _stop_following(self, _event=None) -> None:
        if self._follow_id is not None:
            try:
                self.after_cancel(self._follow_id)
            except tk.TclError:
                pass
            self._follow_id = None

    def _on_main_window_event(self, event) -> None:
        """A hide / show / focus change of the main window itself (the
        binding is on the toplevel, so it also fires for every widget in
        it; only the toplevel's own events matter here)."""
        try:
            if event.widget is self.winfo_toplevel():
                self._place_windows(lift=True)
        except tk.TclError:                 # the main window is closing; this frame is gone
            pass

    def _place_windows(self, lift: bool = False) -> bool:
        """Keep the child windows on this frame: shown and positioned while
        the frame is viewable, hidden otherwise (a withdrawn or iconified
        main window, or a test that never shows one). Stacked in creation
        order (device, then the pad and the screen over it) whenever one
        was shown or moved, or `lift` asks for it, so they never end up
        underneath the main window. False once the widgets are gone."""
        try:
            viewable = self.winfo_viewable()
            ox, oy = self.winfo_rootx(), self.winfo_rooty()
            restack = lift
            for win, x, y, w, h in self._windows:
                if not viewable:
                    if win.state() != "withdrawn":
                        win.withdraw()
                    continue
                wanted = f"{w}x{h}+{ox + x}+{oy + y}"
                if self._placed.get(win) != wanted:
                    self._placed[win] = wanted
                    win.geometry(wanted)
                    restack = True
                if win.state() != "normal":
                    win.deiconify()
                    # Tk on macOS keeps a borderless transient window above
                    # every application (the "utility" level: the Game Boy
                    # stayed on top of the browser), and `deiconify` sets
                    # that again after the window is mapped, so it is
                    # undone here, not in a <Map> binding.
                    win.attributes("-topmost", False)
                    restack = True
            if restack and viewable:
                for win, *_ in self._windows:
                    win.lift()
            return True
        except tk.TclError:                 # the window is going away
            return False

    # ----- screen -----

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.geo.screen_size

    def set_screen(self, image: Image.Image) -> None:
        """Show a PIL `image` (`screen_size`) on the LCD, in place."""
        self.screen_image.paste(image)

    # ----- the ON light -----

    @property
    def power(self) -> bool:
        return self._power

    def set_power(self, on: bool) -> None:
        if on == self._power:
            return
        self._power = on
        self.device.itemconfig(self._led, image=self._led_on if on else self._led_off)

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
        self.pad.itemconfig(self._pad_id, image=self._pad_image(pressed))

    def button_at(self, x: int, y: int) -> str | None:
        """The button under device position (x, y), if any."""
        for key, (_kind, (x0, y0, x1, y1)) in self.geo.regions.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return key
        return None

    def _button_under(self, event) -> str | None:
        """The button under a pad-canvas event."""
        return self.button_at(event.x + self.geo.pad_box[0], event.y + self.geo.pad_box[1])

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

    def _on_motion(self, event) -> None:
        key = self._button_under(event)
        if key == self._hover:
            return
        self._hover = key
        self.pad.config(cursor="hand2" if key else "")
        self._tip.hide()
        if key:
            self._tip.schedule()

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
        key = self._button_under(event)
        self._report_mouse(frozenset({key}) if key else frozenset())

    def _on_mouse_up(self, _event=None) -> None:
        self._report_mouse(frozenset())
