"""Human input (aiboy.gui.controls): the control map and its persistence,
key names, console button names, stick directions, press/release diffs
and the action names the picture lights."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from aiboy import settings
from aiboy.gui import controls
from aiboy.gui.controls import (BUTTONS, DEFAULT_KEYS, DEFAULT_PAD, PAD_INPUTS, ControlMap,
                                HeldButtons, action_name, axis_inputs, diff_presses, key_label,
                                normalize_keysym, pad_kind_of, pad_label)
from aiboy.gui.gameboy import buttons_for_action


class DefaultsTests(unittest.TestCase):
    def test_every_button_is_reachable_from_the_keyboard_and_the_pad(self):
        self.assertEqual(set(DEFAULT_KEYS), set(BUTTONS))
        self.assertEqual(set(DEFAULT_PAD), set(BUTTONS))
        for button in BUTTONS:
            self.assertTrue(DEFAULT_KEYS[button], button)
            self.assertTrue(DEFAULT_PAD[button], button)
            for raw in DEFAULT_PAD[button]:
                self.assertIn(raw, PAD_INPUTS, raw)

    def test_no_input_drives_two_buttons(self):
        for table in (DEFAULT_KEYS, DEFAULT_PAD):
            names = [n for names in table.values() for n in names]
            self.assertEqual(len(names), len(set(names)))

    def test_switch_pro_labels_line_up(self):
        # A Switch Pro's A (label) is the Game Boy's A; B is B; + is START, − SELECT.
        m = ControlMap()
        self.assertEqual(m.pad_to_button["a"], "a")
        self.assertEqual(m.pad_to_button["b"], "b")
        self.assertEqual(m.pad_to_button["start"], "start")
        self.assertEqual(m.pad_to_button["back"], "select")
        self.assertEqual(pad_label("back", "switch"), "−")
        self.assertEqual(pad_label("a", "playstation"), "✕")
        self.assertEqual(pad_label("start", "xbox"), "Menu")
        self.assertEqual(pad_label("dpad_up", "generic"), "D-pad ↑")
        self.assertEqual(pad_kind_of("SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_PRO"), "switch")
        self.assertEqual(pad_kind_of("SDL_CONTROLLER_TYPE_PS5"), "playstation")
        self.assertEqual(pad_kind_of("SDL_CONTROLLER_TYPE_XBOXONE"), "xbox")
        self.assertEqual(pad_kind_of("SDL_CONTROLLER_TYPE_UNKNOWN"), "generic")


class KeyNamesTests(unittest.TestCase):
    def test_normalised_keysyms(self):
        self.assertEqual(normalize_keysym("A"), "a")
        self.assertEqual(normalize_keysym("Shift_R"), "Shift")
        self.assertEqual(normalize_keysym("Meta_L"), "Meta")
        self.assertEqual(normalize_keysym("Return"), "Return")

    def test_labels(self):
        self.assertEqual(key_label("Up"), "↑")
        self.assertEqual(key_label("w"), "W")
        self.assertEqual(key_label("space"), "Space")
        self.assertEqual(key_label("Return"), "Enter")
        self.assertEqual(key_label("KP_5"), "5 (keypad)")


class ControlMapTests(unittest.TestCase):
    def test_default_texts(self):
        m = ControlMap()
        self.assertTrue(m.is_default())
        self.assertEqual(m.key_text("up"), "↑ or W")
        self.assertEqual(m.pad_text("select", "switch"), "−")
        self.assertIn("A = X or K or Space", m.help_text())
        self.assertIn("Controller (My Pad)", m.help_text("switch", "My Pad"))

    def test_binding_replaces_and_moves_an_input(self):
        m = ControlMap()
        m.bind_key("a", "M")                      # replaces X / K / Space, normalised
        self.assertEqual(m.keys["a"], ["m"])
        self.assertEqual(m.key_to_button["m"], "a")
        self.assertNotIn("x", m.key_to_button)
        m.bind_key("b", "m")                      # a key drives one button only
        self.assertEqual(m.keys["a"], [])
        self.assertEqual(m.keys["b"], ["m"])
        m.bind_key("b", "n", add=True)
        self.assertEqual(m.keys["b"], ["m", "n"])
        m.bind_pad("start", "guide")
        self.assertEqual(m.pad["start"], ["guide"])
        m.bind_pad("select", "guide", add=True)
        self.assertEqual(m.pad["start"], [])
        self.assertEqual(m.pad["select"], ["back", "guide"])
        with self.assertRaises(ValueError):
            m.bind_pad("a", "nonsense")
        with self.assertRaises(ValueError):
            m.bind_key("c", "x")
        m.clear_pad("select")
        self.assertEqual(m.pad_text("select"), "—")
        m.reset()
        self.assertTrue(m.is_default())

    def test_round_trip_through_the_settings_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            old = settings.SETTINGS_FILE
            settings.SETTINGS_FILE = path
            try:
                m = ControlMap.load()
                self.assertTrue(m.is_default())
                m.bind_key("a", "m"); m.bind_pad("b", "triggerright")
                m.save()
                saved = json.loads(path.read_text())["controls"]
                self.assertEqual(saved["keys"]["a"], ["m"])
                self.assertEqual(saved["pad"]["b"], ["triggerright"])
                again = ControlMap.load()
                self.assertEqual(again.to_dict(), m.to_dict())
                self.assertEqual(settings.get("auto_improve_presets"), True)   # untouched
            finally:
                settings.SETTINGS_FILE = old

    def test_garbage_in_the_settings_falls_back(self):
        m = ControlMap(keys={"a": "m", "zz": ["q"], "b": 5}, pad={"a": ["a", "nonsense"], "up": 3})
        self.assertEqual(m.keys["a"], ["m"])
        self.assertEqual(m.keys["b"], [])
        self.assertEqual(m.pad["a"], ["a"])
        self.assertEqual(m.pad["up"], [])
        self.assertTrue(ControlMap(keys="no", pad=None).is_default())


class HeldButtonsTests(unittest.TestCase):
    def test_union_of_sources(self):
        held = HeldButtons()
        self.assertEqual(held.held(), frozenset())
        held.set("keyboard", {"right"})
        held.set("gamepad", {"a"})
        self.assertEqual(held.held(), {"right", "a"})
        held.set("keyboard", set())
        self.assertEqual(held.held(), {"a"})
        held.clear()
        self.assertEqual(held.held(), frozenset())

    def test_thread_safe_updates(self):
        held = HeldButtons()

        def hammer(source):
            for i in range(2000):
                held.set(source, {BUTTONS[i % len(BUTTONS)]})
                held.held()
        threads = [threading.Thread(target=hammer, args=(f"s{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(held.held() <= set(BUTTONS))


class AxisAndDiffTests(unittest.TestCase):
    def test_axis_inputs_with_deadzone(self):
        self.assertEqual(axis_inputs("leftx", 0.0), frozenset())
        self.assertEqual(axis_inputs("leftx", 0.3), frozenset())
        self.assertEqual(axis_inputs("leftx", 0.9), {"leftx+"})
        self.assertEqual(axis_inputs("lefty", -1.0), {"lefty-"})
        self.assertEqual(axis_inputs("triggerright", 0.8), {"triggerright"})
        self.assertEqual(axis_inputs("triggerright", -1.0), frozenset())

    def test_press_release_diff_in_button_order(self):
        pressed, released = diff_presses(frozenset({"right", "a"}), frozenset({"right", "b"}))
        self.assertEqual(pressed, ["b"])
        self.assertEqual(released, ["a"])
        pressed, released = diff_presses(frozenset(), frozenset(BUTTONS))
        self.assertEqual(pressed, list(BUTTONS))
        self.assertEqual(released, [])

    def test_action_names_light_the_right_buttons(self):
        self.assertEqual(action_name(frozenset()), "NOOP")
        self.assertEqual(action_name(frozenset({"a", "right", "b"})), "RIGHT+A+B")
        for combo in ({"right", "a"}, {"up"}, {"start", "select"}, set(BUTTONS)):
            self.assertEqual(buttons_for_action(action_name(frozenset(combo))), combo)


class GamepadTests(unittest.TestCase):
    def test_poller_starts_and_stops(self):
        seen = []
        held = HeldButtons()
        m = ControlMap()
        pad = controls.Gamepad(held, m, on_change=seen.append)
        pad.start()
        pad.stop()
        pad.join(timeout=10)
        self.assertFalse(pad.is_alive())
        self.assertEqual(held.held(), frozenset())
        self.assertTrue(all(isinstance(n, str) for n in seen))     # only names, when a pad exists
        pad.raw_held = frozenset({"a", "dpad_left"})              # remap applies a new table
        m.bind_pad("select", "a")
        pad.remap()
        self.assertEqual(held.held(), {"select", "left"})


if __name__ == "__main__":
    unittest.main()
