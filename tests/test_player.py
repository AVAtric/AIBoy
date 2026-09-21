"""The playback engine's emulator-free parts: the sound gate, when sound is
possible, and the boot video's sound switch."""
import threading
import unittest
from unittest import mock

from aiboy.gui import player


class _FakeSDL:
    """Enough of PySDL2 for the gate: device ids 2 and 3 are open."""
    SDL_AUDIO_STOPPED, SDL_AUDIO_PLAYING, SDL_AUDIO_PAUSED = 0, 1, 2

    def __init__(self):
        self.status = {2: self.SDL_AUDIO_PLAYING, 3: self.SDL_AUDIO_PLAYING}
        self.cleared = []

    def SDL_GetAudioDeviceStatus(self, dev):
        return self.status.get(dev, self.SDL_AUDIO_STOPPED)

    def SDL_PauseAudioDevice(self, dev, pause):
        self.status[dev] = self.SDL_AUDIO_PAUSED if pause else self.SDL_AUDIO_PLAYING

    def SDL_ClearQueuedAudio(self, dev):
        self.cleared.append(dev)


class SoundGateTests(unittest.TestCase):
    def test_open_devices_are_the_ones_sdl_reports_as_not_stopped(self):
        sdl = _FakeSDL()
        self.assertEqual(player.open_audio_devices(sdl), [2, 3])
        sdl.status = {}
        self.assertEqual(player.open_audio_devices(sdl), [])

    def test_gate_follows_the_wish_and_only_acts_on_a_change(self):
        sdl = _FakeSDL()
        wanted = threading.Event()
        gate = player.SoundGate(wanted, sdl)
        gate.apply()                                   # not wanted: muted, queue dropped
        self.assertEqual(sdl.status, {2: sdl.SDL_AUDIO_PAUSED, 3: sdl.SDL_AUDIO_PAUSED})
        self.assertEqual(sdl.cleared, [2, 3])
        gate.apply()                                   # nothing changed: no SDL calls
        self.assertEqual(sdl.cleared, [2, 3])
        wanted.set()
        gate.apply()                                   # wanted: playing, stale queue dropped first
        self.assertEqual(sdl.status, {2: sdl.SDL_AUDIO_PLAYING, 3: sdl.SDL_AUDIO_PLAYING})
        self.assertEqual(sdl.cleared, [2, 3, 2, 3])
        wanted.clear()
        gate.apply()
        self.assertEqual(sdl.status[2], sdl.SDL_AUDIO_PAUSED)

    def test_gate_without_sdl_does_nothing(self):
        gate = player.SoundGate(threading.Event(), None)
        gate.apply()                                   # no error, nothing to do

    def test_sound_only_at_real_speed(self):
        self.assertTrue(player.sound_possible(1.0))
        for speed in (0.0, 0.5, 2.0, 4.0):
            self.assertFalse(player.sound_possible(speed), speed)
        for label, mult in player.SPEED_CHOICES:
            self.assertEqual(player.sound_possible(mult), label.startswith("1×"), label)

    def test_player_set_sound_drives_the_event(self):
        p = player.EmbeddedPlayer(player.LatestFrame(), __import__("queue").Queue())
        self.assertFalse(p.sound_wanted.is_set())
        p.set_sound(True)
        self.assertTrue(p.sound_wanted.is_set())
        p.set_sound(False)
        self.assertFalse(p.sound_wanted.is_set())


class IntroSoundTests(unittest.TestCase):
    def test_boot_video_always_plays_its_jingle(self):
        """The Sound checkbox is for the games; the boot video plays its
        sound like a real Game Boy (AIBOY_NO_INTRO skips the video)."""
        import numpy as np
        video = player.IntroVideo.__new__(player.IntroVideo)
        video.frames = np.zeros((3, 144, 160, 3), dtype=np.uint8)
        video.fps = 600.0
        video.idle_index = 2
        video.sound_path = "nowhere.wav"
        done = threading.Event()
        with mock.patch.object(player, "play_sound", return_value=None) as play_sound:
            t = video.play(player.LatestFrame(), threading.Event(), done.set)
            t.join(timeout=5)
        self.assertTrue(done.is_set())
        play_sound.assert_called_once()


if __name__ == "__main__":
    unittest.main()
