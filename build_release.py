"""Build a self-contained release of Game Boy AI for the computer this runs on.

    python build_release.py                # -> release/GameBoyAI/ and a zip next to it
    python build_release.py --no-roms      # leave the ROMs folder empty (for handing the build on)
    python build_release.py --no-selftest  # skip running the built app afterwards
    python build_release.py --no-zip

What it does
  1. Profiles this machine (cores, performance cores, memory, chip) and writes
     system_profile.json: the number of emulators the release runs in parallel
     is chosen from it, so every built-in preset uses the cores it has and no
     more (see paths.recommended_n_envs / presets.fit_to_machine).
  2. Runs PyInstaller (one-folder build; a windowed GameBoyAI.app on macOS)
     with the boot video, the built-in presets and the profile inside.
  3. Assembles release/GameBoyAI/: the app, a ROMs/ folder (your ROM files are
     copied in unless --no-roms), an empty models/ folder and a README.txt.
  4. Self-test: the built app probes a ROM and trains for a few hundred steps
     with two emulator processes, which exercises the frozen-app process
     spawning that training depends on.

Needs PyInstaller in the same environment as the app (`pip install pyinstaller`).
The build is for this machine's platform and CPU type; build on each kind of
machine you want to ship to.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "GameBoyAI"
BUILD_DIR = ROOT / "build"
RELEASE_DIR = ROOT / "release" / APP_NAME
PROFILE_FILE = ROOT / "system_profile.json"
LOCAL_MODULES = ("env", "games", "gui", "main", "paths", "player", "presets", "presets_tab",
                 "runs", "tuning", "widgets", "wizard")
MAX_ENVS = 12                 # beyond this the PPO update, not the rollout, dominates
MEMORY_PER_ENV_GB = 0.25      # one PyBoy worker with its share of the trainer, generous
MEMORY_SHARE = 0.5            # never plan to use more than half of the machine's memory


# ------------------------- 1. machine profile -------------------------

def _sysctl(key: str) -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                                  check=True).stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def memory_bytes() -> int | None:
    if sys.platform == "darwin":
        return _sysctl("hw.memsize")
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            return None
    if sys.platform.startswith("win"):
        import ctypes
        kernel = ctypes.windll.kernel32                              # type: ignore[attr-defined]
        total = ctypes.c_ulonglong(0)
        if kernel.GetPhysicallyInstalledSystemMemory(ctypes.byref(total)):
            return int(total.value) * 1024
    return None


def profile_machine() -> dict:
    cores = os.cpu_count() or 4
    perf = _sysctl("hw.perflevel0.logicalcpu") if sys.platform == "darwin" else None
    mem = memory_bytes()
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip() or chip
        except (OSError, subprocess.CalledProcessError):
            pass
    n_envs = min(MAX_ENVS, cores)
    if mem:
        n_envs = min(n_envs, max(1, int(mem / 2**30 * MEMORY_SHARE / MEMORY_PER_ENV_GB)))
    return {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform, "machine": platform.machine(), "chip": chip,
        "cores": cores, "performance_cores": perf,
        "memory_gb": round(mem / 2**30, 1) if mem else None,
        "n_envs": max(1, n_envs),
        "python": platform.python_version(),
    }


# ------------------------- 2. PyInstaller -------------------------

def make_icon() -> Path | None:
    """App icon from the boot video's logo frame (macOS .icns / Windows .ico)."""
    try:
        import numpy as np
        from PIL import Image
        frames = np.load(ROOT / "assets" / "gb_intro.npz")
        img = Image.fromarray(frames["frames"][int(frames["idle_index"])])
        img = img.crop((0, 40, 160, 104)).resize((512, 512), Image.NEAREST)   # the logo band
        suffix = ".icns" if sys.platform == "darwin" else ".ico"
        out = BUILD_DIR / f"{APP_NAME}{suffix}"
        BUILD_DIR.mkdir(exist_ok=True)
        img.save(out)
        return out
    except Exception as e:                                  # an icon is a nicety, never a blocker
        print(f"[build] no icon: {e}")
        return None


def run_pyinstaller(profile: dict) -> Path:
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        sys.exit("PyInstaller is not installed in this environment: pip install pyinstaller")
    PROFILE_FILE.write_text(json.dumps(profile, indent=2))
    sep = os.pathsep
    args = [
        str(ROOT / "main.py"), "--name", APP_NAME, "--noconfirm", "--clean", "--onedir",
        "--distpath", str(BUILD_DIR / "dist"), "--workpath", str(BUILD_DIR / "work"),
        "--specpath", str(BUILD_DIR),
        "--add-data", f"{ROOT / 'assets'}{sep}assets",
        "--add-data", f"{ROOT / 'builtin_presets.json'}{sep}.",
        "--add-data", f"{PROFILE_FILE}{sep}.",
        "--collect-all", "pyboy",             # Cython modules + the plugin folder PyBoy imports by name
        "--collect-all", "sdl2dll",           # the SDL2 frameworks / DLLs
        "--collect-submodules", "sdl2",
        "--collect-all", "stable_baselines3",  # reads its version.txt at import
        "--collect-all", "gymnasium",
        "--collect-submodules", "tensorboard",
        "--exclude-module", "pytest",
    ]
    for mod in LOCAL_MODULES:
        args += ["--hidden-import", mod]
    if sys.platform in ("darwin", "win32"):
        args.append("--windowed")
    if sys.platform == "darwin":
        args += ["--osx-bundle-identifier", "com.gameboyai.app"]
    icon = make_icon()
    if icon is not None:
        args += ["--icon", str(icon)]
    print(f"[build] pyinstaller {' '.join(args)}")
    pyi.run(args)
    if sys.platform == "darwin":
        return BUILD_DIR / "dist" / f"{APP_NAME}.app"
    return BUILD_DIR / "dist" / APP_NAME


# ------------------------- 3. release folder -------------------------

README = """Game Boy AI — train an agent to play Super Mario Land
======================================================

1. Put your own Super Mario Land ROM file, named mario.gb, into the ROMs folder
   next to the app. (The app does not include any game.)
2. Start {app}. The Wizard tab opens: pick what the agent should learn and press
   Start. It looks for good settings, trains, and then plays on the Game Boy
   screen. Everything it makes lands in the models folder next to the app.
3. Advanced tabs (Train, Tune, Presets) show and control the same runs in full.

Built {built_at} on {chip}: {cores} cores, {memory_gb} GB memory.
This build runs {n_envs} game emulators in parallel while training. It is made
for this kind of computer ({platform}, {machine}); build again on another kind.

If macOS refuses to open the app, right-click it and choose Open once.
Errors are written to gui_errors.log next to the app.
"""


def assemble_release(built: Path, profile: dict, with_roms: bool) -> Path:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)
    shutil.copytree(built, RELEASE_DIR / built.name, symlinks=True)
    (RELEASE_DIR / "models").mkdir()
    roms = RELEASE_DIR / "ROMs"
    roms.mkdir()
    if with_roms and (ROOT / "ROMs").exists():
        for rom in (ROOT / "ROMs").iterdir():
            if rom.suffix.lower() in (".gb", ".gbc"):
                shutil.copy2(rom, roms / rom.name)
    (RELEASE_DIR / "README.txt").write_text(README.format(app=built.name, **profile))
    return RELEASE_DIR


def zip_release(release: Path, profile: dict) -> Path:
    tag = f"{profile['platform']}-{profile['machine']}"
    out = release.parent / f"{APP_NAME}-{tag}.zip"
    if out.exists():
        out.unlink()
    if sys.platform == "darwin":
        # ditto keeps the .app's symlinks, permissions and code signature intact
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(release), str(out)], check=True)
    else:
        shutil.make_archive(str(out.with_suffix("")), "zip", release.parent, release.name)
    return out


# ------------------------- 4. self-test -------------------------

def executable(release: Path) -> Path:
    if sys.platform == "darwin":
        return release / f"{APP_NAME}.app" / "Contents" / "MacOS" / APP_NAME
    exe = release / APP_NAME / (APP_NAME + (".exe" if sys.platform.startswith("win") else ""))
    return exe


def selftest(release: Path) -> None:
    exe = executable(release)
    rom = release / "ROMs" / "mario.gb"
    if not rom.exists():
        print("[selftest] no ROMs/mario.gb in the release; skipping")
        return
    print("[selftest] probing the ROM with the built app…")
    out = subprocess.run([str(exe), "probe-rom", str(rom)], capture_output=True, text=True,
                         timeout=120)
    if out.returncode != 0 or '"title"' not in out.stdout:
        sys.exit(f"[selftest] probe failed (rc={out.returncode}):\n{out.stdout}\n{out.stderr[-2000:]}")
    print(f"[selftest] {out.stdout.strip().splitlines()[-1]}")
    print("[selftest] training 600 steps with 2 emulator processes…")
    out = subprocess.run([str(exe), "train", "--n-envs", "2", "--timesteps", "600", "--n-steps", "64",
                          "--batch-size", "64", "--eval-freq", "600", "--checkpoint-freq", "600",
                          "--n-eval-episodes", "1", "--run-name", "selftest"],
                         capture_output=True, text=True, timeout=600)
    run_dir = release / "models" / "mario" / "selftest"
    ok = out.returncode == 0 and (run_dir / "checkpoints" / "final.zip").exists()
    shutil.rmtree(run_dir, ignore_errors=True)
    if not ok:
        sys.exit(f"[selftest] training failed (rc={out.returncode}):\n{out.stdout[-3000:]}\n"
                 f"{out.stderr[-3000:]}")
    print("[selftest] OK — the built app trains with parallel emulators")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-roms", action="store_true", help="do not copy ROMs/ into the release")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    t0 = time.time()
    profile = profile_machine()
    print(f"[build] {profile['chip']}: {profile['cores']} cores, {profile['memory_gb']} GB "
          f"-> {profile['n_envs']} parallel emulators")
    built = run_pyinstaller(profile)
    release = assemble_release(built, profile, with_roms=not args.no_roms)
    print(f"[build] release folder: {release}")
    if not args.no_selftest:
        selftest(release)
    if not args.no_zip:
        print(f"[build] zip: {zip_release(release, profile)}")
    print(f"[build] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
