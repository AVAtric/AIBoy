"""Build a self-contained AIboy app for the computer this runs on.

    python build_release.py                # -> release/AIboy/ and a zip next to it
    python build_release.py --no-roms      # leave the ROMs folder empty (for handing the build on)
    python build_release.py --no-selftest  # skip running the built app afterwards
    python build_release.py --no-zip

Works the same way on macOS, Windows and Linux; run it in the Python
environment that runs `python main.py`. If PyInstaller is missing it is
installed into that environment first (with pip).

What it does
  1. Profiles this machine (cores, performance cores, memory, chip) and writes
     system_profile.json: the number of emulators the release runs in parallel
     is chosen from it, so every built-in preset uses the cores it has and no
     more (see paths.recommended_n_envs / presets.fit_to_machine).
  2. Runs PyInstaller (one-folder build; a windowed AIboy.app on macOS, an
     AIboy/ folder with AIboy.exe on Windows, AIboy/AIboy on Linux) with the
     boot video, the built-in presets and the profile inside.
     On macOS the bundle-root symlink `SDL2` PyInstaller makes is removed:
     it shadows the `sdl2` Python package on a case-insensitive disk, which
     left the built app without game-controller support.
  3. Assembles release/AIboy/: the app, a ROMs/ folder (your ROM files are
     copied in unless --no-roms), empty models/ and assets/ folders and a
     README.txt. The app keeps its own files (models, presets,
     experience.jsonl, original artwork in assets/) next to itself, never
     inside, so a newer build can replace it in place.
  4. Self-test: the built app probes a ROM and trains for a few hundred steps
     with two emulator processes, which exercises the frozen-app process
     spawning that training depends on.

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
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "AIboy"
PACKAGE = "aiboy"
BUILD_DIR = ROOT / "build"
RELEASE_DIR = ROOT / "release" / APP_NAME
PROFILE_FILE = ROOT / "system_profile.json"
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


def ensure_pyinstaller():
    """Import PyInstaller, installing it into this environment if needed."""
    try:
        import PyInstaller.__main__ as pyi
        return pyi
    except ImportError:
        pass
    print("[build] PyInstaller is not installed; installing it with pip…")
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"]).returncode
    if rc != 0:
        sys.exit("[build] could not install PyInstaller; run `pip install pyinstaller` and retry")
    try:
        import PyInstaller.__main__ as pyi
        return pyi
    except ImportError:
        sys.exit("[build] PyInstaller still not importable after installing it")


def shipped_assets(sep: str) -> list[str]:
    """`--add-data` pairs for assets/: the shipped files only. A local
    `orig_*` original (the real device photo or boot video, see
    aiboy.paths.local_or_shipped) is for the developer's own window and must
    never travel in a release; whoever owns one puts it into the assets/
    folder next to the built app."""
    args: list[str] = []
    for f in sorted((ROOT / "assets").iterdir()):
        if f.is_file() and not f.name.startswith("orig_") and not f.name.startswith("."):
            args += ["--add-data", f"{f}{sep}assets"]
    return args


def run_pyinstaller(profile: dict) -> Path:
    pyi = ensure_pyinstaller()
    PROFILE_FILE.write_text(json.dumps(profile, indent=2))
    sep = os.pathsep
    args = [
        str(ROOT / "main.py"), "--name", APP_NAME, "--noconfirm", "--clean", "--onedir",
        "--distpath", str(BUILD_DIR / "dist"), "--workpath", str(BUILD_DIR / "work"),
        "--specpath", str(BUILD_DIR),
        *shipped_assets(sep),
        "--add-data", f"{ROOT / 'builtin_presets.json'}{sep}.",
        "--add-data", f"{PROFILE_FILE}{sep}.",
        "--collect-all", "pyboy",             # Cython modules + the plugin folder PyBoy imports by name
        "--collect-all", "sdl2dll",           # the SDL2 frameworks / DLLs
        "--collect-submodules", "sdl2",
        "--collect-all", "stable_baselines3",  # reads its version.txt at import
        "--collect-all", "gymnasium",
        "--collect-all", "tensorboard",       # its web UI is a data file (webfiles.zip)
        "--collect-submodules", PACKAGE,      # the tabs are imported lazily; include every module
        "--paths", str(ROOT),
        "--exclude-module", "pytest",
        "--exclude-module", "pygame",         # pulled in by PySDL2's examples; never used
    ]
    if sys.platform in ("darwin", "win32"):
        args.append("--windowed")
    if sys.platform == "darwin":
        args += ["--osx-bundle-identifier", "com.aiboy.app"]
    icon = make_icon()
    if icon is not None:
        args += ["--icon", str(icon)]
    print(f"[build] pyinstaller {' '.join(args)}")
    pyi.run(args)
    if sys.platform == "darwin":
        built = BUILD_DIR / "dist" / f"{APP_NAME}.app"
        remove_package_shadows(built)
        return built
    return BUILD_DIR / "dist" / APP_NAME


# Python packages whose name, on a case-insensitive file system, a top-level
# file in the bundle may shadow. PyInstaller symlinks every macOS framework
# binary to the bundle's root (`SDL2` -> sdl2dll/dll/SDL2.framework/…/SDL2);
# with that file in place `import sdl2.dll` fails inside the app ("No module
# named 'sdl2.dll'") and the game controller is silently off. PySDL2 loads
# the library through pysdl2-dll's own path, so the symlink is not needed.
SHADOWED_PACKAGES = ("sdl2",)


def remove_package_shadows(app: Path) -> list[Path]:
    """Delete bundle-root entries that collide with SHADOWED_PACKAGES by
    case-insensitive name. Returns what was removed."""
    removed: list[Path] = []
    for folder in (app / "Contents" / "Frameworks", app / "Contents" / "Resources"):
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            if entry.name.lower() in SHADOWED_PACKAGES and not entry.is_dir():
                entry.unlink()
                removed.append(entry)
    for entry in removed:
        print(f"[build] removed {entry.relative_to(app)}: it would shadow the Python package")
    return removed


# ------------------------- 3. release folder -------------------------

README = """AIboy — teach an AI to play Super Mario Land
============================================

1. Put your own Super Mario Land ROM file, named mario.gb, into the ROMs folder
   next to the app. (The app does not include any game.)
2. Start {app}. The Wizard tab opens: pick what the agent should learn and press
   Start. AIboy looks for good settings, trains, and then plays on the Game Boy
   screen. Everything it makes lands in the models folder next to the app.
3. AIboy remembers every setting it tries (experience.jsonl next to the app):
   the next search skips what it already knows and explores what it doesn't,
   and the Experience tab shows everything it has learned so far.
4. The other tabs (Train, Tune, Presets) show and control the same runs in full.
5. The Game Boy on the screen carries AIboy lettering and plays an AIboy boot
   video. If you own the real artwork, put it into the assets folder next to
   the app as orig_gb_interface.png (the photo), orig_gb_intro.npz and
   orig_gb_intro.wav (the boot video and its chime); AIboy shows those instead.

Built {built_at} on {chip}: {cores} cores, {memory_gb} GB memory.
This build runs {n_envs} game emulators in parallel while training. It is made
for this kind of computer ({platform}, {machine}); build again on another kind
with `python build_release.py` from the source folder.

If macOS refuses to open the app, right-click it and choose Open once.
Errors are written to gui_errors.log next to the app.
"""


def assemble_release(built: Path, profile: dict, with_roms: bool) -> Path:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)
    shutil.copytree(built, RELEASE_DIR / built.name, symlinks=True)
    (RELEASE_DIR / "models").mkdir()
    (RELEASE_DIR / "assets").mkdir()          # for the user's own artwork (paths.LOCAL_ASSET_DIR)
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


SELFTEST_TB_PORT = 6099


def check_tensorboard(exe: Path, logdir: Path) -> str | None:
    """The TensorBoard button runs `<app> tensorboard`; make sure the bundled
    copy serves its web page (a missing webfiles.zip or plugin would only
    show up when a user clicks the button). Returns what went wrong, or None."""
    print("[selftest] serving the run's curves with the built app's TensorBoard…")
    log_path = BUILD_DIR / "selftest-tensorboard.log"
    with log_path.open("w") as log:                 # a file: a full pipe would stall the child
        proc = subprocess.Popen([str(exe), "tensorboard", "--logdir", str(logdir),
                                 "--port", str(SELFTEST_TB_PORT)],
                                stdout=log, stderr=subprocess.STDOUT)
    url = f"http://localhost:{SELFTEST_TB_PORT}/"
    deadline = time.monotonic() + 90
    page = ""
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    page = r.read(4000).decode("utf-8", "replace")
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    if "tensorboard" not in page.lower():
        out = log_path.read_text(errors="replace")
        return (f"[selftest] the built app's TensorBoard did not answer on {url} "
                f"(rc={proc.returncode}):\n{out[-3000:]}")
    print(f"[selftest] OK — TensorBoard answers on {url}")
    return None


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
    tb_error = None
    if ok:
        print("[selftest] OK — the built app trains with parallel emulators")
        tb_error = check_tensorboard(exe, run_dir.parent)
    # Clean up before any exit: the release must never ship the test run.
    shutil.rmtree(run_dir, ignore_errors=True)
    (release / "experience.jsonl").unlink(missing_ok=True)      # the app ships with no memory
    if not ok:
        sys.exit(f"[selftest] training failed (rc={out.returncode}):\n{out.stdout[-3000:]}\n"
                 f"{out.stderr[-3000:]}")
    if tb_error:
        sys.exit(tb_error)
    print("[selftest] checking game-controller support in the built app…")
    out = subprocess.run([str(exe), "controller-test", "--seconds", "1"], capture_output=True,
                         text=True, timeout=60)
    if "controllers are off" in out.stdout:
        sys.exit(f"[selftest] the built app cannot load SDL2:\n{out.stdout}")
    print("[selftest] OK — SDL2 loads in the built app (a pad is picked up when connected)")


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
