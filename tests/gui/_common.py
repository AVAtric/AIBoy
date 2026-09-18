"""Shared set-up for the GUI checks: run from the project root, isolate the
wizard's trial prefix, and silence dialogs so the checks can run unattended.

These are not unit tests (they open windows and, for the end-to-end checks,
train real models for ~30 s). Run them one at a time with the app closed:

    python tests/gui/check_smoke.py
    python tests/gui/check_games.py
    python tests/gui/check_layout.py
    python tests/gui/check_e2e.py          # search -> preset -> train -> watch
    python tests/gui/check_e2e_auto.py     # the same via the Auto-complete box

Every check prints a final OK line and cleans up everything it created.
"""
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("AIBOY_WIZARD_PREFIX", "e2etest")
os.environ.setdefault("AIBOY_NO_INTRO", "1")           # no boot video / sound during checks
# The checks train real (tiny) models; their results must not enter the
# user's real experience file, so they get one of their own (inherited by
# the trainer subprocesses through the environment).
import tempfile
EXPERIENCE_FILE = Path(tempfile.gettempdir()) / "aiboy-e2etest-experience.jsonl"
os.environ.setdefault("AIBOY_EXPERIENCE_FILE", str(EXPERIENCE_FILE))

TEST_PRESETS = ("__check_preset", "__check_copy", "__check_copy2", "__check_new",
                "__check_e2e_preset")
T0 = time.time()
MARKER = 0.0987          # ent_coef value the smoke check writes into a built-in override


def say(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def silence_dialogs(gui_module, answer_yes: bool = True):
    """Replace message boxes with recorders; returns the recorder dict."""
    rec = {"errors": [], "infos": [], "warnings": []}
    gui_module.messagebox.showerror = lambda t, m, **k: rec["errors"].append(str(m))
    gui_module.messagebox.showinfo = lambda t, m, **k: rec["infos"].append(str(m))
    gui_module.messagebox.showwarning = lambda t, m, **k: rec["warnings"].append(str(m))
    gui_module.messagebox.askyesno = lambda *a, **k: answer_yes
    return rec


def cleanup(extra_runs=()):
    """Remove presets and runs created by a check. Never touches real runs."""
    import shutil
    from aiboy import presets
    from aiboy import runs
    for name in TEST_PRESETS:
        if presets.is_user(name):
            presets.delete(name)
    # A built-in override carrying the marker value can only be ours.
    if presets.load_user().get(presets.RECOMMENDED_PRESET, {}).get("ent_coef") == MARKER:
        presets.reset(presets.RECOMMENDED_PRESET)
    for prefix in ("e2etest", "zzcheck"):
        runs.delete_tune_data("mario", prefix)
    for run in extra_runs:
        if run and runs.is_run_name(run):
            base = runs.run_paths("mario", run)["base"]
            if base.exists():
                shutil.rmtree(base)
    for name in ("zzcheck-single", "zzcheck-exists", "zz_manifest_check"):
        base = runs.run_paths("mario", name)["base"]
        if base.exists():
            shutil.rmtree(base)
    if str(EXPERIENCE_FILE) == os.environ.get("AIBOY_EXPERIENCE_FILE"):
        EXPERIENCE_FILE.unlink(missing_ok=True)
