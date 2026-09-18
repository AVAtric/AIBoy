"""AIboy learns: the same 2-candidate search run twice trains nothing the
second time, the wizard reports what it knows, the Experience tab lists the
trials, a related goal borrows that knowledge, "Let AIboy choose" plans
from the experience, and "Improve presets now" acts on it. ~40 s."""
import sys, time, tkinter as tk
from _common import say, silence_dialogs, cleanup, T0

from aiboy import experience, presets, tuning
from aiboy.gui import app as gui, wizard
rec = silence_dialogs(gui); cleanup()
SMOKE_BEFORE = presets.load_user().get(presets.SMOKE_TEST_PRESET)   # restored at the end

def restore_smoke_preset():
    if SMOKE_BEFORE is None:
        if presets.is_overridden(presets.SMOKE_TEST_PRESET): presets.reset(presets.SMOKE_TEST_PRESET)
    else:
        presets.upsert(presets.SMOKE_TEST_PRESET, SMOKE_BEFORE)
root = tk.Tk(); app = gui.AIboyGUI(root); w = app.wizard
state = {"s": "first"}
SWEEP = {"ent_coef": [0.02, 0.03]}      # both differ from the preset (0.01), or one would be known at once

def fail(msg):
    say("FAIL: " + msg); app._on_close(); restore_smoke_preset(); cleanup(); sys.exit(1)

def setup_search():
    w.goal_var.set(presets.SMOKE_TEST_PRESET); w._on_goal_changed()
    w.template_var.set(tuning.TEMPLATE_PLAIN[tuning.DEFAULT_TEMPLATE]); w._on_template_changed()
    w.trial_steps_var.set(4000); w.seeds_var.set(1); w._sync_tune_tab()
    app.set_sweep(SWEEP, "check sweep")
    w.auto_var.set(False)

def tick():
    if time.time() - T0 > 420: fail(f"timeout in {state['s']}")
    s = state["s"]
    if s == "first":
        if app.experience.records: fail("experience file not isolated: " + str(app.experience.path))
        setup_search()
        if "not tried this goal yet" not in w.known_note.cget("text"): fail("known note: " + w.known_note.cget("text"))
        plan, err = app.tune_plan()
        if err or app.known_trials(plan) != 0 or len(plan["combos"]) != 3: fail(f"plan {err} {plan and plan['combos']}")
        if "already known" in w.summary_var.get(): fail("summary before: " + w.summary_var.get())
        w.start_search(); state["s"] = "first-running"; state["t0"] = time.time()
        if w.phase != "tuning": fail("search did not start: " + w.status_var.get())
    elif s == "first-running" and w.phase == "idle":
        results = app.tune_results()
        if len(results) != 3 or any(r.reused for r in results) or not all(r.records for r in results):
            fail(f"first results: {[(r.index, len(r.records), r.reused, r.runs) for r in results]}; "
                 f"log tail: {app.log_text.get('end-25l', 'end')}")
        recs = app.experience.for_game("mario")
        if len(recs) != 3 or not all(r.complete and r.kind == "trial" and r.source == "wizard" for r in recs): fail(f"records: {[(r.kind, r.source, r.complete) for r in recs]}")
        cfg = presets.load_all()[presets.SMOKE_TEST_PRESET]
        fps = app.experience.fps(tuning.trial_config(cfg, {}, 4000))
        say(f"first search: {time.time() - state['t0']:.0f} s, {len(recs)} records, measured {fps and round(fps)} steps/s")
        if not fps: fail("no measured speed")
        if "Best known settings so far" not in w.known_note.cget("text"): fail("known note after: " + w.known_note.cget("text"))
        say("wizard knows: " + w.known_note.cget("text")[:100])
        plan, _ = app.tune_plan()
        if app.known_trials(plan) != 3: fail(f"known trials {app.known_trials(plan)}")
        if "all 3 short training runs already known" not in w.summary_var.get(): fail("summary: " + w.summary_var.get())
        say("plan: " + w.summary_var.get()[:110])
        if "measured speed" not in app.tune_summary_var.get(): fail("tune summary: " + app.tune_summary_var.get())
        tab = app.experience_tab; rows = tab.tree.get_children()
        if len(rows) != 3: fail(f"experience tab rows {len(rows)}")
        tab.tree.selection_set(rows[0]); tab._on_select()
        text = tab.insight.get("1.0", "end")
        if "Best known settings" not in text or "curiosity" not in text: fail("insight: " + text)
        say("insight: " + text.splitlines()[1][:90])
        # a related goal (marathon: same eyes, another level mode) borrows the campaign's knowledge
        marathon = next(n for n in app.presets_by_name() if "Marathon, all levels" in n and presets.is_builtin(n))
        w.goal_var.set(marathon); w._on_goal_changed(); w.trial_steps_var.set(4000); w._sync_tune_tab()
        if "but for the campaign" not in w.known_note.cget("text"): fail("borrowed note: " + w.known_note.cget("text"))
        say("borrowed: " + w.known_note.cget("text")[:100])
        w.template_var.set(wizard.AUTO_CHOICE); w._on_template_changed()
        know = app.experience.knowledge_for(w._trial_task())
        borrowed = tuning.config_diff(know.config, tuning.full_config(app.presets_by_name()[marathon], {}), experience.HYPER_FIELDS)
        if borrowed and "worked best for the campaign" not in w.template_note.cget("text"): fail("borrowed plan: " + w.template_note.cget("text"))
        if borrowed and app.tune_plan()[0]["combos"][1] != borrowed: fail("borrowed settings not tried first: " + str(app.tune_plan()[0]["combos"][:2]))
        w.template_var.set(tuning.TEMPLATE_PLAIN[tuning.DEFAULT_TEMPLATE]); w._on_template_changed()
        setup_search()
        # AIboy acts on what it knows: the improvement is applied only on a clear win,
        # recorded, and the preset then holds the winning settings
        if app.tune_results() and app.experience.improvements({presets.SMOKE_TEST_PRESET: presets.load_all()[presets.SMOKE_TEST_PRESET]}):
            improved = app.improve_presets()
            if not improved: fail("improvement found but not applied")
            imp = improved[0]
            cfg = presets.load_all()[presets.SMOKE_TEST_PRESET]
            if any(cfg[k] != v for k, v in imp.overrides.items()): fail("preset not updated: " + str(cfg))
            if not app.experience.improvement_history(presets.SMOKE_TEST_PRESET): fail("improvement not recorded")
            if app.experience.improvements({presets.SMOKE_TEST_PRESET: cfg}): fail("improvement repeats")
            if len(tab.tree.get_children()) != 4: fail("experience tab does not list the improvement")
            say("improved: " + imp.summary())
        else:
            if app.improve_presets(): fail("improved without a clear win")
            say("no clear win among the 3 tests: nothing improved (correct)")
        restore_smoke_preset(); app.refresh_presets(); app.experience.forget([r.id for r in app.experience.improvement_history(presets.SMOKE_TEST_PRESET)]); app.on_experience_changed()
        setup_search()
        # the same search again: everything comes from memory
        state["t0"] = time.time(); w.start_search(); state["s"] = "second-running"
        if w.phase != "tuning": fail("second search did not start")
    elif s == "second-running" and w.phase == "idle":
        elapsed = time.time() - state["t0"]
        results = app.tune_results()
        if sum(r.reused for r in results) != 3 or any(r.runs for r in results): fail("second search trained something: " + str([(r.reused, r.runs) for r in results]))
        if elapsed > 20: fail(f"second search took {elapsed:.0f} s")
        if len(app.experience.for_game("mario")) != 3: fail("records grew on reuse")
        say(f"second search: {elapsed:.1f} s, all 3 trials known; winner: {w.status_var.get()[:80]}")
        # "Let AIboy choose": a plan from the experience with an explanation
        w.template_var.set(wizard.AUTO_CHOICE); w._on_template_changed()
        plan, err = app.tune_plan()
        if err or len(plan["combos"]) != 7: fail(f"auto plan: {err} {plan and len(plan['combos'])}")
        if not w.template_note.cget("text") or not w.summary_var.get().startswith("Plan: try 6 variations"): fail("auto note/summary: " + w.summary_var.get())
        say("auto: " + w.template_note.cget("text")[:90])
        # forgetting works and the wizard notices
        app.experience.clear(); app.on_experience_changed()
        if app.known_trials(app.tune_plan()[0]) != 0 or "not tried" not in w.known_note.cget("text"): fail("forget")
        say("EXPERIENCE_OK"); app._on_close(); restore_smoke_preset(); cleanup(); return
    root.after(400, tick)

root.after(200, tick); root.mainloop()
