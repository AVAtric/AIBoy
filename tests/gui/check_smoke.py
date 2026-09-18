"""Build the GUI and exercise every tab without training: wizard navigation,
presets tab (edit / override / reset / create / rename / delete), Train tab
markers, screen panel, housekeeping. Dialogs are auto-answered."""
import tkinter as tk
from _common import say, silence_dialogs, cleanup

from aiboy import presets, runs, tuning
from aiboy.gui import app as gui
from _common import MARKER

rec = silence_dialogs(gui)
cleanup()
root = tk.Tk(); app = gui.AIboyGUI(root); root.update()
w = app.wizard
try:
    # ---- wizard: plan mirrors, skip search, preset step, names ----
    assert app.nb.index(app.nb.select()) == 0 and w.step == 0
    assert w.goal_var.get() == presets.RECOMMENDED_PRESET
    plan, err = app.tune_plan(); assert err is None and plan["prefix"] == "e2etest" and plan["keep_best"] == 9
    w.template_var.set(tuning.TEMPLATE_PLAIN["Learning rate (5 configs)"]); w._on_template_changed()
    plan, _ = app.tune_plan(); assert len(plan["combos"]) == 6 and plan["combos"][0] == {}   # + baseline
    assert plan["metric"] == tuning.METRIC_LATE and plan["n_seeds"] == 2                     # Normal effort
    assert w.trial_steps_var.get() == 100_000            # 5% of the 2M goal
    assert w.summary_var.get().startswith("Plan: try 5 variations plus the goal's own settings (12 short"), w.summary_var.get()
    w.effort_var.set("Quick"); w._on_effort_changed(); assert w.seeds_var.get() == 1 and w.trial_steps_var.get() == 50_000
    w.effort_var.set("Normal"); w._on_effort_changed()
    w.template_var.set(tuning.TEMPLATE_PLAIN["Broad random search (use Random, ~12 trials)"]); w._on_template_changed()
    plan, _ = app.tune_plan(); assert plan["search"] == "random" and len(plan["combos"]) == 13, len(plan["combos"])
    w.template_var.set(tuning.TEMPLATE_PLAIN["Learning rate (5 configs)"]); w._on_template_changed()
    w.search_var.set("no"); w._on_search_choice(); root.update()
    assert str(w.template_combo.cget("state")) == "disabled" and w.summary_var.get().startswith("Plan: train for")
    assert "goal's own settings" in w.summary_var.get()
    w.auto_var.set(False); w._update_summary(); assert "waits for you" in w.summary_var.get()
    assert "It plays through the game" in w.goal_note.cget("text"), w.goal_note.cget("text")
    w.start(); root.update()                              # search off + auto off -> Save step
    assert w.step == 1 and w.config["timesteps"] == 2_000_000
    assert w.preset_name_var.get() == "Campaign, recommended (wizard)", w.preset_name_var.get()
    w.overrides = {"ent_coef": 0.03}; w.config["ent_coef"] = 0.03; w.preset_name_var.set(""); w.goto(1)
    assert w.preset_name_var.get() == "Campaign, recommended · ent 0.03"
    assert w.config_text.tag_ranges("tuned")
    assert "Entropy coef" in w.config_text.get("1.0", "end")          # plain field labels, not keys
    w.on_game_changed(False); assert "mario.gb" in w.rom_hint_var.get(); w.on_game_changed(True); assert not w.rom_hint_var.get()
    w.preset_name_var.set("__check_preset"); w.save_and_continue(); root.update()
    assert w.step == 2 and presets.is_user("__check_preset") and app.preset_var.get() == "__check_preset"
    assert w.run_name_var.get() == "check-preset", w.run_name_var.get()
    assert app.form.vars["ent_coef"].get() == 0.03 and "≈" in w.train_eta_var.get()
    w.goto(3); root.update(); assert not app.playing_active()
    w.restart(); assert w.step == 0
    say("wizard navigation ok")

    # ---- presets tab ----
    pt = app.presets_tab
    rows = pt.tree.get_children(); assert len(rows) == len(presets.load_all())
    app.preset_var.set(presets.RECOMMENDED_PRESET); app.apply_preset()   # Train tab follows edits below
    pt.tree.selection_set(presets.RECOMMENDED_PRESET); pt._on_select(); root.update()
    assert pt.kind_var.get() == "(built-in)" and str(pt.form.widgets[0].cget("state")) == "normal"
    shipped = presets.BUILTIN_PRESETS[presets.RECOMMENDED_PRESET]["ent_coef"]
    pt.form.vars["ent_coef"].set(MARKER); root.update(); assert pt._dirty
    pt.save(); root.update(); assert presets.kind(presets.RECOMMENDED_PRESET) == "modified"
    assert app.form.vars["ent_coef"].get() == MARKER
    pt.reset_default(); root.update(); assert presets.kind(presets.RECOMMENDED_PRESET) == "built-in"
    assert app.form.vars["ent_coef"].get() == shipped
    pt._create("__check_copy", pt._template()); root.update()
    assert pt._current == "__check_copy" and pt.kind_var.get() == "(user)"
    pt.form.vars["ent_coef"].set(0.07); root.update(); pt.save(); root.update()
    assert presets.load_user()["__check_copy"]["ent_coef"] == 0.07
    presets.rename("__check_copy", "__check_copy2"); pt._current = "__check_copy2"; app.refresh_presets(); root.update()
    assert pt.tree.selection() == ("__check_copy2",)
    pt.load_into_train(); root.update()
    assert app.preset_var.get() == "__check_copy2" and app.preset_state_var.get() == ""
    app.form.vars["seed"].set(42); root.update(); assert app.preset_state_var.get().startswith("modified")
    pt.use_in_wizard(); root.update(); assert w.goal_var.get() == "__check_copy2"
    pt._create("__check_new", app.current_config()); assert presets.load_user()["__check_new"]["seed"] == 42
    say("presets tab ok")

    # ---- train tab hints, validation ----
    ex = runs.run_paths("mario", "zzcheck-exists")["logs"]; ex.mkdir(parents=True, exist_ok=True); (ex / "best_model.zip").write_bytes(b"x")
    app.run_name_var.set("zzcheck-exists"); app._on_run_name_changed(); assert "exists" in app.run_hint_var.get()
    app.run_name_var.set("brand-new-run"); app._on_run_name_changed(); assert "new run" in app.run_hint_var.get()
    app.run_name_var.set("")
    app.form.vars["n_envs"].set("abc")
    try:
        app.current_config(); raise SystemExit("expected ValueError")
    except ValueError as e:
        assert "Envs (" in str(e)
    app.form.vars["n_envs"].set(10)
    assert app.form.vars["obs_type"].get() == "tiles"

    # ---- screen panel ----
    app.refresh_models(); assert app.select_model(runs.best_model_for_run("mario", "zzcheck-exists"))
    say("model note: " + app.model_note_var.get()[:60]); assert "older run" in app.model_note_var.get()
    app.play_stat_vars["reward"].set("1"); app.clear_screen(); assert app.play_stat_vars["reward"].get() == "—"
    assert "power" in app.play_stat_vars

    # ---- housekeeping ----
    for name in ("zzcheck-001", "zzcheck-002-s1"):
        d = runs.run_paths("mario", name)["logs"]; d.mkdir(parents=True, exist_ok=True); (d / "best_model.zip").write_bytes(b"x")
    (runs.MODELS_ROOT / "mario" / "_tune").mkdir(exist_ok=True); (runs.MODELS_ROOT / "mario" / "_tune" / "zzcheck.json").write_text("{}")
    app.tune_run_prefix_var.set("zzcheck"); app.refresh_run_names()
    assert app.delete_tune_data("zzcheck") is True and not runs.tune_trial_runs("mario", "zzcheck")
    assert "Deleted 2 trial run" in app.status_bar_var.get()
    d = runs.run_paths("mario", "zzcheck-single")["checkpoints"]; d.mkdir(parents=True); (d / "final.zip").write_bytes(b"x")
    for i in range(3): (d / f"ppo_{(i+1)*1000}_steps.zip").write_bytes(b"x")
    app.run_name_var.set("zzcheck-single"); app._compact_run()
    assert sorted(p.name for p in d.glob("*.zip")) == ["final.zip"], list(d.glob("*.zip"))
    app._delete_run(); assert not d.parent.exists() and app.run_name_var.get() == ""
    app.tune_run_prefix_var.set("tune")
    w.search_var.set("yes"); w._on_search_choice()
    app.set_inputs_disabled(True); assert str(app.btn_tune_delete.cget("state")) == "disabled" and str(w.btn_clear_search.cget("state")) == "disabled"
    assert str(app.run_name_combo.cget("state")) == "disabled" and str(app.form.widgets[0].cget("state")) == "disabled"
    app.set_inputs_disabled(False)
    # inputs come back in their original state: the run-name box stays typeable
    assert str(app.run_name_combo.cget("state")) == "normal", app.run_name_combo.cget("state")
    assert str(app.preset_combo.cget("state")) == "readonly" and str(w.goal_combo.cget("state")) == "readonly"
    assert str(w.template_combo.cget("state")) == "readonly"          # search widgets follow the yes/no choice
    assert str(app.form.vars and app.form.widgets[2].cget("state")) == "readonly"     # obs_type combobox
    pt.tree.selection_set(presets.RECOMMENDED_PRESET); pt._on_select(); pt._show(pt._current); pt._show(pt._current)
    assert str(pt.form.widgets[2].cget("state")) == "readonly"                        # repeated enable keeps it read-only
    # metric switch re-ranks
    app.tune_metric_var.set(tuning.METRIC_PER_MINUTE); app._on_metric_changed()
    assert app.tune_tree.heading("score")["text"] == tuning.METRIC_PER_MINUTE
    app.tune_metric_var.set(tuning.METRIC_BEST); app._on_metric_changed()
    assert not rec["errors"], rec["errors"]
    say("housekeeping ok")
finally:
    app._on_close()
    cleanup()
print("SMOKE_OK")
