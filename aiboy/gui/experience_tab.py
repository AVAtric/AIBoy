"""Experience tab: everything AIboy has tried, concluded and changed.

Top: every remembered trial, run and preset improvement of the selected
game, newest first. Bottom: what the records say about the task of the
selected row (best known settings, and how each knob's values compared),
plus the actions: load the settings into the Train tab, keep them as a
preset, forget records, and the switch that lets AIboy change presets by
itself.

The tab only shows the experience file; the learning itself happens in
experience.py and is used by the wizard, the Tune tab and the app's preset
improvements.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from aiboy import experience, runs, settings, tuning
from aiboy.gui.widgets import MONO, MONO_BOLD, THEME, info_icon, make_table, tooltip
from aiboy.tuning import mode_label

KIND_LABEL = {experience.KIND_TRIAL: "test", experience.KIND_RUN: "training",
              experience.KIND_IMPROVEMENT: "improved"}
ABOUT = ("AIboy remembers every short test and every training it finishes: the settings, how "
         "the score developed and how long it took. Searches skip what is already known, the "
         "wizard explores around the best known settings (borrowing from a related goal when "
         "a goal is new), time estimates use the speed this computer really reached, and "
         "presets are updated when tests show clearly better settings for their goal.")


def when_label(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def row_values(r: experience.Record) -> tuple:
    if r.kind == experience.KIND_IMPROVEMENT:
        return (when_label(r.created_at), KIND_LABEL[r.kind], r.source, mode_label(r.config),
                tuning.compact_config(r.config), "—", "—", "—", "—", r.run_name)
    m = r.metrics()
    late = f"{m.late:.0f}" if m is not None else "—"
    steps = tuning.format_steps(r.steps_done)
    if not r.complete:
        target = int(r.config.get("timesteps", 0)) or 1
        steps += f" ({100 * r.steps_done / target:.0f}%)"
    mode = mode_label(r.config)
    if r.config.get("obs_type") == "pixels":
        mode += " (pixels)"
    return (when_label(r.created_at), KIND_LABEL.get(r.kind, r.kind), r.source, mode,
            tuning.compact_config(r.config), steps, late, f"{r.fps:.0f}" if r.fps else "—",
            tuning.format_duration(r.duration) if r.duration else "—", r.run_name)


def insight_text(exp: experience.Experience, record: experience.Record | None,
                 metric: str = tuning.METRIC_LATE) -> str:
    """Plain summary of what the records say about one task."""
    if record is None:
        return "Select a row to see what AIboy concluded about that goal."
    if record.kind == experience.KIND_IMPROVEMENT:
        return (f"Preset '{record.run_name}' changed by AIboy on "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(record.created_at))}: "
                f"{record.note}\nA built-in preset can be reset to its shipped values on the "
                f"Presets tab.")
    cfg = record.config
    task_name = (f"{mode_label(cfg)} · {cfg.get('obs_type')} · "
                 f"{tuning.format_steps(int(cfg.get('timesteps', 0)))}-step tests")
    if record.env_version != experience.ENV_VERSION:
        return (f"{task_name}: recorded with an older version of the game logic "
                f"({record.env_version}); kept for reference, never reused.")
    know = exp.best_for_task(cfg, metric)
    if know is None:
        return f"{task_name}: no finished short tests yet, so nothing to compare."
    lines = [f"{task_name}: {know.n_trials} tests remembered.",
             f"Best known settings: {tuning.compact_config(know.config)} "
             f"→ score {know.score:.0f} (average of {know.n_seeds} seed(s))."]
    effects = exp.effects(cfg, metric)
    for key, rows in effects.items():
        parts = [f"{tuning.format_value(v)} → {s:.0f}" + (f" (×{n})" if n > 1 else "")
                 for v, s, n in rows[:5]]
        lines.append(f"{tuning.PLAIN_KEYS.get(key, key)} ({tuning.SHORT_KEYS.get(key, key)}): "
                     + ", ".join(parts))
    if not effects:
        lines.append("Every test so far used the same values, so no knob can be compared yet.")
    return "\n".join(lines)


class ExperienceTab:
    def __init__(self, app, parent: ttk.Frame):
        self.app = app
        self._rows: dict[str, experience.Record] = {}
        self._build(parent)
        self.refresh()

    # ---------- layout ----------

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        head = ttk.Frame(parent)
        head.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.summary_var = tk.StringVar(value="")
        ttk.Label(head, textvariable=self.summary_var, font=MONO_BOLD, wraplength=600).pack(
            side="left")
        info_icon(head, ABOUT).pack(side="left", padx=(8, 0))

        table_frame = ttk.Frame(parent)
        table_frame.grid(row=2, column=0, sticky="nsew")
        self.tree = make_table(table_frame, [
            ("when", "when", 82, "w", False), ("kind", "what", 54, "w", False),
            ("source", "by", 50, "w", False), ("mode", "mode", 78, "w", False),
            ("settings", "settings", 180, "w", True), ("steps", "steps", 60, "e", False),
            ("late", "score", 54, "e", False), ("fps", "fps", 44, "e", False),
            ("time", "time", 48, "e", False), ("run", "run", 76, "w", True),
        ], height=9)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.tree.bind("<Double-1>", lambda e: self.load_into_train())

        box = ttk.LabelFrame(parent, text="What AIboy concluded", padding=6)
        box.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        self.insight = tk.Text(box, height=6, font=MONO, wrap="word", relief="flat",
                               background=THEME.panel[0], foreground=THEME.panel[1],
                               highlightthickness=0)
        self.insight.grid(row=0, column=0, sticky="ew")
        self.insight.config(state="disabled")

        btns = ttk.Frame(parent)
        btns.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.btn_to_train = ttk.Button(btns, text="Load into Train tab", command=self.load_into_train)
        self.btn_to_train.pack(side="left")
        self.btn_save = ttk.Button(btns, text="Save as preset…", command=self.save_as_preset)
        self.btn_save.pack(side="left", padx=4)
        self.btn_forget_all = ttk.Button(btns, text="Forget all…", command=self.forget_all)
        self.btn_forget_all.pack(side="right")
        self.btn_forget = ttk.Button(btns, text="Forget selected…", command=self.forget_selected)
        self.btn_forget.pack(side="right", padx=4)
        ttk.Button(btns, text="Reload", command=self._reload).pack(side="right", padx=4)

        improve = ttk.Frame(parent)
        improve.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.auto_improve_var = tk.BooleanVar(value=bool(settings.get("auto_improve_presets")))
        self.auto_improve_check = ttk.Checkbutton(
            improve, variable=self.auto_improve_var, command=self._on_auto_improve_changed,
            text="Improve presets automatically")
        self.auto_improve_check.pack(side="left")
        tooltip(self.auto_improve_check,
                "After every finished test or training, change a preset when the tests show "
                "settings that clearly beat its own. Built-in presets can always be reset to "
                "their shipped values on the Presets tab.")
        self.btn_improve = ttk.Button(improve, text="Improve presets now",
                                      command=lambda: self.app.improve_presets())
        self.btn_improve.pack(side="left", padx=(12, 0))
        tooltip(self.btn_improve, "Check every preset of this game against what AIboy has "
                                  "measured, and change the ones it can clearly improve.")
        tooltip(self.btn_forget, "Remove the selected record. AIboy may then train these "
                                 "settings again in a later search.")

    # ---------- data ----------

    def _reload(self) -> None:
        self.app.experience.reload()
        self.refresh()

    def refresh(self) -> None:
        exp = self.app.experience
        game = self.app.game
        selected = self.tree.selection()
        keep = selected[0] if selected else None
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._rows = {}
        for r in reversed(exp.for_game(game)):
            self._rows[r.id] = r
            tags = ()
            if r.kind == experience.KIND_IMPROVEMENT:
                tags = ("modified",)
            elif not r.complete:
                tags = ("muted",)
            elif r.kind == experience.KIND_RUN:
                tags = ("user",)
            self.tree.insert("", "end", iid=r.id, values=row_values(r), tags=tags)
        s = exp.summary(game)
        parts = [f"{s['trials']} short test(s)", f"{s['runs']} training run(s)"]
        if s["improvements"]:
            parts.append(f"{s['improvements']} preset improvement(s)")
        if s["incomplete"]:
            parts.append(f"{s['incomplete']} unfinished")
        self.summary_var.set(" · ".join(parts) + f" · {s['compute_hours']:.1f} h of compute "
                             f"remembered · {runs.format_size(exp.size_bytes())} in "
                             f"{exp.path.name}")
        if keep in self._rows:
            self.tree.selection_set(keep)
        self._on_select()

    def _selected(self) -> experience.Record | None:
        sel = self.tree.selection()
        return self._rows.get(sel[0]) if sel else None

    def _on_select(self) -> None:
        record = self._selected()
        self.insight.config(state="normal")
        self.insight.delete("1.0", "end")
        self.insight.insert("1.0", insight_text(self.app.experience, record))
        self.insight.config(state="disabled")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has = self._selected() is not None
        busy = self.app.busy()
        self.btn_to_train.config(state="normal" if has and not busy else "disabled")
        self.btn_save.config(state="normal" if has else "disabled")
        self.btn_forget.config(state="normal" if has and not busy else "disabled")
        self.btn_forget_all.config(state="normal" if self._rows and not busy else "disabled")
        self.btn_improve.config(state="disabled" if busy else "normal")

    def set_inputs_disabled(self, disabled: bool) -> None:
        self._update_buttons()

    def _on_auto_improve_changed(self) -> None:
        settings.put("auto_improve_presets", bool(self.auto_improve_var.get()))

    # ---------- actions ----------

    def load_into_train(self) -> None:
        record = self._selected()
        if record is None or self.app.busy():
            return
        self.app.load_config_into_train(record.config)
        self.app.preset_var.set("")
        self.app.show_tab(self.app.train_tab)
        self.app.flash(f"Settings of {KIND_LABEL.get(record.kind, record.kind)} "
                       f"'{record.run_name or record.id}' loaded — set a run name and Start.")

    def save_as_preset(self) -> None:
        record = self._selected()
        if record is None:
            return
        name = simpledialog.askstring("Save as preset", "Name for these settings:",
                                      parent=self.app.root,
                                      initialvalue=f"{mode_label(record.config)} · "
                                                   f"{tuning.compact_config(record.config)}"[:90])
        name = (name or "").strip()
        if not name:
            return
        if self.app.save_preset_named(name, record.config):
            self.app.flash(f"Preset '{name}' saved.")

    def forget_selected(self) -> None:
        record = self._selected()
        if record is None or self.app.busy():
            return
        if not messagebox.askyesno("Forget", f"Forget this {KIND_LABEL.get(record.kind, record.kind)} "
                                             f"({record.run_name or record.id})?\n\nAIboy may then "
                                             f"train these settings again in a later search."):
            return
        self.app.experience.forget([record.id])
        self.app.on_experience_changed()

    def forget_all(self) -> None:
        if self.app.busy() or not self._rows:
            return
        n = len(self.app.experience.records)
        if not messagebox.askyesno("Forget everything",
                                   f"Forget all {n} remembered tests and runs (every game)?\n\n"
                                   f"Presets, models and run folders are not touched. This "
                                   f"cannot be undone."):
            return
        self.app.experience.clear()
        self.app.on_experience_changed()
        self.app.flash("AIboy's experience was cleared.")
