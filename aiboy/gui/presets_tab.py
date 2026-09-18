"""Presets tab: browse, create, edit and organise training presets.

Top: every preset in a table. Bottom: the selected preset in the same
Basic / Advanced form the Train tab uses; every preset is editable.

Three kinds of preset:
  built-in   shipped in builtin_presets.json; editing one stores a user
             override under the same name (the shipped values are never
             touched) and "Reset to default" drops the override again
  modified   a built-in with such an override
  user       created here, on the Train tab, or by the wizard; stored in
             training_presets.json; can be renamed and deleted
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from aiboy import presets
from aiboy.gui.widgets import MONO_BOLD, THEME, ConfigForm, info_icon, make_table, tooltip
from aiboy.tuning import mode_label

KIND_NOTE = {
    "built-in": "Built-in preset. You can edit it; Save stores your version and "
                "'Reset to default' brings the shipped values back.",
    "modified": "Built-in preset with your changes. 'Reset to default' restores the shipped values.",
    "user": "User preset. Edit, rename or delete it freely.",
}
ABOUT = ("A preset is a complete training configuration. Select one to edit it below; "
         "double-click to load it into the Train tab. Built-in presets keep their shipped "
         "values behind your edits, so they can always be reset.")


class PresetsTab:
    def __init__(self, app, parent: ttk.Frame):
        self.app = app
        self._all: dict[str, dict] = {}
        self._current: str | None = None
        self._dirty = False
        self._loading = False
        self._build(parent)

    # ---------- layout ----------

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        table_frame = ttk.Frame(top)
        table_frame.grid(row=0, column=0, sticky="nsew")
        self.tree = make_table(table_frame, [
            ("name", "preset", 280, "w", True), ("kind", "type", 76, "w", False),
            ("mode", "mode", 104, "w", False), ("steps", "steps", 90, "e", False),
            ("envs", "envs", 44, "e", False), ("obs", "obs", 56, "w", False),
        ], height=7)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.tree.bind("<Double-1>", lambda e: self.load_into_train())

        lbtns = ttk.Frame(top)
        lbtns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.btn_new = ttk.Button(lbtns, text="New…", command=self.new)
        self.btn_new.pack(side="left")
        self.btn_from_train = ttk.Button(lbtns, text="New from Train tab…",
                                         command=self.new_from_train)
        self.btn_from_train.pack(side="left", padx=4)
        self.btn_dup = ttk.Button(lbtns, text="Duplicate…", command=self.duplicate)
        self.btn_dup.pack(side="left", padx=4)
        ttk.Button(lbtns, text="Reload from disk", command=self.app.refresh_presets).pack(
            side="right")
        info_icon(lbtns, ABOUT).pack(side="right", padx=(0, 8))
        tooltip(self.btn_new, "A new preset starting from the selected one's values.")
        tooltip(self.btn_from_train, "A new preset with the values currently on the Train tab.")
        lbtns2 = ttk.Frame(top)
        lbtns2.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.btn_rename = ttk.Button(lbtns2, text="Rename…", command=self.rename)
        self.btn_rename.pack(side="left")
        self.btn_delete = ttk.Button(lbtns2, text="Delete", command=self.delete)
        self.btn_delete.pack(side="left", padx=4)
        self.btn_reset = ttk.Button(lbtns2, text="Reset to default", command=self.reset_default)
        self.btn_reset.pack(side="left", padx=4)

        editor = ttk.Frame(parent)
        editor.grid(row=2, column=0, sticky="ew")
        editor.columnconfigure(0, weight=1)
        head = ttk.Frame(editor)
        head.grid(row=0, column=0, sticky="ew")
        self.title_var = tk.StringVar(value="No preset selected")
        ttk.Label(head, textvariable=self.title_var, font=MONO_BOLD).pack(side="left")
        self.kind_var = tk.StringVar(value="")
        kind_lbl = ttk.Label(head, textvariable=self.kind_var, foreground=THEME.muted)
        kind_lbl.pack(side="left", padx=8)
        tooltip(kind_lbl, lambda: KIND_NOTE.get(self.kind_var.get().strip("()"), ""))
        self.note_var = tk.StringVar(value="")
        ttk.Label(editor, textvariable=self.note_var, foreground=THEME.muted, wraplength=640).grid(
            row=1, column=0, sticky="w", pady=(0, 6))
        self.form = ConfigForm(editor, on_change=self._on_edit)
        self.form.grid(row=2, column=0, sticky="ew")

        rbtns = ttk.Frame(editor)
        rbtns.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.btn_save = ttk.Button(rbtns, text="Save changes", command=self.save)
        self.btn_save.pack(side="left")
        self.btn_revert = ttk.Button(rbtns, text="Revert", command=self._reload_form)
        self.btn_revert.pack(side="left", padx=4)
        self.btn_to_train = ttk.Button(rbtns, text="Load into Train tab",
                                       command=self.load_into_train)
        self.btn_to_train.pack(side="right")
        self.btn_to_wizard = ttk.Button(rbtns, text="Use in Wizard", command=self.use_in_wizard)
        self.btn_to_wizard.pack(side="right", padx=4)
        self._show(None)

    # ---------- data ----------

    def refresh(self) -> None:
        """Re-read presets and redraw the table, keeping the selection."""
        self._all = self.app.presets_by_name()
        names = presets.sorted_names(self._all, self.app.game)
        user = presets.load_user()
        for row in self.tree.get_children():
            self.tree.delete(row)
        for name in names:
            cfg = self._all[name]
            kind = presets.kind(name, user)
            self.tree.insert("", "end", iid=name, values=(
                name, kind, mode_label(cfg), f"{int(cfg.get('timesteps', 0)):,}",
                cfg.get("n_envs", "?"), cfg.get("obs_type", "tiles"),
            ), tags=(kind,) if kind != "built-in" else ())
        if self._current in names:
            # selection_set fires <<TreeviewSelect>>, but _on_select ignores a
            # re-selection of the current name, so load the form explicitly
            # (unless the user is mid-edit).
            self.tree.selection_set(self._current)
            self.tree.see(self._current)
            if not self._dirty:
                self._show(self._current)
            else:
                self._update_buttons()
        else:
            self._current = None
            self._show(None)

    def _on_select(self) -> None:
        sel = self.tree.selection()
        name = sel[0] if sel else None
        if name == self._current:
            return
        if self._dirty and not messagebox.askyesno(
                "Presets", f"Discard unsaved changes to '{self._current}'?"):
            if self._current is not None:
                self.tree.selection_set(self._current)
            return
        self._current = name
        self._show(name)

    def _show(self, name: str | None) -> None:
        cfg = self._all.get(name) if name else None
        self._loading = True
        if cfg is None:
            self.title_var.set("No preset selected")
            self.kind_var.set("")
            self.note_var.set("Select a preset above, or click New… to create one.")
            self.form.set_enabled(False)
        else:
            kind = presets.kind(name)
            self.title_var.set(name)
            self.kind_var.set(f"({kind})")
            history = self.app.experience.improvement_history(name)
            note = ""
            if history:
                last = history[-1]
                note = (f"AIboy improved it on "
                        f"{time.strftime('%Y-%m-%d', time.localtime(last.created_at))}: {last.note}")
            self.note_var.set(note)
            self.form.set_config(cfg)
            self.form.set_enabled(True)
        self._loading = False
        self._dirty = False
        self._update_buttons()

    def _reload_form(self) -> None:
        self._show(self._current)

    def _on_edit(self) -> None:
        if self._loading or self._current is None:
            return
        if not self._dirty:
            self._dirty = True
            self._update_buttons()

    def _update_buttons(self) -> None:
        name = self._current
        has = name is not None and name in self._all
        kind = presets.kind(name) if has else None
        busy = self.app.busy()
        self.btn_save.config(state="normal" if has and self._dirty else "disabled")
        self.btn_revert.config(state="normal" if has and self._dirty else "disabled")
        self.btn_rename.config(state="normal" if kind == "user" else "disabled")
        self.btn_delete.config(state="normal" if kind == "user" else "disabled")
        self.btn_reset.config(state="normal" if kind == "modified" else "disabled")
        self.btn_dup.config(state="normal" if has else "disabled")
        self.btn_from_train.config(state="disabled" if busy else "normal")
        self.btn_to_train.config(state="normal" if has and not busy else "disabled")
        self.btn_to_wizard.config(state="normal" if has and not busy else "disabled")

    def set_inputs_disabled(self, disabled: bool) -> None:
        # Editing presets while a run is active is harmless; only the actions
        # that would change the running tabs are locked.
        self._update_buttons()

    # ---------- actions ----------

    def _ask_name(self, title: str, prompt: str, initial: str = "") -> str | None:
        name = simpledialog.askstring(title, prompt, parent=self.app.root, initialvalue=initial)
        name = (name or "").strip()
        return name or None

    def _create(self, name: str, cfg: dict) -> bool:
        if name in self._all and not messagebox.askyesno(
                "Presets", f"A preset named '{name}' already exists. Overwrite it?"):
            return False
        try:
            presets.upsert(name, cfg)
        except (ValueError, OSError) as e:
            messagebox.showerror("Presets", str(e))
            return False
        self._current = name
        self._dirty = False
        self.app.refresh_presets()
        self.app.flash(f"Preset '{name}' saved.")
        return True

    def _template(self) -> dict:
        """Starting values for a new preset: the selected one, else the recommended."""
        if self._current in self._all:
            return dict(self._all[self._current])
        return dict(self._all.get(presets.RECOMMENDED_PRESET)
                    or next(iter(self._all.values()), {"game": self.app.game}))

    def new(self) -> None:
        name = self._ask_name("New preset", "Name for the new preset:")
        if name:
            cfg = self._template()
            cfg["game"] = self.app.game
            self._create(name, cfg)

    def new_from_train(self) -> None:
        try:
            cfg = self.app.current_config()
        except ValueError as e:
            messagebox.showerror("Presets", f"The Train tab has an invalid value:\n{e}")
            return
        name = self._ask_name("New preset", "Name for the Train tab's current settings:")
        if name:
            self._create(name, cfg)

    def duplicate(self) -> None:
        if self._current is None or self._current not in self._all:
            return
        name = self._ask_name("Duplicate preset", "Name for the copy:",
                              initial=f"{self._current} (copy)")
        if name:
            self._create(name, dict(self._all[self._current]))

    def rename(self) -> None:
        old = self._current
        if old is None or presets.kind(old) != "user":
            return
        new = self._ask_name("Rename preset", "New name:", initial=old)
        if not new or new == old:
            return
        try:
            presets.rename(old, new)
        except (ValueError, OSError) as e:
            messagebox.showerror("Presets", str(e))
            return
        if self.app.preset_var.get() == old:
            self.app.preset_var.set(new)
        self._current = new
        self.app.refresh_presets()

    def delete(self) -> None:
        name = self._current
        if name is None or presets.kind(name) != "user":
            return
        if not messagebox.askyesno("Delete preset", f"Delete preset '{name}'?"):
            return
        try:
            presets.delete(name)
        except (ValueError, OSError) as e:
            messagebox.showerror("Presets", str(e))
            return
        if self.app.preset_var.get() == name:
            self.app.preset_var.set("")
        self._current = None
        self._dirty = False
        self.app.refresh_presets()
        self.app.flash(f"Preset '{name}' deleted.")

    def reset_default(self) -> None:
        name = self._current
        if name is None or presets.kind(name) != "modified":
            return
        try:
            presets.reset(name)
        except (ValueError, OSError) as e:
            messagebox.showerror("Presets", str(e))
            return
        self._dirty = False
        self.app.refresh_presets()
        if self.app.preset_var.get() == name:
            self.app.apply_preset()
        self.app.flash(f"'{name}' reset to its shipped values.")

    def save(self) -> None:
        name = self._current
        if name is None:
            return
        try:
            cfg = {"game": self.app.game, **self.form.get_config()}
            presets.upsert(name, cfg)
        except (ValueError, OSError) as e:
            messagebox.showerror("Presets", str(e))
            return
        self._dirty = False
        self.app.refresh_presets()
        if self.app.preset_var.get() == name:
            self.app.apply_preset()
        self.app.flash(f"Preset '{name}' saved.")

    def load_into_train(self) -> None:
        name = self._current
        cfg = self._all.get(name) if name else None
        if cfg is None or self.app.busy():
            return
        self.app.load_config_into_train(cfg)
        self.app.preset_var.set(name)
        self.app.show_tab(self.app.train_tab)
        self.app.flash(f"'{name}' loaded into the Train tab.")

    def use_in_wizard(self) -> None:
        name = self._current
        if name is None or name not in self._all:
            return
        self.app.wizard.set_goal(name)
        self.app.show_tab(self.app.wizard_tab)
