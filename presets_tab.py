"""Presets tab: browse, edit and organise training presets.

Left: every preset (built-in and user) in a table. Right: the selected
preset in the same Basic / Advanced form the Train tab uses. Built-ins are
read-only; duplicate one to get an editable user copy. User presets live in
`training_presets.json`, built-ins in `builtin_presets.json`.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import presets
from widgets import MONO_BOLD, MUTED, ConfigForm, make_table

MODE_LABEL = {"default": "campaign", "random": "random", "sequential": "sequential",
              "marathon": "marathon"}


def mode_label(cfg: dict) -> str:
    level = str(cfg.get("start_level", "default"))
    return MODE_LABEL.get(level, f"level {level}")


class PresetsTab:
    def __init__(self, app, parent: ttk.Frame):
        self.app = app
        self._all: dict[str, dict] = {}
        self._current: str | None = None
        self._build(parent)

    # ---------- layout ----------

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(parent, wraplength=980, foreground="#444",
                  text="Presets are complete training configurations. Built-in presets are "
                       "read-only; duplicate one to make an editable copy. User presets are "
                       "stored in training_presets.json.").grid(
            row=0, column=0, sticky="w", pady=(0, 8))

        # ---- top: table ----
        top = ttk.Frame(parent)
        top.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        table_frame = ttk.Frame(top)
        table_frame.grid(row=0, column=0, sticky="nsew")
        self.tree = make_table(table_frame, [
            ("name", "preset", 420, "w", True), ("kind", "type", 70, "w", False),
            ("mode", "mode", 100, "w", False), ("steps", "steps", 100, "e", False),
            ("envs", "envs", 50, "e", False), ("obs", "obs", 60, "w", False),
        ], height=7)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())

        lbtns = ttk.Frame(top)
        lbtns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.btn_new = ttk.Button(lbtns, text="New from Train tab…", command=self.new_from_train)
        self.btn_new.pack(side="left")
        self.btn_dup = ttk.Button(lbtns, text="Duplicate…", command=self.duplicate)
        self.btn_dup.pack(side="left", padx=4)
        self.btn_rename = ttk.Button(lbtns, text="Rename…", command=self.rename)
        self.btn_rename.pack(side="left", padx=4)
        self.btn_delete = ttk.Button(lbtns, text="Delete", command=self.delete)
        self.btn_delete.pack(side="left", padx=4)
        ttk.Button(lbtns, text="Reload", command=self.app.refresh_presets).pack(side="right")

        # ---- bottom: editor ----
        right = ttk.Frame(parent)
        right.grid(row=2, column=0, sticky="ew")
        right.columnconfigure(0, weight=1)
        self.title_var = tk.StringVar(value="No preset selected")
        ttk.Label(right, textvariable=self.title_var, font=MONO_BOLD).grid(
            row=0, column=0, sticky="w")
        self.note_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.note_var, foreground=MUTED, wraplength=960).grid(
            row=1, column=0, sticky="w", pady=(0, 6))
        self.form = ConfigForm(right, on_change=self._on_edit)
        self.form.grid(row=2, column=0, sticky="ew")
        self._dirty = False
        self._loading = False

        rbtns = ttk.Frame(right)
        rbtns.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.btn_save = ttk.Button(rbtns, text="Save changes", command=self.save)
        self.btn_save.pack(side="left")
        self.btn_revert = ttk.Button(rbtns, text="Revert", command=self._reload_form)
        self.btn_revert.pack(side="left", padx=4)
        self.btn_to_train = ttk.Button(rbtns, text="Load into Train tab",
                                       command=self.load_into_train)
        self.btn_to_train.pack(side="right")
        self.btn_to_wizard = ttk.Button(rbtns, text="Use in Wizard",
                                        command=self.use_in_wizard)
        self.btn_to_wizard.pack(side="right", padx=4)
        self._update_buttons()

    # ---------- data ----------

    def refresh(self) -> None:
        """Re-read presets and redraw the table, keeping the selection."""
        self._all = self.app.presets_by_name()
        names = presets.sorted_names(self._all, self.app.game)
        for row in self.tree.get_children():
            self.tree.delete(row)
        for name in names:
            cfg = self._all[name]
            self.tree.insert("", "end", iid=name, values=(
                name, "built-in" if presets.is_builtin(name) else "user",
                mode_label(cfg), f"{int(cfg.get('timesteps', 0)):,}",
                cfg.get("n_envs", "?"), cfg.get("obs_type", "tiles"),
            ), tags=() if presets.is_builtin(name) else ("best",))
        if self._current in names:
            # selection_set fires <<TreeviewSelect>>, but _on_select ignores a
            # re-selection of the current name, so load the form explicitly
            # (unless the user is mid-edit).
            self.tree.selection_set(self._current)
            self.tree.see(self._current)
            if not self._dirty:
                self._show(self._current)
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
            self.note_var.set("Select a preset on the left.")
            self.form.set_enabled(False)
        else:
            builtin = presets.is_builtin(name)
            self.title_var.set(name)
            self.note_var.set("Built-in preset (read-only). Duplicate it to edit."
                              if builtin else "User preset. Edit the fields and click Save changes.")
            self.form.set_config(cfg)
            self.form.set_enabled(not builtin)
        self._loading = False
        self._dirty = False
        self._update_buttons()

    def _reload_form(self) -> None:
        self._show(self._current)

    def _on_edit(self) -> None:
        if self._loading or self._current is None:
            return
        self._dirty = True
        self._update_buttons()

    def _update_buttons(self) -> None:
        name = self._current
        has = name is not None and name in self._all
        user = has and not presets.is_builtin(name)
        busy = self.app.busy()
        self.btn_save.config(state="normal" if user and self._dirty else "disabled")
        self.btn_revert.config(state="normal" if user and self._dirty else "disabled")
        self.btn_rename.config(state="normal" if user else "disabled")
        self.btn_delete.config(state="normal" if user else "disabled")
        self.btn_dup.config(state="normal" if has else "disabled")
        self.btn_new.config(state="disabled" if busy else "normal")
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

    def _create(self, name: str, cfg: dict) -> None:
        if presets.is_user(name) and not messagebox.askyesno(
                "Presets", f"A preset named '{name}' already exists. Overwrite it?"):
            return
        try:
            presets.upsert(name, cfg)
        except (ValueError, OSError) as e:
            messagebox.showerror("Presets", str(e))
            return
        self._current = name
        self.app.refresh_presets()

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
        if self._current is None:
            return
        cfg = self._all.get(self._current)
        if cfg is None:
            return
        name = self._ask_name("Duplicate preset", "Name for the copy:",
                              initial=f"{self._current} (copy)")
        if name:
            self._create(name, dict(cfg))

    def rename(self) -> None:
        old = self._current
        if old is None or presets.is_builtin(old):
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
        if name is None or presets.is_builtin(name):
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

    def save(self) -> None:
        name = self._current
        if name is None or presets.is_builtin(name):
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

    def load_into_train(self) -> None:
        name = self._current
        cfg = self._all.get(name) if name else None
        if cfg is None:
            return
        self.app.load_config_into_train(cfg)
        self.app.preset_var.set(name)
        self.app.show_tab(self.app.train_tab)

    def use_in_wizard(self) -> None:
        name = self._current
        if name is None or name not in self._all:
            return
        self.app.wizard.set_goal(name)
        self.app.show_tab(self.app.wizard_tab)
