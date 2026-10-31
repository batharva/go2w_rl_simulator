"""Model management: checkpoint history, load / resume / evaluate / compare, stored training config."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from rl.model_manager import ModelManager


class ModelPanel(QWidget):
    load_requested = Signal(str)
    resume_requested = Signal(str)
    evaluate_requested = Signal(list)

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.mm = ModelManager(cfg)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["File", "Kind", "Timesteps", "Stage", "Eval score", "Saved"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 190)
        self.entries: List[Dict[str, Any]] = []
        lay = QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (("Refresh", self.refresh), ("Load into simulation", self._load), ("Resume training from…", self._resume),
                         ("Evaluate / compare selected", self._evaluate), ("Show training config", self._show_config)):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def selected_paths(self) -> List[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.entries[r]["path"] for r in rows]

    def refresh(self) -> None:
        self.entries = self.mm.list_models()
        self.table.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            vals = [e["name"], e["kind"], f"{e['timesteps']:,}" if e["timesteps"] else "-",
                    f"{e['stage']} (d={e['difficulty']:.2f})" if e["stage"] else "-",
                    f"{e['score']:.2f}" if isinstance(e["score"], (int, float)) else "-",
                    e["saved_at"] or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["mtime"]))]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def _one(self) -> str:
        sel = self.selected_paths()
        return sel[0] if sel else ""

    def _load(self) -> None:
        if p := self._one():
            self.load_requested.emit(p)

    def _resume(self) -> None:
        if p := self._one():
            self.resume_requested.emit(p)

    def _evaluate(self) -> None:
        if sel := self.selected_paths():
            self.evaluate_requested.emit(sel)

    def _show_config(self) -> None:
        p = self._one()
        if not p:
            return
        meta = self.mm.read_meta(p)
        d = QDialog(self); d.setWindowTitle(f"Training configuration - {Path(p).name}"); d.resize(640, 640)
        t = QPlainTextEdit(json.dumps(meta or {"info": "no side-car file found"}, indent=2, default=str))
        t.setReadOnly(True)
        QVBoxLayout(d).addWidget(t)
        d.exec()
