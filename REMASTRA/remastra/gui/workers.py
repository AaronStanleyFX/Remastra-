"""Exécution des traitements en tâche de fond (l'interface reste fluide)."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    progress = Signal(float)
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            res = self.fn(*self.args, progress=self.progress.emit, log=self.log.emit, **self.kwargs)
            self.done.emit(res)
        except BaseException as e:  # noqa: BLE001  (SystemExit de libs tierces)
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
