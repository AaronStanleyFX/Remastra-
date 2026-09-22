"""Point d'entrée de l'application graphique."""
from __future__ import annotations

import os
import sys


def _asset(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(base, "assets", name)
    if not os.path.exists(p):
        p = os.path.join(base, "remastra", "assets", name)
    return p


def main():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if os.name == "nt":
        try:  # icône propre dans la barre des tâches Windows
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RealAuthor.Remastra.1")
        except Exception:
            pass
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen

    app = QApplication(sys.argv)
    app.setApplicationName("REMASTRA")
    app.setOrganizationName("REMASTRA")
    app.setWindowIcon(QIcon(_asset("icon.png")))

    from .theme import QSS

    app.setStyleSheet(QSS)
    splash = None
    sp = _asset("splash.png")
    if os.path.exists(sp) and "--no-splash" not in sys.argv:
        splash = QSplashScreen(QPixmap(sp), Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

    from .main_window import MainWindow

    win = MainWindow()
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            QTimer.singleShot(400, lambda a=arg: win.open_file(a))
            break

    def show():
        win.show()
        if splash:
            splash.finish(win)

    QTimer.singleShot(900 if splash else 0, show)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
