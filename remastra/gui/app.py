"""Point d'entrée de l'application graphique.

Au lancement : fenêtre de choix de la langue (English / Français), puis écran
de démarrage et interface. Options de ligne de commande :
  --lang en|fr   impose la langue (le lanceur l'utilise après l'installation)
  --no-splash    sans écran de démarrage
"""
from __future__ import annotations

import locale
import os
import sys


def _asset(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(base, "assets", name)
    if not os.path.exists(p):
        p = os.path.join(base, "remastra", "assets", name)
    return p


def _arg_lang(argv):
    for i, a in enumerate(argv):
        if a == "--lang" and i + 1 < len(argv):
            return argv[i + 1].lower()[:2]
        if a.startswith("--lang="):
            return a.split("=", 1)[1].lower()[:2]
    return None


def _system_lang():
    try:
        loc = (locale.getlocale()[0] or os.environ.get("LANG", "")).lower()
    except Exception:
        loc = ""
    return "fr" if loc.startswith(("fr", "french")) else "en"


def main():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if os.name == "nt":
        try:  # icône propre dans la barre des tâches Windows
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RealAuthor.Remastra.1")
        except Exception:
            pass
    from PySide6.QtCore import QSettings, Qt, QTimer
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen

    from .. import i18n

    app = QApplication(sys.argv)
    app.setApplicationName("REMASTRA")
    app.setOrganizationName("REMASTRA")
    app.setWindowIcon(QIcon(_asset("icon.png")))

    from .theme import QSS

    app.setStyleSheet(QSS)

    # ---------------------------------------------------------- langue ------
    settings = QSettings("REMASTRA", "REMASTRA")
    saved = settings.value("lang", "", str) or _system_lang()
    lang = _arg_lang(sys.argv)
    if lang not in i18n.LANGS:
        if settings.value("lang_remember", False, bool):
            lang = saved
        else:
            from .lang_dialog import ask_language

            lang, remember = ask_language(saved)
            if lang is None:          # fenêtre fermée : on quitte
                return 0
            settings.setValue("lang_remember", remember)
    settings.setValue("lang", lang)
    i18n.set_lang(lang)

    # ---------------------------------------------------------- démarrage ---
    splash = None
    sp = _asset(f"splash_{lang}.png")
    if not os.path.exists(sp):
        sp = _asset("splash.png")
    if os.path.exists(sp) and "--no-splash" not in sys.argv:
        splash = QSplashScreen(QPixmap(sp), Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

    from .main_window import MainWindow

    win = MainWindow()
    args = [a for i, a in enumerate(sys.argv[1:], 1)
            if not a.startswith("--") and sys.argv[i - 1] != "--lang"]
    for arg in args:
        if os.path.isfile(arg):
            QTimer.singleShot(400, lambda a=arg: win.open_file(a))
            break

    def show():
        win.show()
        if splash:
            splash.finish(win)

    QTimer.singleShot(900 if splash else 0, show)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
