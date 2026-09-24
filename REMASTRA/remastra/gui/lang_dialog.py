"""Fenêtre de choix de la langue, affichée au lancement de REMASTRA."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from . import theme as T
from .widgets import Logo

TEXT = {
    "title": "Choose your language  ·  Choisissez votre langue",
    "remember": "Don't ask again  ·  Ne plus demander",
}


class LangCard(QPushButton):
    """Grande carte cliquable : badge circulaire + nom de la langue."""

    chosen = Signal(str)

    def __init__(self, code: str, name: str, sub: str, parent=None):
        super().__init__(parent)
        self.code, self.name, self.sub = code, name, sub
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(230, 190)
        self.setCheckable(True)
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self.clicked.connect(lambda: self.chosen.emit(self.code))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        hot = self.underMouse() or self.hasFocus() or self.isChecked()
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        if hot:
            g.setColorAt(0, QColor(255, 60, 172, 70))
            g.setColorAt(1, QColor(43, 217, 254, 45))
        else:
            g.setColorAt(0, QColor(T.PANEL))
            g.setColorAt(1, QColor(T.PANEL2))
        p.setBrush(g)
        pen_g = QLinearGradient(r.topLeft(), r.topRight())
        pen_g.setColorAt(0, QColor(T.ACCENT1))
        pen_g.setColorAt(1, QColor(T.ACCENT3))
        p.setPen(QPen(pen_g, 2.2) if hot else QPen(QColor(T.BORDER), 1.2))
        p.drawRoundedRect(r, 20, 20)
        # badge
        c = QPointF(r.center().x(), r.top() + 66)
        glow = QRadialGradient(c, 58)
        glow.setColorAt(0, QColor(255, 60, 172, 90 if hot else 40))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, 58, 58)
        bg = QLinearGradient(c.x() - 36, c.y() - 36, c.x() + 36, c.y() + 36)
        bg.setColorAt(0, QColor(T.ACCENT1))
        bg.setColorAt(0.55, QColor(T.ACCENT2))
        bg.setColorAt(1, QColor(T.ACCENT3))
        p.setBrush(bg)
        p.drawEllipse(c, 36, 36)
        f = QFont("Segoe UI", 20, QFont.Black)
        p.setFont(f)
        p.setPen(QColor("white"))
        p.drawText(QRectF(c.x() - 36, c.y() - 36, 72, 72), Qt.AlignCenter, self.code.upper())
        p.setFont(QFont("Segoe UI", 16, QFont.Bold))
        p.drawText(QRectF(r.left(), r.top() + 112, r.width(), 30), Qt.AlignCenter, self.name)
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(T.MUTED))
        p.drawText(QRectF(r.left(), r.top() + 142, r.width(), 22), Qt.AlignCenter, self.sub)
        p.end()

    def enterEvent(self, e):
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.update()
        super().leaveEvent(e)


class LanguageDialog(QDialog):
    def __init__(self, default: str = "fr", parent=None):
        super().__init__(parent)
        self.choice = None
        self.setWindowTitle("REMASTRA")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(640, 470)
        self._drag = None

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 28, 40, 26)
        v.setSpacing(14)
        top = QHBoxLayout()
        top.addWidget(Logo(26))
        top.addStretch()
        close = QPushButton("×")
        close.setFixedSize(30, 30)
        close.setCursor(Qt.PointingHandCursor)
        close.setStyleSheet(f"QPushButton {{ border:none; border-radius:15px; color:{T.MUTED};"
                            f" background:transparent; font-size:20px; }}"
                            f"QPushButton:hover {{ background:{T.PANEL2}; color:white; }}")
        close.clicked.connect(self.reject)
        top.addWidget(close, 0, Qt.AlignTop)
        v.addLayout(top)
        v.addSpacing(6)
        t = QLabel(TEXT["title"])
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-size:17px; font-weight:700; color:white;")
        v.addWidget(t)
        v.addSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(26)
        row.addStretch()
        self.cards = {}
        for code, name, sub in (("en", "English", "Continue in English"),
                                ("fr", "Français", "Continuer en français")):
            c = LangCard(code, name, sub)
            c.chosen.connect(self._choose)
            self.cards[code] = c
            row.addWidget(c)
        row.addStretch()
        v.addLayout(row)
        v.addSpacing(6)
        self.remember = QCheckBox(TEXT["remember"])
        self.remember.setStyleSheet(f"color:{T.MUTED};")
        rh = QHBoxLayout()
        rh.addStretch()
        rh.addWidget(self.remember)
        rh.addStretch()
        v.addLayout(rh)
        self.cards.get(default, self.cards["fr"]).setFocus()

    # --- sélection au clavier : ← → + Entrée --------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Tab):
            other = "fr" if self.cards["en"].hasFocus() else "en"
            self.cards[other].setFocus()
            for c in self.cards.values():
                c.update()
        elif k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._choose("en" if self.cards["en"].hasFocus() else "fr")
        elif k == Qt.Key_Escape:
            self.reject()

    def _choose(self, code):
        self.choice = code
        self.accept()

    # --- fenêtre sans bordure déplaçable -------------------------------------
    def mousePressEvent(self, e):
        self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, 24, 24)
        g = QLinearGradient(0, 0, 0, r.height())
        g.setColorAt(0, QColor("#11132A"))
        g.setColorAt(1, QColor(T.BG))
        p.fillPath(path, g)
        p.setClipPath(path)
        for cx, cy, col in ((0.1, 1.0, QColor(255, 60, 172, 55)), (0.95, 0.0, QColor(43, 217, 254, 45))):
            rg = QRadialGradient(r.width() * cx, r.height() * cy, 300)
            rg.setColorAt(0, col)
            rg.setColorAt(1, QColor(0, 0, 0, 0))
            p.fillRect(r, rg)
        p.setClipping(False)
        p.setPen(QPen(QColor(T.BORDER), 1.2))
        p.drawPath(path)
        p.end()


def ask_language(default: str = "fr") -> tuple[str | None, bool]:
    """Affiche la fenêtre ; retourne (code langue ou None si fermée, mémoriser ?)."""
    dlg = LanguageDialog(default)
    ok = dlg.exec()
    return (dlg.choice if ok else None), dlg.remember.isChecked()


if __name__ == "__main__":  # aperçu rapide
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyleSheet(T.QSS)
    print(ask_language())
