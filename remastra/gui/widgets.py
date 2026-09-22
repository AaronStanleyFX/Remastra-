"""Widgets personnalisés : logo, forme d'onde, spectre, vumètres, schéma d'enceintes."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath,
                           QPen, QRadialGradient)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from . import theme as T


def card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("Card")
    return f


def label(text, obj=None, wrap=False):
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


def grad(x1, y1, x2, y2, stops=((0, T.ACCENT1), (0.55, T.ACCENT2), (1, T.ACCENT3))):
    g = QLinearGradient(x1, y1, x2, y2)
    for p, c in stops:
        g.setColorAt(p, QColor(c))
    return g


# --------------------------------------------------------------------------- #
class Logo(QWidget):
    """Logo vectoriel : onde + wordmark dégradé."""

    def __init__(self, size=28, tagline=True, parent=None):
        super().__init__(parent)
        self.sz = size
        self.tagline = tagline
        self.setMinimumHeight(int(size * (2.0 if tagline else 1.5)))
        self.setMinimumWidth(int(size * 10.5))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.sz
        # Emblème : cercle + onde
        r = QRectF(2, 4, s * 1.35, s * 1.35)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad(r.left(), r.top(), r.right(), r.bottom())))
        p.drawEllipse(r)
        p.setPen(QPen(QColor("white"), max(2, s / 11), Qt.SolidLine, Qt.RoundCap))
        cx, cy = r.center().x(), r.center().y()
        hs = [0.25, 0.55, 0.9, 0.55, 0.3]
        for i, h in enumerate(hs):
            x = cx + (i - 2) * s * 0.2
            p.drawLine(QPointF(x, cy - h * s * 0.38), QPointF(x, cy + h * s * 0.38))
        # Wordmark
        f = QFont("Segoe UI", int(s * 0.78), QFont.Black)
        f.setLetterSpacing(QFont.AbsoluteSpacing, s * 0.12)
        p.setFont(f)
        x0 = r.right() + s * 0.35
        path = QPainterPath()
        path.addText(x0, r.top() + s * 1.03, f, "REMASTRA")
        br = path.boundingRect()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad(br.left(), 0, br.right(), 0,
                               ((0, "#FFFFFF"), (0.45, "#FFC2EC"), (1, T.ACCENT3)))))
        p.drawPath(path)
        if self.tagline:
            f2 = QFont("Segoe UI", max(7, int(s * 0.3)), QFont.DemiBold)
            f2.setLetterSpacing(QFont.AbsoluteSpacing, s * 0.09)
            p.setFont(f2)
            p.setPen(QColor(T.MUTED))
            p.drawText(QPointF(x0 + 2, r.top() + s * 1.55), "AI AUDIO REMASTER STUDIO")
        p.end()


# --------------------------------------------------------------------------- #
class WaveformView(QWidget):
    """Forme d'onde min/max, sélection à la souris, tête de lecture, overlay A/B."""

    seekRequested = Signal(float)          # secondes
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None, height=150, color=None, normalize=False):
        super().__init__(parent)
        self.normalize = normalize
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.color = QColor(color) if color else None
        self.env = None
        self.env_b = None
        self.duration = 0.0
        self.playhead = 0.0
        self.sel = None
        self._drag = None
        self.placeholder = "Glissez un fichier audio ou vidéo ici"
        self.setMouseTracking(True)

    @staticmethod
    def envelope(audio: np.ndarray, n=1600):
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if mono.size == 0:
            return None
        n = min(n, mono.size)
        cut = mono[: (mono.size // n) * n].reshape(n, -1)
        return np.stack([cut.min(axis=1), cut.max(axis=1),
                         np.sqrt(np.mean(cut ** 2, axis=1))])

    def set_audio(self, audio, sr, overlay=None):
        self.env = self.envelope(audio) if audio is not None else None
        if self.env is not None and self.normalize:
            self.env = self.env / (np.max(np.abs(self.env[:2])) + 1e-9)
        self.env_b = self.envelope(overlay) if overlay is not None else None
        self.duration = audio.shape[-1] / sr if audio is not None else 0.0
        self.sel = None
        self.update()

    def set_playhead(self, sec):
        self.playhead = sec
        self.update()

    def _x2t(self, x):
        return max(0.0, min(self.duration, x / max(1, self.width()) * self.duration))

    def mousePressEvent(self, e):
        if self.duration <= 0:
            return
        self._drag = e.position().x()

    def mouseMoveEvent(self, e):
        if self._drag is not None and abs(e.position().x() - self._drag) > 4:
            a, b = sorted([self._x2t(self._drag), self._x2t(e.position().x())])
            self.sel = (a, b)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        if self.sel and abs(e.position().x() - self._drag) > 4:
            self.selectionChanged.emit(*self.sel)
        else:
            self.sel = None
            self.seekRequested.emit(self._x2t(e.position().x()))
        self._drag = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor("#0E1122"))
        bg.setColorAt(1, QColor("#090B17"))
        p.fillRect(self.rect(), bg)
        p.setPen(QPen(QColor(255, 255, 255, 14)))
        for i in range(1, 8):
            p.drawLine(int(w * i / 8), 0, int(w * i / 8), h)
        p.drawLine(0, h // 2, w, h // 2)
        if self.env is None:
            p.setPen(QColor(T.MUTED))
            f = p.font()
            f.setPointSize(11)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            p.end()
            return
        mid = h / 2
        amp = h / 2 * 0.92

        def draw(env, fill, rms_fill):
            n = env.shape[1]
            xs = np.linspace(0, w, n)
            path = QPainterPath()
            path.moveTo(xs[0], mid - env[1, 0] * amp)
            for i in range(1, n):
                path.lineTo(xs[i], mid - env[1, i] * amp)
            for i in range(n - 1, -1, -1):
                path.lineTo(xs[i], mid - env[0, i] * amp)
            path.closeSubpath()
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(path)
            rp = QPainterPath()
            rp.moveTo(xs[0], mid - env[2, 0] * amp)
            for i in range(1, n):
                rp.lineTo(xs[i], mid - env[2, i] * amp)
            for i in range(n - 1, -1, -1):
                rp.lineTo(xs[i], mid + env[2, i] * amp)
            p.setBrush(rms_fill)
            p.drawPath(rp)

        if self.env_b is not None:
            c = QColor(T.MUTED)
            c.setAlpha(60)
            draw(self.env_b, c, QColor(255, 255, 255, 25))
        if self.color is not None:
            c1 = QColor(self.color)
            c1b = QColor(self.color)
            c1b.setAlpha(150)
            g = grad(0, 0, w, 0, ((0, c1.name()), (1, c1b.name(QColor.HexArgb))))
        else:
            g = grad(0, 0, w, 0)
        c2 = QColor("white")
        c2.setAlpha(70)
        draw(self.env, QBrush(g), c2)

        if self.sel:
            x1 = self.sel[0] / self.duration * w
            x2 = self.sel[1] / self.duration * w
            p.fillRect(QRectF(x1, 0, x2 - x1, h), QColor(43, 217, 254, 40))
            p.setPen(QPen(QColor(T.ACCENT3), 1))
            p.drawLine(QPointF(x1, 0), QPointF(x1, h))
            p.drawLine(QPointF(x2, 0), QPointF(x2, h))
        if self.duration > 0:
            x = self.playhead / self.duration * w
            p.setPen(QPen(QColor("white"), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            p.setBrush(QColor("white"))
            p.drawEllipse(QPointF(x, 5), 4, 4)
        p.end()


# --------------------------------------------------------------------------- #
class SpectrumView(QWidget):
    """Spectre moyen avant/après + courbe d'EQ appliquée."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.a = self.b = self.eq = None

    def set_data(self, before=None, after=None, eq=None):
        self.a, self.b, self.eq = before, after, eq
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0A0C18"))
        L, R, Tm, B = 38, 10, 10, 22
        pw, ph = w - L - R, h - Tm - B
        fmin, fmax = 20, 20000

        def fx(f):
            return L + (math.log10(f) - math.log10(fmin)) / (math.log10(fmax) - math.log10(fmin)) * pw

        p.setFont(QFont("Segoe UI", 8))
        for f in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            x = fx(f)
            p.setPen(QPen(QColor(255, 255, 255, 18)))
            p.drawLine(QPointF(x, Tm), QPointF(x, Tm + ph))
            p.setPen(QColor(T.MUTED))
            p.drawText(QPointF(x - 10, h - 6), f"{f // 1000}k" if f >= 1000 else str(f))
        series = [s for s in (self.a, self.b) if s is not None]
        if not series:
            p.setPen(QColor(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Spectre — lancez une analyse")
            p.end()
            return
        top = max(np.max(s[1]) for s in series) + 3
        rng = 70.0

        def fy(db):
            return Tm + (top - db) / rng * ph

        for i in range(0, 8):
            y = Tm + i * ph / 7
            p.setPen(QPen(QColor(255, 255, 255, 12)))
            p.drawLine(QPointF(L, y), QPointF(L + pw, y))

        def curve(s, color, fill_alpha):
            f, d = s
            m = (f >= fmin) & (f <= fmax)
            f, d = f[m], d[m]
            path = QPainterPath()
            path.moveTo(fx(f[0]), fy(d[0]))
            for fi, di in zip(f[1:], d[1:]):
                path.lineTo(fx(fi), max(Tm, min(Tm + ph, fy(di))))
            fill = QPainterPath(path)
            fill.lineTo(fx(f[-1]), Tm + ph)
            fill.lineTo(fx(f[0]), Tm + ph)
            c = QColor(color)
            gg = QLinearGradient(0, Tm, 0, Tm + ph)
            c.setAlpha(fill_alpha)
            gg.setColorAt(0, c)
            c2 = QColor(color)
            c2.setAlpha(0)
            gg.setColorAt(1, c2)
            p.setPen(Qt.NoPen)
            p.setBrush(gg)
            p.drawPath(fill)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(color), 2))
            p.drawPath(path)

        if self.a is not None:
            curve(self.a, "#8A90B4", 50)
        if self.b is not None:
            curve(self.b, T.ACCENT1, 90)
        if self.eq is not None:
            f, g = self.eq
            p.setPen(QPen(QColor(T.ACCENT3), 2, Qt.DashLine))
            path = QPainterPath()
            y0 = Tm + ph / 2
            first = True
            for fi, gi in zip(f, g):
                if fi < fmin or fi > fmax:
                    continue
                pt = QPointF(fx(fi), y0 - gi * ph / 30)
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            p.drawPath(path)
        # Légende
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        items = ([("Avant", "#8A90B4")] if self.a is not None and self.b is not None else []) + \
            ([("Après", T.ACCENT1)] if self.b is not None else []) + \
            ([("EQ IA", T.ACCENT3)] if self.eq is not None else [])
        x = L + 8
        for name, col in items:
            p.setBrush(QColor(col))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(x, Tm + 6, 10, 10), 3, 3)
            p.setPen(QColor(T.TEXT))
            p.drawText(QPointF(x + 14, Tm + 15), name)
            x += 70
        p.end()


# --------------------------------------------------------------------------- #
class MetricTile(QFrame):
    def __init__(self, title, unit="", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.t = label(title.upper(), "Muted")
        self.t.setStyleSheet("font-size:10px; font-weight:700; letter-spacing:1px;")
        self.v = QLabel("—")
        self.v.setStyleSheet("font-size:22px; font-weight:800;")
        self.u = label(unit, "Muted")
        self.u.setStyleSheet("font-size:11px;")
        lay.addWidget(self.t)
        lay.addWidget(self.v)
        lay.addWidget(self.u)

    def set(self, value, color=None, sub=None):
        self.v.setText(value)
        self.v.setStyleSheet(f"font-size:22px; font-weight:800; color:{color or T.TEXT};")
        if sub is not None:
            self.u.setText(sub)


class LabeledSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, title, lo, hi, val, suffix="", step=1.0, decimals=0, parent=None):
        super().__init__(parent)
        self.step, self.dec, self.suffix = step, decimals, suffix
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.t = QLabel(title)
        self.t.setStyleSheet(f"color:{T.MUTED}; font-weight:600;")
        self.v = QLabel()
        self.v.setStyleSheet(f"color:{T.TEXT}; font-weight:700;")
        top.addWidget(self.t)
        top.addStretch()
        top.addWidget(self.v)
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(int(lo / step), int(hi / step))
        self.s.setValue(int(val / step))
        self.s.valueChanged.connect(self._ch)
        lay.addLayout(top)
        lay.addWidget(self.s)
        self._ch()

    def _ch(self, *_):
        self.v.setText(f"{self.value():.{self.dec}f}{self.suffix}")
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        return self.s.value() * self.step

    def setValue(self, v):
        self.s.setValue(int(v / self.step))


# --------------------------------------------------------------------------- #
SPEAKER_POS = {
    # nom : (azimut°, anneau)  anneau 0 = plan horizontal, 1 = hauteurs
    "C": (0, 0), "L": (-30, 0), "R": (30, 0), "Lw": (-60, 0), "Rw": (60, 0),
    "Ls": (-110, 0), "Rs": (110, 0), "Lss": (-90, 0), "Rss": (90, 0),
    "Lrs": (-145, 0), "Rrs": (145, 0),
    "Ltf": (-45, 1), "Rtf": (45, 1), "Ltm": (-90, 1), "Rtm": (90, 1),
    "Ltr": (-135, 1), "Rtr": (135, 1),
}


class SpeakerMap(QWidget):
    """Schéma vue de dessus de la configuration d'enceintes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 240)
        self.names: list[str] = ["L", "R"]

    def set_layout(self, names):
        self.names = list(names)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2 + 6
        R = min(w, h) * 0.40
        rg = QRadialGradient(cx, cy, R * 1.25)
        rg.setColorAt(0, QColor(120, 75, 160, 70))
        rg.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(rg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), R * 1.25, R * 1.25)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.DashLine))
        p.drawEllipse(QPointF(cx, cy), R, R)
        p.drawEllipse(QPointF(cx, cy), R * 0.55, R * 0.55)
        # auditeur
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(T.ACCENT3))
        p.drawEllipse(QPointF(cx, cy), 7, 7)
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        for nm in self.names:
            if nm == "LFE":
                x, y = cx, cy - R * 0.66
                p.setBrush(QColor(T.WARN))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRectF(x - 13, y - 9, 26, 18), 4, 4)
                p.setPen(QColor("#111"))
                p.drawText(QRectF(x - 13, y - 9, 26, 18), Qt.AlignCenter, "LFE")
                continue
            az, ring = SPEAKER_POS.get(nm, (0, 0))
            if len(self.names) == 1:
                az = 0
            rr = R if ring == 0 else R * 0.55
            a = math.radians(az)
            x, y = cx + rr * math.sin(a), cy - rr * math.cos(a)
            col = QColor(T.ACCENT1) if ring == 0 else QColor(T.ACCENT3)
            glow = QRadialGradient(x, y, 20)
            c = QColor(col)
            c.setAlpha(110)
            glow.setColorAt(0, c)
            glow.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(glow)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, y), 20, 20)
            p.setBrush(col)
            if ring == 0:
                p.drawEllipse(QPointF(x, y), 11, 11)
            else:
                p.drawRoundedRect(QRectF(x - 11, y - 11, 22, 22), 5, 5)
            p.setPen(QColor("white"))
            p.drawText(QRectF(x - 20, y + 12, 40, 14), Qt.AlignCenter, nm)
        p.setPen(QColor(T.MUTED))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(0, 2, w, 14), Qt.AlignCenter, "▲ AVANT")
        p.setBrush(QColor(T.ACCENT1))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(10, h - 10), 4, 4)
        p.setBrush(QColor(T.ACCENT3))
        p.drawRect(QRectF(80, h - 14, 8, 8))
        p.setPen(QColor(T.MUTED))
        p.drawText(QPointF(18, h - 6), "Horizontal")
        p.drawText(QPointF(92, h - 6), "Hauteur")
        p.end()


class DropArea(QFrame):
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())
