"""Fenêtre principale REMASTRA."""
from __future__ import annotations

import os
import tempfile

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSlider, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import __version__
from ..core import analysis as an
from ..core import audio_io, denoise, export, mastering, spatial, stems
from . import theme as T
from .widgets import (DropArea, LabeledSlider, Logo, MetricTile, SpeakerMap, SpectrumView,
                      WaveformView, card, label)
from .workers import Worker

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:  # pragma: no cover
    QMediaPlayer = QAudioOutput = None


def fmt_time(s):
    s = max(0, s)
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def scroll(widget):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(widget)
    widget.setObjectName("Page")
    widget.setStyleSheet("QWidget#Page { background: transparent; }")
    sa.viewport().setObjectName("PageViewport")
    sa.viewport().setStyleSheet("QWidget#PageViewport { background: transparent; }")
    return sa


class MainWindow(QMainWindow):
    STAGES = ["Original", "Débruité", "Master", "Mix stems"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"REMASTRA — AI Audio Remaster Studio  v{__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        self.tmp = tempfile.mkdtemp(prefix="remastra_")
        self.path = None
        self.sr = 48000
        self.audio: dict[str, np.ndarray | None] = {k: None for k in self.STAGES}
        self.stems: dict[str, np.ndarray] = {}
        self.stem_ctrl: dict[str, dict] = {}
        self.master_res = None
        self.reference = None
        self.noise_sel = None
        self.worker = None
        self.preview_files: dict[str, str] = {}
        self.listen = "Original"

        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_topbar())
        mid = QHBoxLayout()
        mid.setSpacing(0)
        mid.addWidget(self._build_sidebar())
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(20, 16, 20, 10)
        cv.setSpacing(14)
        cv.addWidget(self._build_wave_card())
        self.stack = QStackedWidget()
        for builder in (self._page_import, self._page_denoise, self._page_stems,
                        self._page_master, self._page_export):
            self.stack.addWidget(scroll(builder()))
        cv.addWidget(self.stack, 1)
        mid.addWidget(center, 1)
        v.addLayout(mid, 1)
        v.addWidget(self._build_bottombar())
        self._setup_player()
        self._nav[0].setChecked(True)
        self._refresh_enabled()
        QTimer.singleShot(300, self._probe_engines)
        act = QAction(self)
        act.setShortcut(QKeySequence(Qt.Key_Space))
        act.triggered.connect(self.toggle_play)
        self.addAction(act)

    # ================================================================== UI ==
    def _build_topbar(self):
        bar = QWidget(objectName="TopBar")
        bar.setFixedHeight(74)
        h = QHBoxLayout(bar)
        h.setContentsMargins(18, 8, 18, 8)
        h.addWidget(Logo(28))
        h.addStretch()
        self.btn_prev = QPushButton("⏮", objectName="Transport")
        self.btn_play = QPushButton("▶", objectName="Transport")
        self.btn_stop = QPushButton("■", objectName="Transport")
        self.btn_play.setStyleSheet(f"background:{T.ACCENT1}; border:none; color:white;")
        for b in (self.btn_prev, self.btn_play, self.btn_stop):
            h.addWidget(b)
        self.btn_prev.clicked.connect(lambda: self.player and self.player.setPosition(0))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop)
        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setStyleSheet("font-family:Consolas,monospace; font-size:15px; font-weight:700; padding:0 12px;")
        h.addWidget(self.lbl_time)
        h.addSpacing(10)
        h.addWidget(label("ÉCOUTE", "Muted"))
        self.stage_btns = QButtonGroup(self)
        for i, st in enumerate(self.STAGES):
            b = QPushButton(st, objectName="Chip")
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.stage_btns.addButton(b, i)
            h.addWidget(b)
        self.stage_btns.idClicked.connect(lambda i: self.set_listen(self.STAGES[i]))
        h.addSpacing(10)
        h.addWidget(label("🔊", None))
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(80)
        self.vol.setFixedWidth(110)
        h.addWidget(self.vol)
        return bar

    def _build_sidebar(self):
        sb = QWidget(objectName="Sidebar")
        sb.setFixedWidth(250)
        v = QVBoxLayout(sb)
        v.setContentsMargins(14, 20, 14, 16)
        v.setSpacing(6)
        v.addWidget(label("WORKFLOW", "Muted"))
        self._nav = []
        grp = QButtonGroup(self)
        items = ["①  Import && Analyse", "②  Débruitage IA", "③  Stems IA",
                 "④  Remaster IA", "⑤  Export"]
        for i, t in enumerate(items):
            b = QPushButton(t, objectName="Nav")
            b.setCheckable(True)
            grp.addButton(b, i)
            v.addWidget(b)
            self._nav.append(b)
        grp.idClicked.connect(lambda i: self.stack.setCurrentIndex(i))
        v.addStretch()
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.addWidget(label("MOTEURS IA", "Muted"))
        self.lbl_engines = label("Détection…", None, True)
        self.lbl_engines.setStyleSheet("font-size:12px; line-height:150%;")
        cl.addWidget(self.lbl_engines)
        v.addWidget(c)
        return sb

    def _build_wave_card(self):
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 12)
        top = QHBoxLayout()
        self.lbl_file = label("Aucun fichier chargé", "H2")
        self.lbl_fileinfo = label("", "Muted")
        top.addWidget(self.lbl_file)
        top.addSpacing(10)
        top.addWidget(self.lbl_fileinfo)
        top.addStretch()
        self.lbl_sel = label("Astuce : cliquez pour vous déplacer, glissez pour sélectionner", "Muted")
        top.addWidget(self.lbl_sel)
        v.addLayout(top)
        self.wave = WaveformView(height=140)
        self.wave.seekRequested.connect(self.seek)
        self.wave.selectionChanged.connect(self._on_selection)
        v.addWidget(self.wave)
        return c

    def _build_bottombar(self):
        bar = QWidget(objectName="BottomBar")
        v = QVBoxLayout(bar)
        v.setContentsMargins(18, 8, 18, 8)
        h = QHBoxLayout()
        self.lbl_status = label("Prêt.", "Muted")
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setFixedWidth(360)
        self.prog.setValue(0)
        self.btn_log = QPushButton("Journal ▾", objectName="Chip")
        self.btn_log.setCheckable(True)
        h.addWidget(self.lbl_status, 1)
        h.addWidget(self.prog)
        h.addWidget(self.btn_log)
        v.addLayout(h)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setFixedHeight(130)
        self.logbox.hide()
        self.btn_log.toggled.connect(self.logbox.setVisible)
        v.addWidget(self.logbox)
        return bar

    def _page_header(self, v, title, subtitle, tag=None):
        h = QHBoxLayout()
        h.addWidget(label(title, "H1"))
        if tag:
            h.addWidget(label(tag, "Tag"))
        h.addStretch()
        v.addLayout(h)
        v.addWidget(label(subtitle, "Muted", True))

    # ------------------------------------------------------------ Import --
    def _page_import(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        self._page_header(v, "Import & Analyse",
                          "Musique, doublage, voix, podcast, bande-son vidéo… REMASTRA analyse "
                          "le loudness (EBU R128 / ITU-R BS.1770-4), le true-peak, la dynamique "
                          "et le spectre.")
        drop = DropArea()
        drop.setObjectName("Card")
        drop.setStyleSheet(f"QFrame#Card {{ border: 2px dashed {T.ACCENT2}; }}")
        drop.fileDropped.connect(self.open_file)
        dl = QHBoxLayout(drop)
        dl.setContentsMargins(24, 22, 24, 22)
        col = QVBoxLayout()
        col.addWidget(label("Déposez votre fichier ici", "H2"))
        col.addWidget(label("WAV · FLAC · MP3 · OGG · OPUS · M4A · AAC · AIFF · WMA · "
                            "MP4 · MOV · MKV (piste audio)", "Muted"))
        dl.addLayout(col)
        dl.addStretch()
        b = QPushButton("Ouvrir un fichier…", objectName="Primary")
        b.clicked.connect(self.browse)
        dl.addWidget(b)
        v.addWidget(drop)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.tiles = {
            "lufs": MetricTile("Loudness intégré", "LUFS"),
            "tp": MetricTile("True Peak", "dBTP"),
            "lra": MetricTile("Loudness Range", "LU"),
            "crest": MetricTile("Facteur de crête", "dB"),
            "corr": MetricTile("Corrélation stéréo", "-1 … +1"),
            "type": MetricTile("Contenu détecté", "analyse IA"),
        }
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, 0, i)
        v.addLayout(grid)
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.addWidget(label("Spectre moyen", "H2"))
        self.spec_import = SpectrumView()
        cl.addWidget(self.spec_import)
        v.addWidget(c)
        v.addStretch()
        return w

    # ----------------------------------------------------------- Denoise --
    def _page_denoise(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        self._page_header(v, "Débruitage IA",
                          "Supprime proprement le bruit derrière la voix : souffle, ventilation, "
                          "trafic, réverbération de bruit, ronflement secteur, clics. Le moteur "
                          "« Isolation voix » retire même la musique et les ambiances.", "NEURAL")
        row = QHBoxLayout()
        row.setSpacing(14)
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 16, 18, 16)
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(14)
        g.addWidget(label("Moteur", "H2"), 0, 0)
        self.dn_engine = QComboBox()
        self.dn_engine.addItems(denoise.ENGINES)
        g.addWidget(self.dn_engine, 0, 1)
        self.dn_strength = LabeledSlider("Intensité", 0, 100, 80, " %")
        self.dn_reduction = LabeledSlider("Réduction maximale", 6, 60, 30, " dB")
        self.dn_residual = LabeledSlider("Fond conservé (isolation)", -60, 0, -60, " dB")
        g.addWidget(self.dn_strength, 1, 0)
        g.addWidget(self.dn_reduction, 1, 1)
        g.addWidget(self.dn_residual, 2, 0)
        self.dn_engine.currentIndexChanged.connect(self._dn_engine_changed)
        self._dn_engine_changed()
        g.addWidget(label("Restauration", "H2"), 3, 0, 1, 2)
        self.dn_rumble = QCheckBox("Filtre anti-rumble (< 60 Hz)")
        self.dn_rumble.setChecked(True)
        self.dn_hum = QCheckBox("Anti-ronflement secteur")
        self.dn_hum.setChecked(True)
        self.dn_humf = QComboBox()
        self.dn_humf.addItems(["Auto", "50 Hz", "60 Hz"])
        self.dn_click = QCheckBox("Anti-clic / crépitements")
        self.dn_deess = QCheckBox("De-esser (sifflantes)")
        g.addWidget(self.dn_rumble, 4, 0)
        hh = QHBoxLayout()
        hh.addWidget(self.dn_hum)
        hh.addWidget(self.dn_humf)
        hh.addStretch()
        g.addLayout(hh, 4, 1)
        g.addWidget(self.dn_click, 5, 0)
        g.addWidget(self.dn_deess, 5, 1)
        row.addWidget(c, 3)

        c2 = card()
        c2l = QVBoxLayout(c2)
        c2l.setContentsMargins(18, 16, 18, 16)
        c2l.addWidget(label("Profil de bruit", "H2"))
        c2l.addWidget(label("Moteur Spectral Pro : sélectionnez à la souris un passage de bruit "
                            "seul sur la forme d'onde pour un apprentissage précis. Sans sélection, "
                            "le profil est estimé automatiquement. Idéal voix / dialogues ; pour la "
                            "musique, préférez les moteurs IA ou une sélection de bruit seul.", "Muted", True))
        self.lbl_profile = label("Profil : automatique", None)
        self.lbl_profile.setStyleSheet(f"color:{T.ACCENT3}; font-weight:700;")
        c2l.addWidget(self.lbl_profile)
        b = QPushButton("Effacer la sélection")
        b.clicked.connect(self._clear_sel)
        c2l.addWidget(b)
        c2l.addStretch()
        self.btn_denoise = QPushButton("✦  Lancer le débruitage", objectName="Primary")
        self.btn_denoise.clicked.connect(self.run_denoise)
        c2l.addWidget(self.btn_denoise)
        row.addWidget(c2, 2)
        v.addLayout(row)
        v.addStretch()
        return w

    def _dn_engine_changed(self, *_):
        iso = self.dn_engine.currentText().startswith("IA Isolation")
        self.dn_residual.setEnabled(iso)
        self.dn_reduction.setEnabled(self.dn_engine.currentText().startswith("Spectral"))

    # ------------------------------------------------------------- Stems --
    def _page_stems(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        self._page_header(v, "Séparation de stems IA",
                          "Voix · Batterie · Basse · Guitare · Piano · Cordes · Nappes · Synthés. "
                          "Moteur Hybrid Transformer Demucs v4 + séparation étendue REMASTRA. "
                          "Les stems se somment exactement au mix d'origine.", "8 STEMS")
        c = card()
        h = QHBoxLayout(c)
        h.setContentsMargins(18, 14, 18, 14)
        h.addWidget(label("Source", "Muted"))
        self.st_src = QComboBox()
        self.st_src.addItems(["Auto (dernière étape)", "Original", "Débruité"])
        h.addWidget(self.st_src)
        h.addSpacing(16)
        h.addWidget(label("Qualité", "Muted"))
        self.st_q = QComboBox()
        self.st_q.addItems(list(stems.QUALITY))
        self.st_q.setCurrentIndex(1)
        h.addWidget(self.st_q)
        h.addSpacing(16)
        self.st_ext = QCheckBox("Séparation étendue (cordes / nappes / synthés)")
        self.st_ext.setChecked(True)
        h.addWidget(self.st_ext)
        h.addStretch()
        self.btn_stems = QPushButton("✦  Séparer les stems", objectName="Primary")
        self.btn_stems.clicked.connect(self.run_stems)
        h.addWidget(self.btn_stems)
        v.addWidget(c)
        self.stems_box = card()
        self.stems_lay = QVBoxLayout(self.stems_box)
        self.stems_lay.setContentsMargins(14, 12, 14, 12)
        self.stems_lay.setSpacing(8)
        self.stems_lay.addWidget(label("Aucun stem pour l'instant.", "Muted"))
        v.addWidget(self.stems_box)
        hb = QHBoxLayout()
        self.btn_stem_mix = QPushButton("Écouter le mix des stems")
        self.btn_stem_mix.clicked.connect(self.update_stem_mix)
        self.btn_stem_to_master = QPushButton("Envoyer le mix au Remaster →")
        self.btn_stem_to_master.clicked.connect(self._stem_mix_to_master)
        hb.addWidget(self.btn_stem_mix)
        hb.addWidget(self.btn_stem_to_master)
        hb.addStretch()
        v.addLayout(hb)
        v.addStretch()
        return w

    def _build_stem_rows(self):
        while self.stems_lay.count():
            it = self.stems_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.stem_ctrl = {}
        for name, a in self.stems.items():
            col = stems.STEM_COLORS.get(name, T.ACCENT1)
            row = QWidget()
            row.setFixedHeight(54)
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            bar = QLabel()
            bar.setFixedSize(5, 46)
            bar.setStyleSheet(f"background:{col}; border-radius:2px;")
            h.addWidget(bar)
            nm = QLabel(stems.STEM_LABELS.get(name, name))
            nm.setFixedWidth(120)
            nm.setStyleSheet("font-weight:700; font-size:14px;")
            h.addWidget(nm)
            wv = WaveformView(height=46, color=col, normalize=True)
            wv.setMinimumHeight(46)
            wv.setMaximumHeight(46)
            wv.set_audio(a, self.sr)
            h.addWidget(wv, 1)
            mute = QPushButton("M", objectName="Chip")
            solo = QPushButton("S", objectName="Chip")
            for b in (mute, solo):
                b.setCheckable(True)
                b.setFixedWidth(34)
                h.addWidget(b)
            gain = QSlider(Qt.Horizontal)
            gain.setRange(-240, 120)
            gain.setValue(0)
            gain.setFixedWidth(120)
            gl = QLabel("0.0 dB")
            gl.setFixedWidth(56)
            gain.valueChanged.connect(lambda val, l=gl: l.setText(f"{val / 10:+.1f} dB"))
            h.addWidget(gain)
            h.addWidget(gl)
            ex = QPushButton("Exporter")
            ex.clicked.connect(lambda _=False, n=name: self.export_single_stem(n))
            h.addWidget(ex)
            self.stems_lay.addWidget(row)
            self.stem_ctrl[name] = {"mute": mute, "solo": solo, "gain": gain}

    # ------------------------------------------------------------ Master --
    def _page_master(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        self._page_header(v, "Remastérisation IA",
                          "Analyse intelligente → EQ corrective à phase linéaire, compression "
                          "multibande adaptative, excitateur harmonique, image stéréo M/S, glue, "
                          "normalisation loudness et limiteur true-peak x4.", "AI MASTER")
        row = QHBoxLayout()
        row.setSpacing(14)
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 16, 18, 16)
        g.setHorizontalSpacing(22)
        g.setVerticalSpacing(12)
        self.m_src = QComboBox()
        self.m_src.addItems(["Auto (dernière étape)", "Original", "Débruité", "Mix stems"])
        self.m_profile = QComboBox()
        self.m_profile.addItems(list(mastering.PROFILES))
        self.m_loud = QComboBox()
        self.m_loud.addItems(list(mastering.LOUDNESS_TARGETS) + ["Personnalisé…"])
        self.m_custom = QDoubleSpinBox()
        self.m_custom.setRange(-40, -5)
        self.m_custom.setValue(-14)
        self.m_custom.setSuffix(" LUFS")
        self.m_custom.setEnabled(False)
        self.m_loud.currentTextChanged.connect(lambda t: self.m_custom.setEnabled(t == "Personnalisé…"))
        self.m_ceiling = QDoubleSpinBox()
        self.m_ceiling.setRange(-6, 0)
        self.m_ceiling.setSingleStep(0.1)
        self.m_ceiling.setValue(-1.0)
        self.m_ceiling.setSuffix(" dBTP")
        self.m_char = QComboBox()
        self.m_char.addItems(mastering.CHARACTERS)
        pairs = [("Source", self.m_src), ("Profil", self.m_profile), ("Cible loudness", self.m_loud),
                 ("LUFS perso", self.m_custom), ("Plafond true-peak", self.m_ceiling),
                 ("Caractère", self.m_char)]
        for i, (t, wd) in enumerate(pairs):
            g.addWidget(label(t, "Muted"), i // 2 * 2, i % 2)
            g.addWidget(wd, i // 2 * 2 + 1, i % 2)
        self.m_int = LabeledSlider("Correction EQ IA", 0, 100, 70, " %")
        self.m_comp = LabeledSlider("Compression", 0, 100, 50, " %")
        self.m_exc = LabeledSlider("Excitateur / Air", 0, 100, 20, " %")
        self.m_width = LabeledSlider("Largeur stéréo", 50, 150, 100, " %")
        self.m_mono = LabeledSlider("Basses mono sous", 0, 250, 120, " Hz", 10)
        sl = [self.m_int, self.m_comp, self.m_exc, self.m_width, self.m_mono]
        for i, s in enumerate(sl):
            g.addWidget(s, 6 + i // 2, i % 2)
        rh = QHBoxLayout()
        self.btn_ref = QPushButton("Charger une référence…")
        self.btn_ref.setToolTip("Mastering par référence : REMASTRA reproduit l'équilibre "
                                "tonal, le loudness et la largeur d'un morceau de référence.")
        self.btn_ref.clicked.connect(self.load_reference)
        self.lbl_ref = label("Aucune référence", "Muted")
        rh.addWidget(self.btn_ref)
        rh.addWidget(self.lbl_ref, 1)
        g.addLayout(rh, 9, 0, 1, 2)
        self.btn_master = QPushButton("✦  Remastériser avec l'IA", objectName="Primary")
        self.btn_master.clicked.connect(self.run_master)
        g.addWidget(self.btn_master, 10, 0, 1, 2)
        row.addWidget(c, 5)

        right = QVBoxLayout()
        tiles = QGridLayout()
        tiles.setSpacing(10)
        self.mt = {k: MetricTile(t, u) for k, t, u in [
            ("lufs", "Loudness", "LUFS"), ("tp", "True Peak", "dBTP"),
            ("lra", "LRA", "LU"), ("crest", "Crête", "dB")]}
        for i, t in enumerate(self.mt.values()):
            tiles.addWidget(t, 0, i)
        right.addLayout(tiles)
        c3 = card()
        c3l = QVBoxLayout(c3)
        c3l.setContentsMargins(14, 12, 14, 12)
        c3l.addWidget(label("Spectre avant / après", "H2"))
        self.spec_master = SpectrumView()
        c3l.addWidget(self.spec_master)
        right.addWidget(c3)
        c4 = card()
        c4l = QVBoxLayout(c4)
        c4l.setContentsMargins(14, 12, 14, 12)
        c4l.addWidget(label("Décisions du moteur IA", "H2"))
        self.decisions = QListWidget()
        self.decisions.setMinimumHeight(150)
        c4l.addWidget(self.decisions)
        right.addWidget(c4)
        rw = QWidget()
        rw.setLayout(right)
        row.addWidget(rw, 6)
        v.addLayout(row)
        v.addStretch()
        return w

    # ------------------------------------------------------------ Export --
    def _page_export(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        self._page_header(v, "Export",
                          "WAV (PCM 16/24 bits, 32 bits float, RF64 > 4 Go) ou FLAC sans perte "
                          "(16/24 bits). Du mono jusqu'au Dolby Atmos 9.1.6 avec spatialisation IA.",
                          "LOSSLESS")
        row = QHBoxLayout()
        row.setSpacing(14)
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 16, 18, 16)
        g.setVerticalSpacing(12)
        g.addWidget(label("Source", "Muted"), 0, 0)
        self.x_src = QComboBox()
        self.x_src.addItems(["Master (Remaster IA)", "Voix / audio débruité", "Original",
                             "Mix des stems", "Stems séparés (1 fichier par stem)"])
        g.addWidget(self.x_src, 0, 1, 1, 3)
        g.addWidget(label("Format", "Muted"), 1, 0)
        self.x_wav = QRadioButton("WAV")
        self.x_flac = QRadioButton("FLAC")
        self.x_wav.setChecked(True)
        fh = QHBoxLayout()
        fh.addWidget(self.x_wav)
        fh.addWidget(self.x_flac)
        fh.addStretch()
        g.addLayout(fh, 1, 1)
        g.addWidget(label("Résolution", "Muted"), 1, 2)
        self.x_bits = QComboBox()
        g.addWidget(self.x_bits, 1, 3)
        g.addWidget(label("Fréquence", "Muted"), 2, 0)
        self.x_sr = QComboBox()
        for s in export.SAMPLE_RATES:
            self.x_sr.addItem(f"{s / 1000:g} kHz", s)
        self.x_sr.setCurrentIndex(1)
        g.addWidget(self.x_sr, 2, 1)
        self.x_dither = QCheckBox("Dither TPDF + noise shaping")
        self.x_dither.setChecked(True)
        g.addWidget(self.x_dither, 2, 2, 1, 2)
        g.addWidget(label("Canaux", "H2"), 3, 0, 1, 4)
        lg = QGridLayout()
        lg.setSpacing(8)
        self.layout_grp = QButtonGroup(self)
        for i, name in enumerate(spatial.LAYOUTS):
            b = QPushButton(name, objectName="LayoutBtn")
            b.setCheckable(True)
            b.setMinimumHeight(52)
            self.layout_grp.addButton(b, i)
            lg.addWidget(b, i // 4, i % 4)
        self.layout_grp.button(1).setChecked(True)
        self.layout_grp.idClicked.connect(self._layout_changed)
        g.addLayout(lg, 4, 0, 1, 4)
        self.lbl_layout = label("", "Muted", True)
        g.addWidget(self.lbl_layout, 5, 0, 1, 4)
        self.x_usestems = QCheckBox("Spatialisation objet à partir des stems IA (si disponibles)")
        self.x_usestems.setChecked(True)
        g.addWidget(self.x_usestems, 6, 0, 1, 4)
        self.btn_export = QPushButton("⬇  Exporter", objectName="Primary")
        self.btn_export.clicked.connect(self.run_export)
        g.addWidget(self.btn_export, 7, 0, 1, 4)
        row.addWidget(c, 3)
        c2 = card()
        c2l = QVBoxLayout(c2)
        c2l.setContentsMargins(14, 12, 14, 12)
        c2l.addWidget(label("Configuration d'écoute", "H2"))
        self.spk = SpeakerMap()
        c2l.addWidget(self.spk, 1)
        row.addWidget(c2, 2)
        v.addLayout(row)
        v.addStretch()
        self.x_wav.toggled.connect(self._fmt_changed)
        self._fmt_changed()
        self._layout_changed(1)
        return w

    def _fmt_changed(self, *_):
        fmt = "WAV" if self.x_wav.isChecked() else "FLAC"
        self.x_bits.clear()
        self.x_bits.addItems(export.BIT_DEPTHS[fmt])
        self._layout_changed(self.layout_grp.checkedId())

    def _layout_changed(self, i):
        name = list(spatial.LAYOUTS)[max(0, i)]
        self.spk.set_layout(spatial.LAYOUTS[name])
        txt = spatial.LAYOUT_DESC[name]
        if self.x_flac.isChecked() and len(spatial.LAYOUTS[name]) > 8:
            txt += "  —  FLAC est limité à 8 canaux : export multi-mono (1 FLAC par canal)."
        self.lbl_layout.setText(txt)

    # ============================================================ Player ==
    def _setup_player(self):
        self.player = None
        if QMediaPlayer is None:
            return
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.audio_out.setVolume(0.8)
        self.vol.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100))
        self.player.positionChanged.connect(self._on_pos)
        self.player.playbackStateChanged.connect(
            lambda s: self.btn_play.setText("❚❚" if s == QMediaPlayer.PlayingState else "▶"))

    def _on_pos(self, ms):
        s = ms / 1000
        self.wave.set_playhead(s)
        dur = self.wave.duration
        self.lbl_time.setText(f"{fmt_time(s)} / {fmt_time(dur)}")

    def _preview_path(self, stage):
        a = self.audio.get(stage)
        if a is None:
            return None
        if stage not in self.preview_files:
            p = os.path.join(self.tmp, f"preview_{self.STAGES.index(stage)}_{id(a)}.wav")
            audio_io.write_preview_wav(a, self.sr, p)
            self.preview_files[stage] = p
        return self.preview_files[stage]

    def set_listen(self, stage):
        if self.audio.get(stage) is None:
            self.status(f"« {stage} » n'est pas encore disponible.")
            self.stage_btns.button(self.STAGES.index(self.listen)).setChecked(True)
            return
        self.listen = stage
        self.stage_btns.button(self.STAGES.index(stage)).setChecked(True)
        overlay = self.audio["Original"] if stage != "Original" else None
        self.wave.set_audio(self.audio[stage], self.sr, overlay)
        if self.player:
            pos = self.player.position()
            playing = self.player.playbackState() == QMediaPlayer.PlayingState
            self.player.setSource(QUrl.fromLocalFile(self._preview_path(stage)))
            self.player.setPosition(pos)
            if playing:
                self.player.play()

    def toggle_play(self):
        if not self.player or self.audio["Original"] is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.source().isEmpty():
                self.player.setSource(QUrl.fromLocalFile(self._preview_path(self.listen)))
            self.player.play()

    def stop(self):
        if self.player:
            self.player.stop()
            self._on_pos(0)

    def seek(self, sec):
        if self.player:
            self.player.setPosition(int(sec * 1000))
        self._on_pos(sec * 1000)

    def _on_selection(self, a, b):
        self.noise_sel = (a, b)
        self.lbl_sel.setText(f"Sélection : {fmt_time(a)} → {fmt_time(b)}")
        self.lbl_profile.setText(f"Profil : sélection {a:.2f}s – {b:.2f}s")

    def _clear_sel(self):
        self.noise_sel = None
        self.wave.sel = None
        self.wave.update()
        self.lbl_profile.setText("Profil : automatique")
        self.lbl_sel.setText("Astuce : cliquez pour vous déplacer, glissez pour sélectionner")

    # ============================================================ Actions ==
    def status(self, msg):
        self.lbl_status.setText(msg)

    def log(self, msg):
        self.logbox.appendPlainText(msg)
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        if first and not first.startswith("Traceback"):
            self.status(first)

    def _refresh_enabled(self):
        has = self.audio["Original"] is not None
        busy = self.worker is not None and self.worker.isRunning()
        for b in (self.btn_denoise, self.btn_stems, self.btn_master, self.btn_export):
            b.setEnabled(has and not busy)
        self.btn_stem_mix.setEnabled(bool(self.stems) and not busy)
        self.btn_stem_to_master.setEnabled(bool(self.stems) and not busy)

    def _run(self, fn, on_done, *args, title="Traitement", **kw):
        if self.worker and self.worker.isRunning():
            return
        self.prog.setValue(0)
        self.status(f"{title}…")
        self.log(f"── {title} ──")
        self.worker = Worker(fn, *args, **kw)
        self.worker.progress.connect(lambda p: self.prog.setValue(int(p * 1000)))
        self.worker.log.connect(self.log)
        self.worker.done.connect(lambda r: (on_done(r), self.prog.setValue(1000),
                                            self.status(f"{title} terminé ✔"),
                                            self._refresh_enabled()))
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._refresh_enabled)
        self.worker.start()
        self._refresh_enabled()

    def _failed(self, msg):
        self.prog.setValue(0)
        self.status("Erreur : " + msg.splitlines()[0])
        QMessageBox.critical(self, "REMASTRA", msg)
        self._refresh_enabled()

    def _probe_engines(self):
        def probe(progress, log):
            return {"df": denoise.deepfilter_available(), "demucs": stems.demucs_available(),
                    "dev": stems.device_name() if stems.demucs_available() else "cpu"}

        def done(r):
            ok = lambda b: f"<span style='color:{T.OK}'>●</span>" if b else f"<span style='color:{T.BAD}'>●</span>"  # noqa: E731
            self.lbl_engines.setText(
                f"{ok(r['df'])} DeepFilterNet 3<br>{ok(r['demucs'])} Demucs v4 (stems)<br>"
                f"<span style='color:{T.OK}'>●</span> Spectral Pro / Master DSP<br>"
                f"<span style='color:{T.MUTED}'>Calcul : {r['dev'].upper()}</span>")
            if not r["df"]:
                self.dn_engine.setCurrentIndex(2 if not r["demucs"] else 0)
        w = Worker(probe)
        w.done.connect(done)
        w.finished.connect(w.deleteLater)
        self._probe = w
        w.start()

    # ---------------------------------------------------------------- Open --
    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier audio", "", audio_io.AUDIO_FILTER)
        if p:
            self.open_file(p)

    def open_file(self, path):
        def job(progress, log):
            a, sr = audio_io.load_audio(path)
            progress(0.5)
            if sr < 44100:
                log(f"Rééchantillonnage {sr} → 48000 Hz pour le traitement")
                a, sr = audio_io.resample(a, sr, 48000), 48000
            rep = an.analyse(a, sr)
            return a, sr, rep

        def done(r):
            a, sr, rep = r
            self.stop()
            if self.player:
                self.player.setSource(QUrl())
            self.path, self.sr = path, sr
            self.audio = {k: None for k in self.STAGES}
            self.audio["Original"] = a
            self.stems, self.master_res, self.preview_files = {}, None, {}
            self._build_stem_rows_empty()
            self.lbl_file.setText(os.path.basename(path))
            self.lbl_fileinfo.setText(f"{sr / 1000:g} kHz · {a.shape[0]} canal(aux) · "
                                      f"{fmt_time(a.shape[1] / sr)}")
            self._show_report(rep)
            self.set_listen("Original")
            self._on_pos(0)
            self._clear_sel()
            if rep.content == "voix":
                self.m_profile.setCurrentText("Auto (IA)")

        self._run(job, done, title=f"Chargement de {os.path.basename(path)}")

    def _build_stem_rows_empty(self):
        self.stems = {}
        self._build_stem_rows()
        self.stems_lay.addWidget(label("Aucun stem pour l'instant.", "Muted"))

    def _show_report(self, rep: an.Report):
        t = self.tiles
        t["lufs"].set(f"{rep.lufs:.1f}", T.ACCENT3)
        t["tp"].set(f"{rep.true_peak:.1f}", T.BAD if rep.true_peak > -1 else T.OK)
        t["lra"].set(f"{rep.lra:.1f}")
        t["crest"].set(f"{rep.crest:.1f}")
        t["corr"].set(f"{rep.correlation:+.2f}", T.BAD if rep.correlation < 0 else T.TEXT)
        t["type"].set(rep.content.capitalize(), T.ACCENT1)
        self.spec_import.set_data((rep.spectrum_f, rep.spectrum_db))

    def _source(self, choice: str):
        if choice.startswith("Original"):
            return self.audio["Original"]
        if choice.startswith("Débruité"):
            return self.audio["Débruité"] if self.audio["Débruité"] is not None else self.audio["Original"]
        if choice.startswith("Mix"):
            return self.audio["Mix stems"] if self.audio["Mix stems"] is not None else self.audio["Original"]
        for k in ("Débruité", "Original"):
            if self.audio[k] is not None:
                return self.audio[k]
        return None

    def _set_stage(self, stage, a):
        self.audio[stage] = a
        self.preview_files.pop(stage, None)
        self.set_listen(stage)

    # ------------------------------------------------------------- Denoise --
    def run_denoise(self):
        cfg = denoise.DenoiseSettings(
            engine=self.dn_engine.currentText(), strength=self.dn_strength.value(),
            max_reduction_db=self.dn_reduction.value(), residual_db=self.dn_residual.value(),
            dehum=self.dn_hum.isChecked(), hum_freq=self.dn_humf.currentText(),
            declick=self.dn_click.isChecked(), rumble_hp=self.dn_rumble.isChecked(),
            deess=self.dn_deess.isChecked(), noise_profile=self.noise_sel)
        src = self.audio["Original"]
        self._run(denoise.process, lambda y: self._set_stage("Débruité", y), src, self.sr, cfg,
                  title="Débruitage")

    # --------------------------------------------------------------- Stems --
    def run_stems(self):
        src = self._source(self.st_src.currentText())

        def done(d):
            self.stems = d
            self._build_stem_rows()
            self.update_stem_mix()

        self._run(stems.separate, done, src, self.sr, self.st_q.currentText(),
                  self.st_ext.isChecked(), title="Séparation des stems")

    def stem_mix(self):
        if not self.stems:
            return None
        solos = [n for n, c in self.stem_ctrl.items() if c["solo"].isChecked()]
        mix = None
        for n, a in self.stems.items():
            c = self.stem_ctrl.get(n)
            if c is None:
                continue
            if solos and n not in solos:
                continue
            if not solos and c["mute"].isChecked():
                continue
            g = 10 ** (c["gain"].value() / 200)
            mix = a * g if mix is None else mix + a * g
        if mix is None:
            mix = np.zeros_like(next(iter(self.stems.values())))
        return mix.astype(np.float32)

    def update_stem_mix(self):
        m = self.stem_mix()
        if m is not None:
            self._set_stage("Mix stems", m)

    def _stem_mix_to_master(self):
        self.update_stem_mix()
        self.m_src.setCurrentText("Mix stems")
        self._nav[3].click()

    def export_single_stem(self, name):
        p, _ = QFileDialog.getSaveFileName(self, "Exporter le stem",
                                           self._default_name(f"_{name}"), "WAV (*.wav);;FLAC (*.flac)")
        if not p:
            return
        cfg = self._export_cfg()
        cfg.fmt = "FLAC" if p.lower().endswith(".flac") else "WAV"
        if cfg.bits not in export.BIT_DEPTHS[cfg.fmt]:
            cfg.bits = "24 bits"
        cfg.use_stems = False
        a = self.stems[name]
        self._run(lambda progress, log: export.export(p, a, self.sr, cfg, None, log),
                  lambda r: None, title=f"Export du stem {name}")

    # -------------------------------------------------------------- Master --
    def load_reference(self):
        p, _ = QFileDialog.getOpenFileName(self, "Morceau de référence", "", audio_io.AUDIO_FILTER)
        if not p:
            return
        try:
            self.reference = audio_io.load_audio(p)
            self.lbl_ref.setText("Référence : " + os.path.basename(p))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "REMASTRA", str(e))

    def run_master(self):
        src = self._source(self.m_src.currentText())
        loud = self.m_loud.currentText()
        cfg = mastering.MasterSettings(
            profile=self.m_profile.currentText(),
            loudness=loud if loud in mastering.LOUDNESS_TARGETS else "Profil (auto)",
            custom_lufs=self.m_custom.value() if loud == "Personnalisé…" else None,
            ceiling_dbtp=self.m_ceiling.value(), intensity=self.m_int.value(),
            compression=self.m_comp.value(), character=self.m_char.currentText(),
            exciter=self.m_exc.value(), width=self.m_width.value(),
            bass_mono_hz=self.m_mono.value(), reference=self.reference)

        def done(res: mastering.MasterResult):
            self.master_res = res
            self.decisions.clear()
            for d in res.decisions:
                self.decisions.addItem("✦  " + d)
            a, b = res.before, res.after
            self.mt["lufs"].set(f"{b.lufs:.1f}", T.ACCENT3, f"LUFS  (avant {a.lufs:.1f})")
            self.mt["tp"].set(f"{b.true_peak:.1f}", T.OK if b.true_peak <= cfg.ceiling_dbtp + 0.05 else T.BAD,
                              f"dBTP  (avant {a.true_peak:.1f})")
            self.mt["lra"].set(f"{b.lra:.1f}", None, f"LU  (avant {a.lra:.1f})")
            self.mt["crest"].set(f"{b.crest:.1f}", None, f"dB  (avant {a.crest:.1f})")
            self.spec_master.set_data((a.spectrum_f, a.spectrum_db), (b.spectrum_f, b.spectrum_db),
                                      res.eq_curve)
            self._set_stage("Master", res.audio)

        self._run(mastering.master, done, src, self.sr, cfg, title="Remastérisation IA")

    # -------------------------------------------------------------- Export --
    def _export_cfg(self):
        fmt = "WAV" if self.x_wav.isChecked() else "FLAC"
        return export.ExportSettings(
            fmt=fmt, bits=self.x_bits.currentText(), samplerate=self.x_sr.currentData(),
            layout=list(spatial.LAYOUTS)[self.layout_grp.checkedId()],
            use_stems=self.x_usestems.isChecked(), dither=self.x_dither.isChecked())

    def _default_name(self, suffix=""):
        base = os.path.splitext(os.path.basename(self.path or "remastra"))[0]
        d = os.path.dirname(self.path or "")
        return os.path.join(d, f"{base}{suffix}")

    def run_export(self):
        cfg = self._export_cfg()
        src = self.x_src.currentText()
        ext = ".wav" if cfg.fmt == "WAV" else ".flac"
        tag = cfg.layout.replace("é", "e").replace(".", "")
        if src.startswith("Stems séparés"):
            if not self.stems:
                QMessageBox.information(self, "REMASTRA", "Séparez d'abord les stems (étape ③).")
                return
            folder = QFileDialog.getExistingDirectory(self, "Dossier de destination des stems",
                                                      os.path.dirname(self.path or ""))
            if not folder:
                return
            base = os.path.splitext(os.path.basename(self.path))[0]
            items = dict(self.stems)
            sr = self.sr
            cfg.use_stems = False

            def job(progress, log):
                out = []
                for i, (n, a) in enumerate(items.items()):
                    p = os.path.join(folder, f"{base}_{n}_{tag}")
                    out += export.export(p, a, sr, cfg, None, log)
                    progress((i + 1) / len(items))
                return out
            self._run(job, lambda r: self._exported(r), title="Export des stems")
            return
        audio = {"Master": self.audio["Master"], "Voix": self.audio["Débruité"],
                 "Original": self.audio["Original"], "Mix": self.audio["Mix stems"]}[src.split()[0]]
        if audio is None:
            QMessageBox.information(self, "REMASTRA", "Cette source n'est pas encore disponible. "
                                    "Lancez l'étape correspondante ou choisissez une autre source.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Exporter", self._default_name(f"_REMASTRA_{tag}") + ext,
                                           f"{cfg.fmt} (*{ext})")
        if not p:
            return
        st = dict(self.stems) if self.stems else None
        self._run(lambda progress, log: export.export(p, audio, self.sr, cfg, st, log),
                  self._exported, title=f"Export {cfg.fmt} {cfg.layout}")

    def _exported(self, files):
        files = [f for f in files if not f.endswith(".txt")]
        QMessageBox.information(self, "REMASTRA — Export terminé",
                                f"{len(files)} fichier(s) exporté(s) :\n\n" + "\n".join(files[:12]) +
                                ("\n…" if len(files) > 12 else ""))

    def closeEvent(self, e):
        if self.player:
            self.player.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().closeEvent(e)

    def sizeHint(self):
        return QSize(1440, 900)
