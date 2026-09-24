"""Fenêtre principale REMASTRA."""
from __future__ import annotations

import os
import tempfile

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSlider, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import __version__
from ..i18n import tr, tr_msg
from ..core import analysis as an
from ..core import audio_io, denoise, export, mastering, mega, spatial, stems
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


def fill(combo, keys):
    """Remplit un QComboBox : libellé traduit, clé interne en userData."""
    combo.clear()
    for k in keys:
        combo.addItem(tr(k), k)


def select(combo, key):
    i = combo.findData(key)
    if i >= 0:
        combo.setCurrentIndex(i)


def fill_pairs(combo, pairs):
    """Remplit un QComboBox à partir de (libellé, clé) : libellé traduit, clé en userData."""
    combo.clear()
    for text, k in pairs:
        combo.addItem(tr(text), k)


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
        self.active_stem_model = "demucs"
        self.stem_ctrl: dict[str, dict] = {}
        self.master_res = None
        self.reference = None
        self.worker = None
        self.preview_files: dict[str, str] = {}
        self.listen = "Original"
        self.settings = QSettings("REMASTRA", "REMASTRA")

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
        self._engines_ready = False
        self._pending = None
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
        h.addWidget(label(tr("ÉCOUTE"), "Muted"))
        self.stage_btns = QButtonGroup(self)
        for i, st in enumerate(self.STAGES):
            b = QPushButton(tr(st), objectName="Chip")
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
        h.addSpacing(8)
        from .. import i18n
        other = "fr" if i18n.LANG == "en" else "en"
        self.btn_lang = QPushButton(f"🌐 {i18n.LANG.upper()}", objectName="Chip")
        self.btn_lang.setToolTip(tr("Langue") + f" → {i18n.LANGS[other]}")
        self.btn_lang.clicked.connect(lambda: self.switch_language(other))
        h.addWidget(self.btn_lang)
        return bar

    def switch_language(self, lang):
        """Relance REMASTRA dans l'autre langue."""
        import sys

        from PySide6.QtCore import QProcess
        from PySide6.QtWidgets import QApplication

        self.settings.setValue("lang", lang)
        if getattr(sys, "frozen", False):
            prog, args = sys.executable, ["--lang", lang, "--no-splash"]
        else:
            prog = sys.executable
            args = ["-m", "remastra", "--lang", lang, "--no-splash"]
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if QProcess.startDetached(prog, args, root):
            QApplication.quit()

    def _build_sidebar(self):
        sb = QWidget(objectName="Sidebar")
        sb.setFixedWidth(250)
        v = QVBoxLayout(sb)
        v.setContentsMargins(14, 20, 14, 16)
        v.setSpacing(6)
        v.addWidget(label("WORKFLOW", "Muted"))
        self._nav = []
        grp = QButtonGroup(self)
        items = [tr("①  Import && Analyse"), tr("②  Débruitage IA"), tr("③  Stems IA"),
                 tr("④  Remaster IA"), tr("⑤  Export")]
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
        cl.addWidget(label(tr("MOTEURS IA"), "Muted"))
        self.lbl_engines = label(tr("Détection…"), None, True)
        self.lbl_engines.setStyleSheet("font-size:12px; line-height:150%;")
        cl.addWidget(self.lbl_engines)
        v.addWidget(c)
        return sb

    def _build_wave_card(self):
        c = card()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 12)
        top = QHBoxLayout()
        self.lbl_file = label(tr("Aucun fichier chargé"), "H2")
        self.lbl_fileinfo = label("", "Muted")
        top.addWidget(self.lbl_file)
        top.addSpacing(10)
        top.addWidget(self.lbl_fileinfo)
        top.addStretch()
        self.lbl_sel = label(tr("Astuce : cliquez pour vous déplacer, glissez pour sélectionner"), "Muted")
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
        self.lbl_status = label(tr("Prêt."), "Muted")
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setFixedWidth(360)
        self.prog.setValue(0)
        self.btn_log = QPushButton(tr("Journal ▾"), objectName="Chip")
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
        self._page_header(v, tr("Import & Analyse"),
                          tr("Musique, doublage, voix, podcast, bande-son vidéo… REMASTRA analyse "
                          "le loudness (EBU R128 / ITU-R BS.1770-4), le true-peak, la dynamique "
                          "et le spectre."))
        drop = DropArea()
        drop.setObjectName("Card")
        drop.setStyleSheet(f"QFrame#Card {{ border: 2px dashed {T.ACCENT2}; }}")
        drop.fileDropped.connect(self.open_file)
        dl = QHBoxLayout(drop)
        dl.setContentsMargins(24, 22, 24, 22)
        col = QVBoxLayout()
        col.addWidget(label(tr("Déposez votre fichier ici"), "H2"))
        col.addWidget(label(tr("WAV · FLAC · MP3 · OGG · OPUS · M4A · AAC · AIFF · WMA · "
                            "MP4 · MOV · MKV (piste audio)"), "Muted"))
        dl.addLayout(col)
        dl.addStretch()
        b = QPushButton(tr("Ouvrir un fichier…"), objectName="Primary")
        b.clicked.connect(self.browse)
        dl.addWidget(b)
        v.addWidget(drop)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.tiles = {
            "lufs": MetricTile(tr("Loudness intégré"), "LUFS"),
            "tp": MetricTile("True Peak", "dBTP"),
            "lra": MetricTile(tr("Loudness Range"), "LU"),
            "crest": MetricTile(tr("Facteur de crête"), "dB"),
            "corr": MetricTile(tr("Corrélation stéréo"), "-1 … +1"),
            "type": MetricTile(tr("Contenu détecté"), tr("analyse IA")),
        }
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, 0, i)
        v.addLayout(grid)
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.addWidget(label(tr("Spectre moyen"), "H2"))
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
        self._page_header(v, tr("Débruitage IA"),
                          tr("Supprime proprement le bruit derrière la voix : souffle, ventilation, "
                          "trafic, réverbération de bruit, ronflement secteur, clics. Le moteur "
                          "« Isolation voix » (Demucs) retire même la musique et les ambiances."),
                          "NEURAL")
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 16, 18, 16)
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(14)
        g.addWidget(label(tr("Moteur : IA Isolation voix — Demucs"), "H2"), 0, 0, 1, 2)
        self.dn_residual = LabeledSlider(tr("Fond conservé (isolation)"), -60, 0, -60, " dB")
        g.addWidget(self.dn_residual, 1, 0)
        g.addWidget(label(tr("Restauration"), "H2"), 2, 0, 1, 2)
        self.dn_rumble = QCheckBox(tr("Filtre anti-rumble (< 60 Hz)"))
        self.dn_rumble.setChecked(True)
        self.dn_hum = QCheckBox(tr("Anti-ronflement secteur"))
        self.dn_hum.setChecked(True)
        self.dn_humf = QComboBox()
        fill(self.dn_humf, ["Auto", "50 Hz", "60 Hz"])
        self.dn_click = QCheckBox(tr("Anti-clic / crépitements"))
        self.dn_deess = QCheckBox(tr("De-esser (sifflantes)"))
        g.addWidget(self.dn_rumble, 3, 0)
        hh = QHBoxLayout()
        hh.addWidget(self.dn_hum)
        hh.addWidget(self.dn_humf)
        hh.addStretch()
        g.addLayout(hh, 3, 1)
        g.addWidget(self.dn_click, 4, 0)
        g.addWidget(self.dn_deess, 4, 1)
        self.btn_denoise = QPushButton(tr("✦  Lancer le débruitage"), objectName="Primary")
        self.btn_denoise.clicked.connect(self.run_denoise)
        g.addWidget(self.btn_denoise, 5, 0, 1, 2)
        v.addWidget(c)
        v.addStretch()
        return w

    # ------------------------------------------------------------- Stems --
    def _page_stems(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        self._page_header(v, tr("Séparation de stems IA"),
                          tr("Demucs v4 : 8 stems rapides qui se somment exactement au mix. "
                          "MVSep Mega BS-RoFormer : jusqu'à 53 instruments (voix lead, chœurs, "
                          "grosse caisse, caisse claire, violon, violoncelle, trompette, saxophone, "
                          "orgue, harpe…). Deux modèles BS-RoFormer spécialisés voix : voix "
                          "homme / femme (aufr33), et voix / instrumental haute précision (viperx)."),
                          "8 · 53 · 2 · 1 STEMS")
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 14, 18, 14)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.addWidget(label(tr("Modèle"), "Muted"), 0, 0)
        self.st_model = QComboBox()
        model_pairs = [(stems.DEMUCS_LABEL, "demucs")] + \
            [(mega.model_label(k), k) for k in mega.MODEL_IDS]
        fill_pairs(self.st_model, model_pairs)
        g.addWidget(self.st_model, 0, 1)
        g.addWidget(label("Source", "Muted"), 0, 2)
        self.st_src = QComboBox()
        fill(self.st_src, ["Auto (dernière étape)", "Original", "Débruité"])
        g.addWidget(self.st_src, 0, 3)
        self.btn_stems = QPushButton(tr("✦  Séparer les stems"), objectName="Primary")
        self.btn_stems.clicked.connect(self.run_stems)
        g.addWidget(self.btn_stems, 0, 5)
        self.btn_stem_mix = QPushButton(tr("Écouter le mix des stems"))
        self.btn_stem_mix.clicked.connect(self.update_stem_mix)
        g.addWidget(self.btn_stem_mix, 0, 6)
        g.setColumnStretch(4, 1)

        # Options Demucs
        self.st_demucs_opts = QWidget()
        hd = QHBoxLayout(self.st_demucs_opts)
        hd.setContentsMargins(0, 0, 0, 0)
        hd.addWidget(label(tr("Qualité"), "Muted"))
        self.st_q = QComboBox()
        fill(self.st_q, list(stems.QUALITY))
        self.st_q.setCurrentIndex(1)
        hd.addWidget(self.st_q)
        hd.addSpacing(16)
        self.st_ext = QCheckBox(tr("Séparation étendue (cordes / nappes / synthés)"))
        self.st_ext.setChecked(True)
        hd.addWidget(self.st_ext)
        hd.addStretch()
        g.addWidget(self.st_demucs_opts, 1, 0, 1, 6)

        # Options MVSep Mega 53 stems
        self.st_mega_opts = QWidget()
        gm = QGridLayout(self.st_mega_opts)
        gm.setContentsMargins(0, 0, 0, 0)
        gm.setHorizontalSpacing(12)
        gm.addWidget(label(tr("Mémoire"), "Muted"), 0, 0)
        self.st_mem = QComboBox()
        fill(self.st_mem, list(mega.MEMORY_MODES))
        gm.addWidget(self.st_mem, 0, 1)
        self.st_hide = QCheckBox(tr("Masquer les instruments absents du morceau"))
        self.st_hide.setChecked(True)
        gm.addWidget(self.st_hide, 0, 2, 1, 2)
        gm.addWidget(label(tr("Fichier"), "Muted"), 1, 0)
        self.lbl_mega = label("", None)
        gm.addWidget(self.lbl_mega, 1, 1, 1, 1)
        self.btn_mega_dl = QPushButton(tr("⬇ Télécharger le modèle (1,4 Go)"))
        self.btn_mega_dl.clicked.connect(self.download_mega)
        self.btn_mega_pick = QPushButton(tr("Choisir un .ckpt…"))
        self.btn_mega_pick.clicked.connect(self.pick_mega)
        gm.addWidget(self.btn_mega_dl, 1, 2)
        gm.addWidget(self.btn_mega_pick, 1, 3)
        gm.setColumnStretch(1, 1)
        self.lbl_mega_note = label("", "Muted", True)
        gm.addWidget(self.lbl_mega_note, 2, 0, 1, 4)
        g.addWidget(self.st_mega_opts, 2, 0, 1, 6)
        self.st_model.currentIndexChanged.connect(self._stem_model_changed)
        self.roformer_ckpt = {}
        for mid in mega.MODEL_IDS:
            saved = self.settings.value(f"roformer_ckpt_{mid}", "", str) or None
            self.roformer_ckpt[mid] = mega.find_ckpt(mid, saved)
        self._stem_model_changed()
        self._update_mega_label()
        v.addWidget(c)

        self.stems_box = card()
        self.stems_lay = QVBoxLayout(self.stems_box)
        self.stems_lay.setContentsMargins(14, 12, 14, 12)
        self.stems_lay.setSpacing(8)
        self.stems_lay.addWidget(label(tr("Aucun stem pour l'instant."), "Muted"))
        v.addWidget(self.stems_box)
        v.addStretch()
        return w

    def _is_mega(self):
        return self.st_model.currentData() != "demucs"

    def _roformer_id(self):
        mid = self.st_model.currentData()
        return mid if mid in mega.MODEL_IDS else "mega53"

    def _stem_model_changed(self, *_):
        is_ro = self._is_mega()
        self.st_demucs_opts.setVisible(not is_ro)
        self.st_mega_opts.setVisible(is_ro)
        if is_ro:
            mid = self._roformer_id()
            info = mega.REGISTRY[mid]
            fill(self.st_mem, list(info["memory_modes"]))
            self.btn_mega_dl.setText(
                tr("⬇ Télécharger le modèle ({s:.2f} Go)").format(s=info["ckpt_size"] / 1e9))
            self.lbl_mega_note.setText(tr(info["note_fr"]))
        self._update_mega_label()

    def _update_mega_label(self):
        if not self._is_mega():
            return
        mid = self._roformer_id()
        ckpt = self.roformer_ckpt.get(mid)
        if ckpt:
            self.lbl_mega.setText("✔ " + ckpt)
            self.lbl_mega.setStyleSheet(f"color:{T.OK}; font-weight:600;")
            self.btn_mega_dl.setVisible(False)
        else:
            self.lbl_mega.setText(
                tr("Modèle absent — {f}").format(f=mega.REGISTRY[mid]["ckpt_name"]))
            self.lbl_mega.setStyleSheet(f"color:{T.WARN}; font-weight:600;")
            self.btn_mega_dl.setVisible(True)

    def pick_mega(self):
        mid = self._roformer_id()
        p, _ = QFileDialog.getOpenFileName(self, tr(mega.model_label(mid)), "",
                                           tr("Checkpoint (*.ckpt *.pth *.pt)"))
        if p:
            self.roformer_ckpt[mid] = p
            self.settings.setValue(f"roformer_ckpt_{mid}", p)
            self._update_mega_label()

    def download_mega(self):
        mid = self._roformer_id()

        def done(path):
            self.roformer_ckpt[mid] = path
            self.settings.setValue(f"roformer_ckpt_{mid}", path)
            self._update_mega_label()
        self._run(mega.download, done, mid,
                  title=tr("Téléchargement du modèle {m}").format(m=tr(mega.model_label(mid))))

    def _build_stem_rows(self):
        while self.stems_lay.count():
            it = self.stems_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.stem_ctrl = {}
        mid = getattr(self, "active_stem_model", "demucs")
        active = mega.default_active(self.stems, mid)
        is_mega = mega.is_mega_set(self.stems, mid)
        header_done = False
        if is_mega:
            self.stems_lay.addWidget(label(tr("STEMS PRINCIPAUX — se somment au mix d'origine"), "Muted"))
        for name, a in self.stems.items():
            if is_mega and not header_done and name not in active:
                hdr = label(tr("INSTRUMENTS DÉTAILLÉS ({n}) — se recouvrent avec les stems principaux, "
                               "en mute par défaut").format(n=len(self.stems) - len(active)), "Muted")
                hdr.setStyleSheet("margin-top:10px;")
                self.stems_lay.addWidget(hdr)
                header_done = True
            col = stems.stem_color(name, mid)
            row = QWidget()
            row.setFixedHeight(54)
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            bar = QLabel()
            bar.setFixedSize(5, 46)
            bar.setStyleSheet(f"background:{col}; border-radius:2px;")
            h.addWidget(bar)
            nm = QLabel(tr(stems.stem_label(name, mid)))
            nm.setFixedWidth(150)
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
            mute.setChecked(name not in active)
            gain = QSlider(Qt.Horizontal)
            gain.setRange(-240, 120)
            gain.setValue(0)
            gain.setFixedWidth(120)
            gl = QLabel("0.0 dB")
            gl.setFixedWidth(56)
            gain.valueChanged.connect(lambda val, l=gl: l.setText(f"{val / 10:+.1f} dB"))
            h.addWidget(gain)
            h.addWidget(gl)
            ex = QPushButton(tr("Exporter"))
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
        self._page_header(v, tr("Remastérisation IA"),
                          tr("Analyse intelligente → EQ corrective à phase linéaire, compression "
                          "multibande adaptative, excitateur harmonique, image stéréo M/S, glue, "
                          "normalisation loudness et limiteur true-peak x4."), "AI MASTER")
        row = QHBoxLayout()
        row.setSpacing(14)
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 16, 18, 16)
        g.setHorizontalSpacing(22)
        g.setVerticalSpacing(12)
        self.m_src = QComboBox()
        fill(self.m_src, ["Auto (dernière étape)", "Original", "Débruité", "Mix stems"])
        self.m_profile = QComboBox()
        fill(self.m_profile, list(mastering.PROFILES))
        self.m_loud = QComboBox()
        fill(self.m_loud, list(mastering.LOUDNESS_TARGETS) + ["Personnalisé…"])
        self.m_custom = QDoubleSpinBox()
        self.m_custom.setRange(-40, -5)
        self.m_custom.setValue(-14)
        self.m_custom.setSuffix(" LUFS")
        self.m_custom.setEnabled(False)
        self.m_loud.currentIndexChanged.connect(
            lambda _i: self.m_custom.setEnabled(self.m_loud.currentData() == "Personnalisé…"))
        self.m_ceiling = QDoubleSpinBox()
        self.m_ceiling.setRange(-6, 0)
        self.m_ceiling.setSingleStep(0.1)
        self.m_ceiling.setValue(-1.0)
        self.m_ceiling.setSuffix(" dBTP")
        self.m_char = QComboBox()
        fill(self.m_char, mastering.CHARACTERS)
        pairs = [(tr("Source"), self.m_src), (tr("Profil"), self.m_profile), (tr("Cible loudness"), self.m_loud),
                 (tr("LUFS perso"), self.m_custom), (tr("Plafond true-peak"), self.m_ceiling),
                 (tr("Caractère"), self.m_char)]
        for i, (t, wd) in enumerate(pairs):
            g.addWidget(label(t, "Muted"), i // 2 * 2, i % 2)
            g.addWidget(wd, i // 2 * 2 + 1, i % 2)
        self.m_int = LabeledSlider(tr("Correction EQ IA"), 0, 100, 70, " %")
        self.m_comp = LabeledSlider("Compression", 0, 100, 50, " %")
        self.m_exc = LabeledSlider(tr("Excitateur / Air"), 0, 100, 20, " %")
        self.m_width = LabeledSlider(tr("Largeur stéréo"), 50, 150, 100, " %")
        self.m_mono = LabeledSlider(tr("Basses mono sous"), 0, 250, 120, " Hz", 10)
        sl = [self.m_int, self.m_comp, self.m_exc, self.m_width, self.m_mono]
        for i, s in enumerate(sl):
            g.addWidget(s, 6 + i // 2, i % 2)
        rh = QHBoxLayout()
        self.btn_ref = QPushButton(tr("Charger une référence…"))
        self.btn_ref.setToolTip(tr("Mastering par référence : REMASTRA reproduit l'équilibre "
                                "tonal, le loudness et la largeur d'un morceau de référence."))
        self.btn_ref.clicked.connect(self.load_reference)
        self.lbl_ref = label(tr("Aucune référence"), "Muted")
        rh.addWidget(self.btn_ref)
        rh.addWidget(self.lbl_ref, 1)
        g.addLayout(rh, 9, 0, 1, 2)
        self.btn_master = QPushButton(tr("✦  Remastériser avec l'IA"), objectName="Primary")
        self.btn_master.clicked.connect(self.run_master)
        g.addWidget(self.btn_master, 10, 0, 1, 2)
        row.addWidget(c, 5)

        right = QVBoxLayout()
        tiles = QGridLayout()
        tiles.setSpacing(10)
        self.mt = {k: MetricTile(t, u) for k, t, u in [
            ("lufs", "Loudness", "LUFS"), ("tp", "True Peak", "dBTP"),
            ("lra", "LRA", "LU"), ("crest", tr("Crête"), "dB")]}
        for i, t in enumerate(self.mt.values()):
            tiles.addWidget(t, 0, i)
        right.addLayout(tiles)
        c3 = card()
        c3l = QVBoxLayout(c3)
        c3l.setContentsMargins(14, 12, 14, 12)
        c3l.addWidget(label(tr("Spectre avant / après"), "H2"))
        self.spec_master = SpectrumView()
        c3l.addWidget(self.spec_master)
        right.addWidget(c3)
        c4 = card()
        c4l = QVBoxLayout(c4)
        c4l.setContentsMargins(14, 12, 14, 12)
        c4l.addWidget(label(tr("Décisions du moteur IA"), "H2"))
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
                          tr("WAV (PCM 16/24 bits, 32 bits float, RF64 > 4 Go) ou FLAC sans perte "
                          "(16/24 bits). Du mono jusqu'au Dolby Atmos 9.1.6 et DTS:X avec "
                          "spatialisation IA, plus un repli Binaural pour l'écoute au casque."),
                          "LOSSLESS")
        row = QHBoxLayout()
        row.setSpacing(14)
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(18, 16, 18, 16)
        g.setVerticalSpacing(12)
        g.addWidget(label("Source", "Muted"), 0, 0)
        self.x_src = QComboBox()
        fill(self.x_src, ["Master (Remaster IA)", "Voix / audio débruité", "Original",
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
        g.addWidget(label(tr("Résolution"), "Muted"), 1, 2)
        self.x_bits = QComboBox()
        g.addWidget(self.x_bits, 1, 3)
        g.addWidget(label(tr("Fréquence"), "Muted"), 2, 0)
        self.x_sr = QComboBox()
        for s in export.SAMPLE_RATES:
            self.x_sr.addItem(f"{s / 1000:g} kHz", s)
        self.x_sr.setCurrentIndex(1)
        g.addWidget(self.x_sr, 2, 1)
        self.x_dither = QCheckBox(tr("Dither TPDF + noise shaping"))
        self.x_dither.setChecked(True)
        g.addWidget(self.x_dither, 2, 2, 1, 2)
        g.addWidget(label(tr("Canaux"), "H2"), 3, 0, 1, 4)
        lg = QGridLayout()
        lg.setSpacing(8)
        self.layout_grp = QButtonGroup(self)
        for i, name in enumerate(spatial.LAYOUTS):
            b = QPushButton(tr(name), objectName="LayoutBtn")
            b.setCheckable(True)
            b.setMinimumHeight(52)
            self.layout_grp.addButton(b, i)
            lg.addWidget(b, i // 4, i % 4)
        self.layout_grp.button(1).setChecked(True)
        self.layout_grp.idClicked.connect(self._layout_changed)
        g.addLayout(lg, 4, 0, 1, 4)
        self.lbl_layout = label("", "Muted", True)
        g.addWidget(self.lbl_layout, 5, 0, 1, 4)
        self.x_usestems = QCheckBox(tr("Spatialisation objet à partir des stems IA (si disponibles)"))
        self.x_usestems.setChecked(True)
        g.addWidget(self.x_usestems, 6, 0, 1, 4)
        self.btn_export = QPushButton(tr("⬇  Exporter"), objectName="Primary")
        self.btn_export.clicked.connect(self.run_export)
        g.addWidget(self.btn_export, 7, 0, 1, 4)
        row.addWidget(c, 3)
        c2 = card()
        c2l = QVBoxLayout(c2)
        c2l.setContentsMargins(14, 12, 14, 12)
        c2l.addWidget(label(tr("Configuration d'écoute"), "H2"))
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
        fill(self.x_bits, export.BIT_DEPTHS[fmt])
        self._layout_changed(self.layout_grp.checkedId())

    def _layout_changed(self, i):
        name = list(spatial.LAYOUTS)[max(0, i)]
        self.spk.set_layout(spatial.LAYOUTS[name])
        txt = tr(spatial.LAYOUT_DESC[name])
        if self.x_flac.isChecked() and len(spatial.LAYOUTS[name]) > 8:
            txt += tr("  —  FLAC est limité à 8 canaux : export multi-mono (1 FLAC par canal).")
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
            self.status(f"« {tr(stage)} » n'est pas encore disponible.")
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
        self.lbl_sel.setText(tr("Sélection : {a} → {b}").format(a=fmt_time(a), b=fmt_time(b)))

    def _clear_sel(self):
        self.wave.sel = None
        self.wave.update()
        self.lbl_sel.setText(tr("Astuce : cliquez pour vous déplacer, glissez pour sélectionner"))

    # ============================================================ Actions ==
    def status(self, msg):
        self.lbl_status.setText(tr_msg(msg))

    def log(self, msg):
        msg = tr_msg(msg)
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

    def _run(self, fn, on_done, *args, title=None, **kw):
        title = title or tr("Traitement")
        if self.worker and self.worker.isRunning():
            return
        if not self._engines_ready:
            # PyTorch est en cours d'import dans un autre thread : un import
            # concurrent casserait scipy/torch. On lance la tâche juste après.
            self._pending = (fn, on_done, args, dict(kw, title=title))
            self.status(tr("Initialisation des moteurs IA…"))
            return
        self.prog.setValue(0)
        self.status(f"{title}…")
        self.log(f"── {title} ──")
        self.worker = Worker(fn, *args, **kw)
        self.worker.progress.connect(lambda p: self.prog.setValue(int(p * 1000)))
        self.worker.log.connect(self.log)
        self.worker.done.connect(lambda r: (on_done(r), self.prog.setValue(1000),
                                            self.status(tr("{t} terminé ✔").format(t=title)),
                                            self._refresh_enabled()))
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._refresh_enabled)
        self.worker.start()
        self._refresh_enabled()

    def _failed(self, msg):
        self.prog.setValue(0)
        msg = tr_msg(msg.strip())
        self.status(tr("Erreur : ") + (msg.splitlines() or [""])[0])
        QMessageBox.critical(self, "REMASTRA", msg)
        self._refresh_enabled()

    def _probe_engines(self):
        def probe(progress, log):
            dev, vram = "cpu", 0
            try:
                import torch
                dev = stems.device_name()
                if dev == "cuda":
                    vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            except Exception:
                pass
            return {"demucs": stems.demucs_available(),
                    "mega": mega.available(), "dev": dev, "vram": vram}

        def done(r):
            ok = lambda b: f"<span style='color:{T.OK}'>●</span>" if b else f"<span style='color:{T.BAD}'>●</span>"  # noqa: E731
            ro_lines = "".join(
                f"{ok(r['mega'] and bool(self.roformer_ckpt.get(mid)))} {tr(mega.REGISTRY[mid]['label_fr'])}<br>"
                for mid in mega.MODEL_IDS)
            self.lbl_engines.setText(
                f"{ok(r['demucs'])} Demucs v4 (8 stems + isolation voix)<br>"
                f"{ro_lines}"
                f"<span style='color:{T.OK}'>●</span> Master DSP<br>"
                f"<span style='color:{T.MUTED}'>" + tr("Calcul : {d}").format(d=r['dev'].upper())
                + (f" · {r['vram']:.0f} {tr('Go')} VRAM" if r["vram"] else "") + "</span>")
            if r["vram"]:
                self.st_mem.setCurrentIndex(0 if r["vram"] >= 15 else 1 if r["vram"] >= 10 else 2)
            else:
                self.st_mem.setCurrentIndex(2)
        def ready():
            self._engines_ready = True
            if self._pending:
                fn, on_done, args, kw = self._pending
                self._pending = None
                self._run(fn, on_done, *args, **kw)

        w = Worker(probe)
        w.done.connect(done)
        w.finished.connect(ready)
        w.finished.connect(w.deleteLater)
        self._probe = w
        w.start()

    # ---------------------------------------------------------------- Open --
    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Ouvrir un fichier audio"), "", tr(audio_io.AUDIO_FILTER))
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
            self.lbl_fileinfo.setText(f"{sr / 1000:g} kHz · {a.shape[0]} {tr('canal(aux)')} · "
                                      f"{fmt_time(a.shape[1] / sr)}")
            self._show_report(rep)
            self.set_listen("Original")
            self._on_pos(0)
            self._clear_sel()
            if rep.content == "voix":
                select(self.m_profile, "Auto (IA)")

        self._run(job, done, title=tr_msg(f"Chargement de {os.path.basename(path)}"))

    def _build_stem_rows_empty(self):
        self.stems = {}
        self._build_stem_rows()
        self.stems_lay.addWidget(label(tr("Aucun stem pour l'instant."), "Muted"))

    def _show_report(self, rep: an.Report):
        t = self.tiles
        t["lufs"].set(f"{rep.lufs:.1f}", T.ACCENT3)
        t["tp"].set(f"{rep.true_peak:.1f}", T.BAD if rep.true_peak > -1 else T.OK)
        t["lra"].set(f"{rep.lra:.1f}")
        t["crest"].set(f"{rep.crest:.1f}")
        t["corr"].set(f"{rep.correlation:+.2f}", T.BAD if rep.correlation < 0 else T.TEXT)
        t["type"].set(tr(rep.content).capitalize(), T.ACCENT1)
        self.spec_import.set_data((rep.spectrum_f, rep.spectrum_db))

    def _source(self, choice: str):
        choice = choice or ""
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
            residual_db=self.dn_residual.value(),
            dehum=self.dn_hum.isChecked(), hum_freq=self.dn_humf.currentData(),
            declick=self.dn_click.isChecked(), rumble_hp=self.dn_rumble.isChecked(),
            deess=self.dn_deess.isChecked())
        src = self.audio["Original"]
        self._run(denoise.process, lambda y: self._set_stage("Débruité", y), src, self.sr, cfg,
                  title=tr("Débruitage"))

    # --------------------------------------------------------------- Stems --
    def run_stems(self):
        src = self._source(self.st_src.currentData())

        def done(d):
            self.stems = d
            self._build_stem_rows()
            self.update_stem_mix()

        if self._is_mega():
            mid = self._roformer_id()
            ckpt = self.roformer_ckpt.get(mid)
            if not ckpt:
                info = mega.REGISTRY[mid]
                r = QMessageBox.question(
                    self, "REMASTRA", tr("Le modèle {m} ({s:.2f} Go) n'est pas encore "
                    "installé.\n\nLe télécharger maintenant ?").format(
                        m=tr(info["label_fr"]), s=info["ckpt_size"] / 1e9))
                if r == QMessageBox.Yes:
                    self.download_mega()
                return
            self.active_stem_model = mid
            self._run(mega.separate, done, src, self.sr, ckpt, self.st_mem.currentData(),
                      self.st_hide.isChecked(), title=tr("Séparation ({m})").format(
                          m=tr(mega.model_label(mid))), model_id=mid)
            return
        self.active_stem_model = "demucs"
        self._run(stems.separate, done, src, self.sr, self.st_q.currentData(),
                  self.st_ext.isChecked(), title=tr("Séparation des stems"))

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

    def export_single_stem(self, name):
        p, _ = QFileDialog.getSaveFileName(self, tr("Exporter le stem"),
                                           self._default_name(f"_{name}"), "WAV (*.wav);;FLAC (*.flac)")
        if not p:
            return
        cfg = self._export_cfg()
        cfg.fmt = "FLAC" if p.lower().endswith(".flac") else "WAV"
        if cfg.bits not in export.BIT_DEPTHS[cfg.fmt]:
            cfg.bits = "24 bits"
        cfg.use_stems = False
        a = self.stems[name]
        mid = getattr(self, "active_stem_model", "demucs")
        self._run(lambda progress, log: export.export(p, a, self.sr, cfg, None, log),
                  lambda r: None, title=tr_msg(f"Export du stem {tr(stems.stem_label(name, mid))}"))

    # -------------------------------------------------------------- Master --
    def load_reference(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Morceau de référence"), "", tr(audio_io.AUDIO_FILTER))
        if not p:
            return
        try:
            self.reference = audio_io.load_audio(p)
            self.lbl_ref.setText(tr("Référence : ") + os.path.basename(p))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "REMASTRA", str(e))

    def run_master(self):
        src = self._source(self.m_src.currentData())
        loud = self.m_loud.currentData()
        cfg = mastering.MasterSettings(
            profile=self.m_profile.currentData(),
            loudness=loud if loud in mastering.LOUDNESS_TARGETS else "Profil (auto)",
            custom_lufs=self.m_custom.value() if loud == "Personnalisé…" else None,
            ceiling_dbtp=self.m_ceiling.value(), intensity=self.m_int.value(),
            compression=self.m_comp.value(), character=self.m_char.currentData(),
            exciter=self.m_exc.value(), width=self.m_width.value(),
            bass_mono_hz=self.m_mono.value(), reference=self.reference)

        def done(res: mastering.MasterResult):
            self.master_res = res
            self.decisions.clear()
            for d in res.decisions:
                self.decisions.addItem("✦  " + tr_msg(d))
            a, b = res.before, res.after
            self.mt["lufs"].set(f"{b.lufs:.1f}", T.ACCENT3, f"LUFS  ({tr('avant')} {a.lufs:.1f})")
            self.mt["tp"].set(f"{b.true_peak:.1f}", T.OK if b.true_peak <= cfg.ceiling_dbtp + 0.05 else T.BAD,
                              f"dBTP  ({tr('avant')} {a.true_peak:.1f})")
            self.mt["lra"].set(f"{b.lra:.1f}", None, f"LU  ({tr('avant')} {a.lra:.1f})")
            self.mt["crest"].set(f"{b.crest:.1f}", None, f"dB  ({tr('avant')} {a.crest:.1f})")
            self.spec_master.set_data((a.spectrum_f, a.spectrum_db), (b.spectrum_f, b.spectrum_db),
                                      res.eq_curve)
            self._set_stage("Master", res.audio)

        self._run(mastering.master, done, src, self.sr, cfg, title=tr("Remastérisation IA"))

    # -------------------------------------------------------------- Export --
    def _export_cfg(self):
        fmt = "WAV" if self.x_wav.isChecked() else "FLAC"
        return export.ExportSettings(
            fmt=fmt, bits=self.x_bits.currentData(), samplerate=self.x_sr.currentData(),
            layout=list(spatial.LAYOUTS)[self.layout_grp.checkedId()],
            use_stems=self.x_usestems.isChecked(), dither=self.x_dither.isChecked())

    def _default_name(self, suffix=""):
        base = os.path.splitext(os.path.basename(self.path or "remastra"))[0]
        d = os.path.dirname(self.path or "")
        return os.path.join(d, f"{base}{suffix}")

    def run_export(self):
        cfg = self._export_cfg()
        src = self.x_src.currentData()
        ext = ".wav" if cfg.fmt == "WAV" else ".flac"
        tag = cfg.layout.replace("é", "e").replace(".", "").replace(":", "-")
        if src.startswith("Stems séparés"):
            if not self.stems:
                QMessageBox.information(self, "REMASTRA", tr("Séparez d'abord les stems (étape ③)."))
                return
            folder = QFileDialog.getExistingDirectory(self, tr("Dossier de destination des stems"),
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
            self._run(job, lambda r: self._exported(r), title=tr("Export des stems"))
            return
        audio = {"Master": self.audio["Master"], "Voix": self.audio["Débruité"],
                 "Original": self.audio["Original"], "Mix": self.audio["Mix stems"]}[src.split()[0]]
        if audio is None:
            QMessageBox.information(self, "REMASTRA", tr("Cette source n'est pas encore disponible. "
                                    "Lancez l'étape correspondante ou choisissez une autre source."))
            return
        p, _ = QFileDialog.getSaveFileName(self, tr("Exporter"), self._default_name(f"_REMASTRA_{tag}") + ext,
                                           f"{cfg.fmt} (*{ext})")
        if not p:
            return
        st = dict(self.stems) if self.stems else None
        mid = getattr(self, "active_stem_model", "mega53")
        self._run(lambda progress, log: export.export(p, audio, self.sr, cfg, st, log, mid),
                  self._exported, title=f"Export {cfg.fmt} {tr(cfg.layout)}")

    def _exported(self, files):
        files = [f for f in files if not f.endswith(".txt")]
        QMessageBox.information(self, tr("REMASTRA — Export terminé"),
                                tr("{n} fichier(s) exporté(s) :").format(n=len(files)) + "\n\n" +
                                "\n".join(files[:12]) +
                                ("\n…" if len(files) > 12 else ""))

    def closeEvent(self, e):
        if self.player:
            self.player.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().closeEvent(e)

    def sizeHint(self):
        return QSize(1440, 900)
