"""Thème visuel REMASTRA — « Neon Studio »."""

BG = "#07080F"
BG2 = "#0D0F1C"
PANEL = "#121528"
PANEL2 = "#181C34"
BORDER = "#252A4A"
TEXT = "#E8EAF6"
MUTED = "#8A90B4"
ACCENT1 = "#FF3CAC"   # magenta
ACCENT2 = "#784BA0"   # violet
ACCENT3 = "#2BD9FE"   # cyan
OK = "#35E0A1"
WARN = "#FFB020"
BAD = "#FF5470"

QSS = f"""
* {{
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QWidget#Sidebar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {BG2}, stop:1 #0A0B16);
    border-right: 1px solid {BORDER};
}}
QWidget#TopBar {{
    background: {BG2};
    border-bottom: 1px solid {BORDER};
}}
QWidget#BottomBar {{ background: {BG2}; border-top: 1px solid {BORDER}; }}
QFrame#Card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
}}
QLabel#H1 {{ font-size: 22px; font-weight: 700; }}
QLabel#H2 {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Tag {{
    color: {ACCENT3}; background: rgba(43,217,254,0.08);
    border: 1px solid rgba(43,217,254,0.35); border-radius: 9px; padding: 2px 8px;
    font-size: 11px; font-weight: 600;
}}

QPushButton {{
    background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 8px 14px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT3}; background: #1E2342; }}
QPushButton:pressed {{ background: #0F1224; }}
QPushButton:disabled {{ color: #4A5070; border-color: #1C2038; }}
QPushButton#Primary {{
    border: none; color: white; font-size: 14px; padding: 11px 22px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT1}, stop:0.55 {ACCENT2}, stop:1 #2B86C5);
}}
QPushButton#Primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #FF5CBC, stop:0.55 #8F5EC0, stop:1 {ACCENT3});
}}
QPushButton#Primary:disabled {{ background: #2A2E4A; color: #6A7090; }}
QPushButton#Nav {{
    text-align: left; padding: 12px 16px; border: none; border-radius: 12px;
    background: transparent; color: {MUTED}; font-size: 14px; font-weight: 600;
}}
QPushButton#Nav:hover {{ background: rgba(255,255,255,0.04); color: {TEXT}; }}
QPushButton#Nav:checked {{
    color: white;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(255,60,172,0.28), stop:1 rgba(43,217,254,0.06));
    border-left: 3px solid {ACCENT1};
}}
QPushButton#Transport {{
    min-width: 38px; max-width: 38px; min-height: 38px; max-height: 38px;
    border-radius: 19px; padding: 0; font-size: 15px;
}}
QPushButton#LayoutBtn {{
    padding: 10px 6px; border-radius: 12px; font-size: 15px; font-weight: 700;
}}
QPushButton#LayoutBtn:checked {{
    border: 2px solid {ACCENT1}; color: white;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(255,60,172,0.25), stop:1 rgba(120,75,160,0.15));
}}
QPushButton#Chip {{ padding: 4px 10px; border-radius: 8px; font-size: 11px; }}
QPushButton#Chip:checked {{ background: {ACCENT2}; border-color: {ACCENT1}; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 10px; min-height: 20px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {ACCENT2}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT2};
    outline: none; padding: 4px;
}}
QSlider::groove:horizontal {{ height: 6px; background: #1E2240; border-radius: 3px; }}
QSlider::sub-page:horizontal {{
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT1}, stop:1 {ACCENT3});
}}
QSlider::handle:horizontal {{
    background: white; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
    border: 3px solid {ACCENT1};
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px; border: 1px solid {BORDER}; background: {BG2};
}}
QCheckBox::indicator:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT1}, stop:1 {ACCENT3});
    border: none;
}}
QRadioButton::indicator {{ width: 16px; height: 16px; border-radius: 8px; border: 1px solid {BORDER}; background:{BG2}; }}
QRadioButton::indicator:checked {{ background: {ACCENT1}; border: 3px solid {BG2}; }}
QProgressBar {{
    background: #151936; border: none; border-radius: 5px; height: 10px; text-align: center;
    font-size: 10px; color: transparent;
}}
QProgressBar::chunk {{
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT1}, stop:0.5 {ACCENT2}, stop:1 {ACCENT3});
}}
QPlainTextEdit, QListWidget {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 10px;
    font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; color: #B8BEE0;
}}
QListWidget::item {{ padding: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2A3058; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {ACCENT2}; padding: 6px; }}
QSplitter::handle {{ background: {BORDER}; }}
"""
