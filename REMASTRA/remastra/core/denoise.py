"""Débruitage professionnel.

Moteur :
  * « IA Isolation voix — Demucs » : extrait la voix et supprime tout ce qui
    est derrière (musique, ambiance, foule) avec un niveau résiduel réglable.

Modules de restauration : anti-ronflement (50/60 Hz + harmoniques),
anti-clic, filtre anti-rumble, de-esser.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

ENGINES = [
    "IA Isolation voix — Demucs",
]


@dataclass
class DenoiseSettings:
    engine: str = ENGINES[0]
    residual_db: float = -60.0      # isolation voix : niveau du fond conservé
    dehum: bool = True
    hum_freq: str = "Auto"          # Auto / 50 Hz / 60 Hz
    declick: bool = False
    rumble_hp: bool = True
    deess: bool = False
    deess_amount: float = 50.0


def _noop(*_a, **_k):
    pass


# --------------------------------------------------------------------------- #
#  Restauration
# --------------------------------------------------------------------------- #
def rumble_filter(x: np.ndarray, sr: int, fc: float = 60.0) -> np.ndarray:
    sos = signal.butter(4, fc, "highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def detect_hum(x: np.ndarray, sr: int) -> float | None:
    mono = x.mean(axis=0)[: sr * 30]
    f, p = signal.welch(mono, sr, nperseg=min(len(mono), sr * 2))
    def peak(fr):
        m = (f > fr - 1.5) & (f < fr + 1.5)
        ref = (f > fr - 12) & (f < fr + 12) & ~m
        if not np.any(m) or not np.any(ref):
            return 0.0
        return 10 * np.log10(np.max(p[m]) / (np.median(p[ref]) + 1e-20) + 1e-20)
    s50 = peak(50) + peak(100) + peak(150)
    s60 = peak(60) + peak(120) + peak(180)
    best, score = (50.0, s50) if s50 >= s60 else (60.0, s60)
    return best if score > 15 else None


def dehum(x: np.ndarray, sr: int, base: float, harmonics: int = 8) -> np.ndarray:
    y = x.astype(np.float64)
    for k in range(1, harmonics + 1):
        f0 = base * k
        if f0 >= sr / 2 - 100:
            break
        b, a = signal.iirnotch(f0, Q=35 + 5 * k, fs=sr)
        y = signal.filtfilt(b, a, y, axis=-1)
    return y.astype(np.float32)


def declick(x: np.ndarray, sr: int, sensitivity: float = 6.0) -> np.ndarray:
    y = x.copy()
    sos = signal.butter(2, 3000, "highpass", fs=sr, output="sos")
    for c in range(x.shape[0]):
        hp = signal.sosfilt(sos, x[c])
        d = np.abs(np.diff(hp, n=2, prepend=hp[:2]))
        mad = np.median(d) + 1e-9
        mask = d > sensitivity * 8 * mad
        if not np.any(mask):
            continue
        mask = ndimage.binary_dilation(mask, iterations=max(1, sr // 16000))
        med = signal.medfilt(x[c], 9)
        y[c, mask] = med[mask]
    return y


def deesser(x: np.ndarray, sr: int, amount: float = 50.0) -> np.ndarray:
    """De-esser dynamique à bande partagée (5–10 kHz)."""
    if sr < 22050:
        return x
    hi = min(10000, sr / 2 * 0.9)
    sos = signal.butter(4, [5000, hi], "bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, x, axis=-1)
    rest = x - band
    env_b = np.abs(band).max(axis=0)
    env_f = np.abs(x).max(axis=0)
    a = np.exp(-1.0 / (0.005 * sr))
    env_b = signal.lfilter([1 - a], [1, -a], env_b)
    env_f = signal.lfilter([1 - a], [1, -a], env_f) + 1e-9
    ratio = env_b / env_f
    thr = 0.45 - 0.3 * (amount / 100)
    gain = np.where(ratio > thr, (thr / ratio) ** (0.5 + amount / 100), 1.0)
    gain = ndimage.uniform_filter1d(gain, int(0.002 * sr) + 1)
    return (rest + band * gain).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def process(x: np.ndarray, sr: int, cfg: DenoiseSettings, progress=_noop, log=_noop) -> np.ndarray:
    y = x.astype(np.float32).copy()
    if cfg.rumble_hp:
        log("Filtre anti-rumble (HPF 60 Hz, 24 dB/oct)")
        y = rumble_filter(y, sr, 60.0)
    if cfg.dehum:
        base = {"50 Hz": 50.0, "60 Hz": 60.0}.get(cfg.hum_freq) or detect_hum(y, sr)
        if base:
            log(f"Anti-ronflement {base:.0f} Hz + 8 harmoniques")
            y = dehum(y, sr, base)
        else:
            log("Aucun ronflement secteur détecté")
    if cfg.declick:
        log("Anti-clic")
        y = declick(y, sr)
    progress(0.1)

    sub = lambda p: progress(0.1 + 0.8 * p)  # noqa: E731
    from . import stems as stems_mod

    if not stems_mod.demucs_available():
        raise RuntimeError(
            "Le moteur IA Demucs n'est pas installé.\n"
            "Lancez install.bat (ou : pip install torch demucs).")
    log("Demucs : isolation de la voix…")
    parts = stems_mod.separate_vocals(y, sr, progress=sub, log=log)
    res = 10 ** (cfg.residual_db / 20) if cfg.residual_db > -59 else 0.0
    y = parts["vocals"] + res * parts["background"]

    if cfg.deess:
        log("De-esser")
        y = deesser(y, sr, cfg.deess_amount)
    progress(1.0)
    return y.astype(np.float32)
