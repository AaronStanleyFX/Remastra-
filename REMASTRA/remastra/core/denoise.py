"""Débruitage professionnel.

Moteurs :
  * « IA Voix — DeepFilterNet 3 » : réseau de neurones de rehaussement de la
    parole (48 kHz, temps réel), idéal doublage / dialogues.
  * « IA Isolation voix — Demucs » : extrait la voix et supprime tout ce qui
    est derrière (musique, ambiance, foule) avec un niveau résiduel réglable.
  * « Spectral Pro » : estimateur MMSE à SNR a priori « decision-directed »
    (Ephraim-Malah) avec suivi adaptatif du bruit — sans IA, très rapide.

Modules de restauration : anti-ronflement (50/60 Hz + harmoniques),
anti-clic, filtre anti-rumble, de-esser.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal
from scipy.special import exp1

from .audio_io import resample

ENGINES = [
    "IA Voix — DeepFilterNet 3",
    "IA Isolation voix — Demucs",
    "Spectral Pro (MMSE)",
]


@dataclass
class DenoiseSettings:
    engine: str = ENGINES[0]
    strength: float = 80.0          # 0..100 %
    max_reduction_db: float = 30.0  # plancher de réduction
    residual_db: float = -60.0      # isolation voix : niveau du fond conservé
    dehum: bool = True
    hum_freq: str = "Auto"          # Auto / 50 Hz / 60 Hz
    declick: bool = False
    rumble_hp: bool = True
    deess: bool = False
    deess_amount: float = 50.0
    noise_profile: tuple[float, float] | None = None  # secondes (début, fin)


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
#  Spectral Pro (MMSE / decision-directed)
# --------------------------------------------------------------------------- #
def spectral_denoise(x: np.ndarray, sr: int, strength: float = 80.0,
                     max_reduction_db: float = 30.0,
                     profile: tuple[float, float] | None = None,
                     progress=_noop) -> np.ndarray:
    nfft = 4096 if sr > 60000 else 2048
    hop = nfft // 4
    win = signal.windows.hann(nfft, sym=False)
    _, _, Z = signal.stft(x, sr, window=win, nperseg=nfft, noverlap=nfft - hop,
                          boundary="even", padded=True)
    P = (np.abs(Z) ** 2).astype(np.float64)          # (ch, bins, frames)
    n_frames = P.shape[-1]

    # --- Profil de bruit --------------------------------------------------- #
    if profile is not None:
        a, b = int(profile[0] * sr / hop), int(profile[1] * sr / hop)
        a, b = max(0, a), min(n_frames, max(a + 4, b))
        N = P[..., a:b].mean(axis=-1)
    else:
        sm = ndimage.uniform_filter1d(P, size=max(3, int(0.3 * sr / hop)), axis=-1)
        N = np.percentile(sm, 8, axis=-1) * 1.6
    N = np.maximum(N, 1e-14)

    s = strength / 100.0
    over = 1.0 + 1.5 * s                              # sur-soustraction
    floor = 10 ** (-max_reduction_db * (0.3 + 0.7 * s) / 20)
    alpha = 0.98
    G = np.ones_like(P)
    g_prev = np.ones(P.shape[:2])
    gam_prev = np.ones(P.shape[:2])
    Nt = N.copy()
    report = max(1, n_frames // 50)
    for t in range(n_frames):
        pt = P[..., t]
        gamma = pt / (over * Nt)
        xi = alpha * (g_prev ** 2) * gam_prev + (1 - alpha) * np.maximum(gamma - 1, 0)
        # Estimateur MMSE log-spectral (Ephraim-Malah 1985)
        xi = np.maximum(xi, floor ** 2)
        v = np.minimum(xi / (1 + xi) * gamma, 50.0)
        g = xi / (1 + xi) * np.exp(0.5 * exp1(np.maximum(v, 1e-8)))   # MMSE-LSA
        g = np.clip(g, floor, 1.0)
        G[..., t] = g
        # suivi adaptatif du bruit sur les trames « bruit seul »
        quiet = gamma < 1.2
        Nt = np.where(quiet, 0.995 * Nt + 0.005 * pt, Nt)
        Nt = np.minimum(np.maximum(Nt, N * 0.5), N * 4)
        g_prev, gam_prev = g, gamma
        if t % report == 0:
            progress(t / n_frames)
    # Probabilité de présence du signal utile (décision douce) : supprime le
    # « bruit musical » dans les silences sans toucher aux transitoires.
    snr_loc = 10 * np.log10(ndimage.uniform_filter(P, size=(1, 5, 5)) / (Nt[..., None] + 1e-20) + 1e-12)
    pres = 1 / (1 + np.exp(-(snr_loc - (3 + 3 * s)) / 1.5))
    G = np.exp(pres * np.log(G) + (1 - pres) * np.log(floor))
    G = ndimage.uniform_filter(G, size=(1, 3, 3))
    _, y = signal.istft(Z * G, sr, window=win, nperseg=nfft, noverlap=nfft - hop,
                        boundary=True)
    return y[:, : x.shape[1]].astype(np.float32)


# --------------------------------------------------------------------------- #
#  DeepFilterNet 3
# --------------------------------------------------------------------------- #
_DF = None


def _shim_torchaudio():
    """deepfilternet 0.5.x importe torchaudio.backend.common (retiré de torchaudio>=2.1)."""
    try:
        import torchaudio.backend.common  # noqa: F401
    except Exception:
        mod = types.ModuleType("torchaudio.backend.common")

        class AudioMetaData:  # minimal
            def __init__(self, *a, **k):
                pass

        mod.AudioMetaData = AudioMetaData
        sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
        sys.modules["torchaudio.backend.common"] = mod


def deepfilter_available() -> bool:
    try:
        _shim_torchaudio()
        import df.enhance  # noqa: F401
        return True
    except Exception:
        return False


def deepfilter_denoise(x: np.ndarray, sr: int, strength: float = 80.0,
                       progress=_noop, log=_noop) -> np.ndarray:
    global _DF
    _shim_torchaudio()
    import torch
    from df.enhance import enhance, init_df

    if _DF is None:
        log("Chargement du modèle DeepFilterNet 3…")
        _DF = init_df(log_level="ERROR")
    model, st, _ = _DF
    dsr = st.sr()
    y = resample(x, sr, dsr)
    lim = None if strength >= 99 else 6 + strength * 0.5  # dB max d'atténuation
    chunk, ov = dsr * 30, dsr // 2
    out = np.zeros_like(y)
    fade = np.linspace(0, 1, ov, dtype=np.float32)
    total = y.shape[0] * max(1, int(np.ceil(y.shape[1] / chunk)))
    done = 0
    for c in range(y.shape[0]):
        pos = 0
        while pos < y.shape[1]:
            a, b = max(0, pos - ov), min(y.shape[1], pos + chunk)
            seg = torch.from_numpy(np.ascontiguousarray(y[c:c + 1, a:b]))
            with torch.no_grad():
                enh = enhance(model, st, seg, atten_lim_db=lim).numpy()[0]
            if a > 0 and pos - a == ov:
                enh[:ov] *= fade
                out[c, a:pos] *= fade[::-1]
            out[c, a:b] += enh[: b - a]
            pos += chunk
            done += 1
            progress(done / total)
    return resample(out, dsr, sr)


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
    eng = cfg.engine
    if eng.startswith("IA Voix"):
        if deepfilter_available():
            log("DeepFilterNet 3 : rehaussement neuronal de la voix…")
            try:
                y = deepfilter_denoise(y, sr, cfg.strength, sub, log)
            except (Exception, SystemExit) as e:  # modèle non téléchargeable, etc.
                log(f"⚠ DeepFilterNet indisponible ({e}) → bascule sur Spectral Pro")
                eng = ENGINES[2]
        else:
            log("⚠ DeepFilterNet non installé → bascule sur Spectral Pro")
            eng = ENGINES[2]
    elif eng.startswith("IA Isolation"):
        from . import stems as stems_mod

        if stems_mod.demucs_available():
            log("Demucs : isolation de la voix…")
            try:
                parts = stems_mod.separate_vocals(y, sr, progress=sub, log=log)
                res = 10 ** (cfg.residual_db / 20) if cfg.residual_db > -59 else 0.0
                y = parts["vocals"] + res * parts["background"]
            except (Exception, SystemExit) as e:
                log(f"⚠ Demucs indisponible ({e}) → bascule sur Spectral Pro")
                eng = ENGINES[2]
        else:
            log("⚠ Demucs non installé → bascule sur Spectral Pro")
            eng = ENGINES[2]
    if eng.startswith("Spectral"):
        log("Spectral Pro : estimation MMSE du bruit…")
        y = spectral_denoise(y, sr, cfg.strength, cfg.max_reduction_db, cfg.noise_profile, sub)

    if cfg.deess:
        log("De-esser")
        y = deesser(y, sr, cfg.deess_amount)
    progress(1.0)
    return y.astype(np.float32)
