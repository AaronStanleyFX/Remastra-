"""Analyse audio : loudness ITU-R BS.1770-4 / EBU R128, true peak, LRA,
corrélation stéréo, spectre moyen, détection de contenu."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------- #
#  K-weighting (BS.1770)
# --------------------------------------------------------------------------- #
def _biquad_high_shelf(fs, fc=1681.974450955533, gain_db=3.999843853973347, q=0.7071752369554196):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha),
         -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha,
         2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_highpass(fs, fc=38.13547087602444, q=0.5003270373238773):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
    b1, a1 = _biquad_high_shelf(sr)
    b2, a2 = _biquad_highpass(sr)
    y = signal.lfilter(b1, a1, audio, axis=-1)
    return signal.lfilter(b2, a2, y, axis=-1)


def _channel_weights(n_ch: int) -> np.ndarray:
    w = np.ones(n_ch)
    if n_ch >= 4:
        w[3] = 0.0  # LFE exclu
    if n_ch >= 6:
        w[4:] = 1.41  # surrounds (+1.5 dB)
    return w


def _block_power(kw: np.ndarray, sr: int, block_s: float, overlap: float) -> np.ndarray:
    blk = int(block_s * sr)
    hop = max(1, int(blk * (1 - overlap)))
    n = kw.shape[1]
    if n < blk:
        return np.array([np.mean(kw ** 2, axis=1)]) if n else np.zeros((0, kw.shape[0]))
    sq = kw.astype(np.float64) ** 2
    cs = np.concatenate([np.zeros((kw.shape[0], 1)), np.cumsum(sq, axis=1)], axis=1)
    starts = np.arange(0, n - blk + 1, hop)
    return ((cs[:, starts + blk] - cs[:, starts]) / blk).T  # (blocks, ch)


def integrated_loudness(audio: np.ndarray, sr: int) -> float:
    if audio.shape[1] < int(0.4 * sr):
        return -70.0
    kw = k_weight(audio, sr)
    pw = _block_power(kw, sr, 0.4, 0.75)
    w = _channel_weights(audio.shape[0])
    z = pw @ w
    lk = -0.691 + 10 * np.log10(z + 1e-12)
    g = z[lk > -70]
    if g.size == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g2 = z[(lk > -70) & (lk > rel)]
    if g2.size == 0:
        return -70.0
    return float(-0.691 + 10 * np.log10(np.mean(g2)))


def loudness_range(audio: np.ndarray, sr: int) -> float:
    if audio.shape[1] < 3 * sr:
        return 0.0
    kw = k_weight(audio, sr)
    pw = _block_power(kw, sr, 3.0, 2 / 3)
    z = pw @ _channel_weights(audio.shape[0])
    lk = -0.691 + 10 * np.log10(z + 1e-12)
    g = z[lk > -70]
    if g.size < 2:
        return 0.0
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 20
    sel = lk[(lk > -70) & (lk > rel)]
    if sel.size < 2:
        return 0.0
    return float(np.percentile(sel, 95) - np.percentile(sel, 10))


def short_term_curve(audio: np.ndarray, sr: int, hop_s: float = 0.5) -> np.ndarray:
    kw = k_weight(audio, sr)
    blk = int(3 * sr)
    if audio.shape[1] < blk:
        return np.array([integrated_loudness(audio, sr)])
    pw = _block_power(kw, sr, 3.0, 1 - hop_s / 3.0)
    z = pw @ _channel_weights(audio.shape[0])
    return -0.691 + 10 * np.log10(z + 1e-12)


def true_peak_db(audio: np.ndarray, sr: int) -> float:
    """True peak (dBTP) par suréchantillonnage x4 (BS.1770-4 annexe 2)."""
    if audio.size == 0:
        return -120.0
    factor = 4 if sr < 96000 else 2
    peak = 0.0
    step = sr * 20  # blocs de 20 s pour limiter la mémoire
    for s in range(0, audio.shape[1], step):
        seg = audio[:, max(0, s - 64): s + step + 64]
        up = signal.resample_poly(seg, factor, 1, axis=1)
        peak = max(peak, float(np.max(np.abs(up))))
    return float(20 * np.log10(peak + 1e-12))


def sample_peak_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.max(np.abs(audio)) + 1e-12)) if audio.size else -120.0


def stereo_correlation(audio: np.ndarray) -> float:
    if audio.shape[0] < 2:
        return 1.0
    L, R = audio[0].astype(np.float64), audio[1].astype(np.float64)
    den = np.sqrt(np.sum(L * L) * np.sum(R * R)) + 1e-12
    return float(np.sum(L * R) / den)


def average_spectrum(audio: np.ndarray, sr: int, n_bands: int = 96,
                     fmin: float = 20.0, fmax: float = 20000.0):
    """Spectre moyen (dB) sur bandes logarithmiques -> (freqs, db)."""
    mono = audio.mean(axis=0)
    if mono.size > sr * 120:  # sous-échantillonnage temporel pour la vitesse
        idx = np.linspace(0, mono.size - sr * 2, 60).astype(int)
        mono = np.concatenate([mono[i:i + sr * 2] for i in idx])
    nper = 8192
    if mono.size < nper:
        mono = np.pad(mono, (0, nper - mono.size))
    f, p = signal.welch(mono, sr, nperseg=nper, noverlap=nper // 2, window="hann")
    fmax = min(fmax, sr / 2 * 0.98)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    out = np.empty(n_bands)
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out[i] = np.mean(p[m]) if np.any(m) else np.interp(centers[i], f, p)
    return centers, 10 * np.log10(out + 1e-20)


@dataclass
class Report:
    lufs: float = -70.0
    lra: float = 0.0
    true_peak: float = -120.0
    sample_peak: float = -120.0
    crest: float = 0.0
    correlation: float = 1.0
    duration: float = 0.0
    channels: int = 0
    samplerate: int = 0
    content: str = "musique"
    spectrum_f: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spectrum_db: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> str:
        return (f"{self.lufs:.1f} LUFS  |  LRA {self.lra:.1f} LU  |  "
                f"TP {self.true_peak:.1f} dBTP  |  Crest {self.crest:.1f} dB  |  "
                f"Corr {self.correlation:+.2f}")


def detect_content(audio: np.ndarray, sr: int) -> str:
    """Classification heuristique : 'voix' (parole/doublage) ou 'musique'."""
    mono = audio.mean(axis=0)
    if mono.size < sr:
        return "musique"
    f, t, Z = signal.stft(mono[: sr * 60], sr, nperseg=2048, noverlap=1024)
    mag = np.abs(Z) + 1e-10
    e = mag.sum(axis=0)
    low = mag[f < 90].sum(axis=0)
    bass_ratio = float(np.median(low / e))
    # modulation syllabique (~4 Hz) typique de la parole
    env = 20 * np.log10(e)
    pauses = float(np.mean(env < (np.max(env) - 35)))
    flux = float(np.std(np.diff(env)))
    speech_score = (pauses > 0.08) * 1 + (bass_ratio < 0.03) * 1 + (flux > 3.0) * 1
    return "voix" if speech_score >= 2 else "musique"


def analyse(audio: np.ndarray, sr: int, spectrum: bool = True) -> Report:
    r = Report()
    r.channels, r.samplerate = audio.shape[0], sr
    r.duration = audio.shape[1] / sr
    r.lufs = integrated_loudness(audio, sr)
    r.lra = loudness_range(audio, sr)
    r.true_peak = true_peak_db(audio, sr)
    r.sample_peak = sample_peak_db(audio)
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    r.crest = r.sample_peak - 20 * np.log10(rms)
    r.correlation = stereo_correlation(audio)
    r.content = detect_content(audio, sr)
    if spectrum:
        r.spectrum_f, r.spectrum_db = average_spectrum(audio, sr)
    return r
