"""Remastérisation — moteur adaptatif « IA Master ».

Chaîne (ordre studio) :
  1. Nettoyage DC + filtre subsonique 20 Hz
  2. Égalisation corrective à PHASE LINÉAIRE (FIR 8k taps) calculée par
     l'analyse : écart entre le spectre du morceau et la courbe cible du
     profil (ou d'un morceau de référence)
  3. Compression multibande (3 bandes, crossovers à reconstruction parfaite,
     seuils auto-adaptés à la dynamique de chaque bande)
  4. Excitateur harmonique (restauration des aigus) + saturation analogique
  5. Image stéréo M/S : basses en mono, largeur réglable
  6. Compresseur de bus « glue »
  7. Normalisation loudness BS.1770 + limiteur TRUE-PEAK à anticipation
     (détection suréchantillonnée x4), itératif jusqu'à la cible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

from . import analysis as an

PROFILES = {
    #  nom                       tilt  low  air  comp  lufs   width
    "Auto (IA)":                (-4.5, 0.0, 0.0, 0.5, -14.0, 1.00),
    "Pop / Variété":            (-4.3, 1.0, 1.0, 0.55, -14.0, 1.10),
    "Rock / Metal":             (-4.0, 0.5, 0.5, 0.6, -12.0, 1.05),
    "Hip-Hop / Trap":           (-5.0, 3.0, 0.5, 0.6, -11.0, 1.00),
    "Électro / EDM":            (-4.6, 2.5, 1.5, 0.7, -10.0, 1.15),
    "Jazz / Acoustique":        (-4.6, 0.0, 0.5, 0.3, -16.0, 1.00),
    "Classique / Orchestral":   (-4.8, 0.0, 0.0, 0.15, -20.0, 1.00),
    "Cinéma / Bande originale": (-4.6, 1.0, 0.5, 0.35, -24.0, 1.10),
    "Doublage / Dialogue":      (-3.2, -3.0, 0.0, 0.6, -23.0, 1.00),
    "Podcast / Voix":           (-3.0, -2.0, 0.5, 0.7, -16.0, 1.00),
}
LOUDNESS_TARGETS = {
    "Profil (auto)": None,
    "Streaming — Spotify/YouTube (-14 LUFS)": -14.0,
    "Apple Music (-16 LUFS)": -16.0,
    "CD / Club (-9 LUFS)": -9.0,
    "Broadcast EBU R128 (-23 LUFS)": -23.0,
    "Cinéma / ATSC A/85 (-24 LKFS)": -24.0,
    "Podcast (-16 LUFS)": -16.0,
    "Sans normalisation": 0.0,
}
CHARACTERS = ["Neutre", "Chaleureux", "Brillant", "Vintage analogique", "Punchy"]


@dataclass
class MasterSettings:
    profile: str = "Auto (IA)"
    loudness: str = "Profil (auto)"
    custom_lufs: float | None = None
    ceiling_dbtp: float = -1.0
    intensity: float = 70.0        # correction EQ 0..100
    compression: float = 50.0      # 0..100
    character: str = "Neutre"
    exciter: float = 20.0          # 0..100
    width: float = 100.0           # %
    bass_mono_hz: float = 120.0
    reference: tuple[np.ndarray, int] | None = None


@dataclass
class MasterResult:
    audio: np.ndarray
    before: an.Report
    after: an.Report
    decisions: list[str] = field(default_factory=list)
    eq_curve: tuple[np.ndarray, np.ndarray] | None = None


def _noop(*_a, **_k):
    pass


# --------------------------------------------------------------------------- #
#  Utilitaires
# --------------------------------------------------------------------------- #
def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def _octave_smooth(f, db, frac=1.0):
    lf = np.log2(f)
    out = np.empty_like(db)
    for i, c in enumerate(lf):
        m = np.abs(lf - c) <= frac / 2
        out[i] = np.mean(db[m])
    return out


def target_curve(f, tilt, low, air):
    t = tilt * np.log2(f / 1000.0)
    t += low / (1 + (f / 90.0) ** 2)              # plateau grave
    t += air / (1 + (9000.0 / f) ** 2)            # « air »
    return t


def linear_phase_eq(x, sr, freqs, gains_db, taps=None):
    taps = taps or (8191 if sr <= 48000 else 16383)
    nyq = sr / 2
    grid = np.concatenate([[0], freqs[freqs < nyq], [nyq]])
    g = np.concatenate([[gains_db[0]], gains_db[freqs < nyq], [gains_db[-1] - 3]])
    h = signal.firwin2(taps, grid / nyq, 10 ** (g / 20), window="blackmanharris")
    y = signal.fftconvolve(x, h[None, :], mode="full", axes=-1)
    d = (taps - 1) // 2
    return y[:, d: d + x.shape[1]].astype(np.float32)


def _split3(x, sr, f1=120.0, f2=5000.0):
    s1 = signal.butter(4, f1, "lowpass", fs=sr, output="sos")
    s2 = signal.butter(4, f2, "lowpass", fs=sr, output="sos")
    lp1 = signal.sosfiltfilt(s1, x, axis=-1)
    lp2 = signal.sosfiltfilt(s2, x, axis=-1)
    return lp1, lp2 - lp1, x - lp2


def _loud_level(x, sr):
    """Niveau RMS 400 ms au 90e percentile (dB) — sert à caler les seuils."""
    blk = int(0.4 * sr)
    if x.shape[1] < blk:
        return _db(np.sqrt(np.mean(x ** 2)))
    p = an._block_power(x, sr, 0.4, 0.5).mean(axis=1)
    return float(10 * np.log10(np.percentile(p, 90) + 1e-12))


def compressor(x, sr, threshold_db, ratio, attack_ms, release_ms, knee_db=6.0, makeup_db=0.0):
    try:
        import pedalboard as pb

        board = pb.Pedalboard([pb.Compressor(threshold_db=threshold_db, ratio=ratio,
                                             attack_ms=attack_ms, release_ms=release_ms),
                               pb.Gain(gain_db=makeup_db)])
        return board(x.astype(np.float32), sr)
    except Exception:
        # Repli : compresseur RMS à genou souple (détection liée)
        env = np.sqrt(signal.lfilter([1 - np.exp(-1 / (release_ms * 1e-3 * sr))],
                                     [1, -np.exp(-1 / (release_ms * 1e-3 * sr))],
                                     (x ** 2).max(axis=0)))
        lvl = _db(env)
        over = lvl - threshold_db
        gr = np.where(over <= -knee_db / 2, 0.0,
                      np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                               (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
        g = 10 ** ((makeup_db - gr) / 20)
        return (x * g[None]).astype(np.float32)


def saturate(x, drive=0.3, asym=0.0):
    """Saturation douce suréchantillonnée x2 (anti-aliasing)."""
    if drive <= 0:
        return x
    up = signal.resample_poly(x, 2, 1, axis=-1)
    k = 1 + 4 * drive
    y = np.tanh(k * (up + asym)) - np.tanh(k * asym)
    y /= np.tanh(k)
    y = signal.resample_poly(y, 1, 2, axis=-1)[:, : x.shape[1]]
    return y.astype(np.float32)


def exciter(x, sr, amount):
    if amount <= 0:
        return x
    sos = signal.butter(2, 3000, "highpass", fs=sr, output="sos")
    hi = signal.sosfiltfilt(sos, x, axis=-1)
    harm = saturate(hi * 3, 0.8, 0.1) / 3
    sos2 = signal.butter(4, 6000, "highpass", fs=sr, output="sos")
    harm = signal.sosfiltfilt(sos2, harm, axis=-1)
    return (x + harm * (amount / 100) * 0.8).astype(np.float32)


def stereo_image(x, sr, width=1.0, mono_below=120.0):
    if x.shape[0] != 2:
        return x
    M = (x[0] + x[1]) / 2
    S = (x[0] - x[1]) / 2
    if mono_below > 0:
        sos = signal.butter(4, mono_below, "highpass", fs=sr, output="sos")
        S = signal.sosfiltfilt(sos, S)
    S = S * width
    return np.stack([M + S, M - S]).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Limiteur true-peak
# --------------------------------------------------------------------------- #
def true_peak_limiter(x, sr, ceiling_db=-1.0, lookahead_ms=1.5, release_ms=80.0):
    ceil = 10 ** (ceiling_db / 20)
    n = x.shape[1]
    # enveloppe de crête inter-échantillons (x4)
    env = np.zeros(n, dtype=np.float32)
    step = sr * 20
    for s in range(0, n, step):
        seg = x[:, s: s + step]
        up = np.abs(signal.resample_poly(seg, 4, 1, axis=1)).max(axis=0)
        m = seg.shape[1]
        env[s: s + m] = up[: m * 4].reshape(m, 4).max(axis=1)
    g_raw = np.minimum(1.0, ceil / np.maximum(env, 1e-9))
    L = max(1, int(lookahead_ms * 1e-3 * sr))
    g1 = ndimage.minimum_filter1d(g_raw, 2 * L + 1)
    g1 = ndimage.uniform_filter1d(g1, L | 1)
    g1 = np.minimum(g1, g_raw)
    # relâchement exponentiel à cadence de contrôle (blocs de 32)
    B = 32
    nb = int(np.ceil(n / B))
    pad = np.pad(g1, (0, nb * B - n), constant_values=1.0)
    gb_db = _db(pad.reshape(nb, B).min(axis=1))
    a = np.exp(-B / (release_ms * 1e-3 * sr))
    env_db = np.empty(nb)
    cur = 0.0
    for i in range(nb):
        v = gb_db[i]
        cur = v if v < cur else cur * a
        env_db[i] = cur
    centers = np.arange(nb) * B + B / 2
    g_rel = 10 ** (np.interp(np.arange(n), centers, env_db) / 20)
    g = np.minimum(g_rel, g1)
    y = x * g[None, :].astype(np.float32)
    return np.clip(y, -ceil, ceil).astype(np.float32), float(max(0.0, -_db(g.min())))


def normalize_and_limit(x, sr, target_lufs, ceiling_db, log=_noop):
    if target_lufs == 0.0:  # pas de normalisation : limiteur seul
        y, gr = true_peak_limiter(x, sr, ceiling_db)
        return y
    gain = target_lufs - an.integrated_loudness(x, sr)
    out, gr = x, 0.0
    for _ in range(4):
        out, gr = true_peak_limiter(x * 10 ** (gain / 20), sr, ceiling_db)
        tp = an.true_peak_db(out, sr)
        if tp > ceiling_db:  # sécurité inter-échantillons
            out, gr2 = true_peak_limiter(out, sr, ceiling_db - (tp - ceiling_db) - 0.05)
            gr = max(gr, gr2)
        err = target_lufs - an.integrated_loudness(out, sr)
        if abs(err) < 0.15:
            break
        gain += err
    log(f"Limiteur true-peak : gain {gain:+.1f} dB, réduction max {gr:.1f} dB")
    return out


# --------------------------------------------------------------------------- #
#  « Cerveau » IA : analyse -> décisions
# --------------------------------------------------------------------------- #
def auto_profile(x, sr, rep: an.Report) -> str:
    if rep.content == "voix":
        return "Doublage / Dialogue" if rep.lra > 8 or rep.lufs < -24 else "Podcast / Voix"
    f, db = rep.spectrum_f, rep.spectrum_db
    p = 10 ** (db / 10)
    total = p.sum()
    sub = p[f < 90].sum() / total
    air = p[f > 8000].sum() / total
    if rep.crest > 20 and sub < 0.25:
        return "Classique / Orchestral"
    if sub > 0.55:
        return "Hip-Hop / Trap" if air < 0.004 else "Électro / EDM"
    if rep.crest < 13 and air > 0.002:
        return "Rock / Metal"
    if rep.crest > 17:
        return "Jazz / Acoustique"
    return "Pop / Variété"


def master(x: np.ndarray, sr: int, cfg: MasterSettings, progress=_noop, log=_noop) -> MasterResult:
    decisions: list[str] = []
    say = lambda s: (decisions.append(s), log(s))  # noqa: E731

    log("Analyse du signal source…")
    before = an.analyse(x, sr)
    progress(0.08)
    profile = cfg.profile
    if profile == "Auto (IA)":
        profile = auto_profile(x, sr, before)
        say(f"Contenu détecté : {before.content} → profil « {profile} »")
    tilt, low, air, comp_k, lufs_def, width_def = PROFILES[profile]

    target = LOUDNESS_TARGETS.get(cfg.loudness)
    if cfg.custom_lufs is not None:
        target = cfg.custom_lufs
    ref_rep = None
    if cfg.reference is not None:
        rx, rsr = cfg.reference
        ref_rep = an.analyse(rx, rsr)
        say(f"Référence : {ref_rep.lufs:.1f} LUFS, corrélation {ref_rep.correlation:+.2f}")
        if target is None:
            target = round(ref_rep.lufs, 1)
    if target is None:
        target = lufs_def
    say(f"Cible loudness : {'aucune' if target == 0 else f'{target:.1f} LUFS'}, "
        f"plafond {cfg.ceiling_dbtp:.1f} dBTP")

    y = x.astype(np.float32)
    # 1. DC + subsonique
    y = y - y.mean(axis=1, keepdims=True)
    y = signal.sosfiltfilt(signal.butter(2, 20, "highpass", fs=sr, output="sos"), y, axis=-1)
    progress(0.12)

    # 2. EQ corrective à phase linéaire
    f, s_db = before.spectrum_f, before.spectrum_db
    if ref_rep is not None:
        t_db = np.interp(np.log(f), np.log(ref_rep.spectrum_f), ref_rep.spectrum_db)
        t_db = _octave_smooth(f, t_db, 1 / 2)
    else:
        t_db = target_curve(f, tilt, low, air)
    s_sm = _octave_smooth(f, s_db, 1 / 2)
    wmask = (f > 80) & (f < 10000)
    t_db = t_db - np.mean(t_db[wmask]) + np.mean(s_sm[wmask])
    corr = _octave_smooth(f, t_db - s_sm, 2 / 3)
    k = cfg.intensity / 100.0
    lim = 3 + 5 * k
    corr = np.clip(corr * k * 0.8, -lim, lim)
    # Jamais de boost là où il n'y a pas de contenu (subsonique, bandes vides,
    # grave des voix) : on ne remonte pas du bruit.
    empty = s_sm < np.mean(s_sm[wmask]) - 35
    no_boost = (f < 40) | empty | ((f < 80) & (profile in ("Doublage / Dialogue", "Podcast / Voix")))
    corr[no_boost] = np.minimum(corr[no_boost], 0)
    char = cfg.character
    if char == "Chaleureux":
        corr += 1.2 / (1 + (f / 250) ** 2) - 0.8 / (1 + (6000 / f) ** 2)
    elif char == "Brillant":
        corr += 1.8 / (1 + (8000 / f) ** 2)
    elif char == "Vintage analogique":
        corr += 0.8 / (1 + ((f - 300) / 300) ** 2) - 2.5 / (1 + (15000 / f) ** 4)
    elif char == "Punchy":
        corr += 1.5 * np.exp(-0.5 * (np.log2(f / 70) / 0.6) ** 2) + \
            1.0 * np.exp(-0.5 * (np.log2(f / 3500) / 0.8) ** 2)
    corr -= np.mean(corr[wmask]) * 0.5
    y = linear_phase_eq(y, sr, f, corr)
    top = np.argsort(np.abs(corr))[-3:]
    say("EQ phase linéaire : " + ", ".join(
        f"{corr[i]:+.1f} dB @ {f[i]:.0f} Hz" for i in sorted(top)))
    progress(0.35)

    # 3. Compression multibande adaptative
    c = cfg.compression / 100.0 * (0.5 + comp_k)
    if before.crest < 11:
        c *= 0.5
        say("Source déjà dense : compression allégée")
    if c > 0.02:
        bands = _split3(y, sr)
        params = [(2.0 + 2 * c, 30, 180), (1.5 + 1.5 * c, 12, 120), (1.5 + 1.5 * c, 5, 80)]
        out = np.zeros_like(y)
        for b, (ratio, att, rel), name in zip(bands, params, ["grave", "médium", "aigu"]):
            lvl = _loud_level(b, sr)
            thr = lvl - (1 + 6 * c)
            out += compressor(b.astype(np.float32), sr, thr, ratio, att, rel)
        y = out.astype(np.float32)
        say(f"Compression multibande : intensité {c * 100:.0f}% (3 bandes 120 Hz / 5 kHz)")
    progress(0.5)

    # 4. Excitateur + saturation
    exc = cfg.exciter
    if before.spectrum_db.size:
        hf = np.mean(before.spectrum_db[f > 10000]) - np.mean(before.spectrum_db[(f > 1000) & (f < 4000)])
        if hf < -30 and cfg.exciter > 0:
            exc = min(100, exc + 30)
            say("Aigus pauvres détectés (source ancienne ?) : excitateur renforcé")
    y = exciter(y, sr, exc)
    if char in ("Chaleureux", "Vintage analogique"):
        pk = np.max(np.abs(y)) + 1e-9
        y = saturate(y / pk, 0.25 if char == "Chaleureux" else 0.4, 0.05) * pk
        say(f"Saturation analogique « {char} »")
    progress(0.6)

    # 5. Stéréo
    width = cfg.width / 100.0 * width_def
    if y.shape[0] == 2:
        if ref_rep is not None:
            width *= float(np.clip((1 - ref_rep.correlation) / max(0.05, 1 - before.correlation), 0.7, 1.4))
        if before.correlation > 0.95 and cfg.width == 100:
            say("Image très étroite : pas d'élargissement artificiel (quasi-mono)")
            width = 1.0
        y = stereo_image(y, sr, width, cfg.bass_mono_hz)
        say(f"Stéréo M/S : largeur {width * 100:.0f}%, basses mono < {cfg.bass_mono_hz:.0f} Hz")
    progress(0.68)

    # 6. Glue
    if c > 0.02:
        lvl = _loud_level(y, sr)
        y = compressor(y, sr, lvl - 2 - 3 * c, 2.0, 30, 250)
        say("Compresseur de bus « glue » 2:1 (attaque 30 ms)")
    progress(0.78)

    # 7. Loudness + limiteur
    y = normalize_and_limit(y, sr, target, cfg.ceiling_dbtp, log)
    progress(0.95)
    after = an.analyse(y, sr)
    say(f"Résultat : {after.lufs:.1f} LUFS, {after.true_peak:.1f} dBTP, LRA {after.lra:.1f} LU")
    progress(1.0)
    return MasterResult(y, before, after, decisions, (f, corr))
