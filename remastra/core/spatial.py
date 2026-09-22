"""Formats de canaux et upmix spatial IA (mono -> 9.1.6).

Deux moteurs d'upmix :
  * Stems (objet) : chaque stem séparé par l'IA est placé comme un objet
    dans l'espace (voix au centre, pads/cordes en ambiance et hauteurs,
    basse au LFE…). C'est le rendu le plus propre.
  * Spectral : décomposition directe/ambiance (Avendano-Jot) du stéréo
    dans le domaine fréquentiel, centre extrait par cohérence inter-canal,
    ambiance décorrélée vers les surrounds et hauteurs.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

# Ordre des canaux : norme WAVE (bits de dwChannelMask) jusqu'au 7.1,
# puis ordre Dolby Atmos / SMPTE ST 2098 pour les lits immersifs.
LAYOUTS: dict[str, list[str]] = {
    "Mono": ["C"],
    "Stéréo": ["L", "R"],
    "2.0": ["L", "R"],
    "2.1": ["L", "R", "LFE"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "7.1": ["L", "R", "C", "LFE", "Lrs", "Rrs", "Lss", "Rss"],
    "7.1.6": ["L", "R", "C", "LFE", "Lrs", "Rrs", "Lss", "Rss",
              "Ltf", "Rtf", "Ltm", "Rtm", "Ltr", "Rtr"],
    "9.1.6": ["L", "R", "C", "LFE", "Lrs", "Rrs", "Lss", "Rss", "Lw", "Rw",
              "Ltf", "Rtf", "Ltm", "Rtm", "Ltr", "Rtr"],
}
LAYOUT_DESC = {
    "Mono": "1 canal — C",
    "Stéréo": "2 canaux — master stéréo L/R",
    "2.0": "2 canaux — L/R discrets (broadcast, pistes séparées compatibles)",
    "2.1": "3 canaux — L R + LFE (caisson)",
    "5.1": "6 canaux — L R C LFE Ls Rs",
    "7.1": "8 canaux — L R C LFE Lrs Rrs Lss Rss",
    "7.1.6": "14 canaux — 7.1 + 6 hauteurs (Dolby Atmos bed)",
    "9.1.6": "16 canaux — 7.1.6 + Wides (Dolby Atmos bed étendu)",
}

# dwChannelMask WAVE_FORMAT_EXTENSIBLE
_SPK = {"L": 0x1, "R": 0x2, "C": 0x4, "LFE": 0x8, "Lrs": 0x10, "Rrs": 0x20,
        "Lss": 0x200, "Rss": 0x400, "Ls": 0x10, "Rs": 0x20,
        "Ltf": 0x1000, "Rtf": 0x4000, "Ltr": 0x8000, "Rtr": 0x20000}


def channel_mask(layout: str) -> int:
    if layout == "Mono":
        return 0x4
    if layout in ("Stéréo", "2.0"):
        return 0x3
    if layout == "2.1":
        return 0x3 | 0x8
    if layout == "5.1":
        return 0x3F      # FL FR FC LFE BL BR (5.1 WAVE standard)
    if layout == "7.1":
        return 0x63F     # FL FR FC LFE BL BR SL SR
    return 0             # 7.1.6 / 9.1.6 : hauteurs médianes hors masque -> ordre documenté


def _lp(x, sr, fc, order=4):
    return signal.sosfiltfilt(signal.butter(order, fc, "lowpass", fs=sr, output="sos"), x, axis=-1)


def _hp(x, sr, fc, order=4):
    return signal.sosfiltfilt(signal.butter(order, fc, "highpass", fs=sr, output="sos"), x, axis=-1)


def _decorrelate(x, sr, seed=0, delay_ms=12.0):
    """Décorrélation par filtre passe-tout à phase aléatoire + léger retard."""
    rng = np.random.default_rng(seed)
    n = 1024
    ph = rng.uniform(-np.pi, np.pi, n // 2 + 1)
    ph[0] = ph[-1] = 0
    h = np.fft.irfft(np.exp(1j * ph), n) * np.hanning(n)
    h /= np.sqrt(np.sum(h ** 2)) + 1e-12
    d = int(delay_ms * 1e-3 * sr)
    h = np.concatenate([np.zeros(d), h])
    y = signal.fftconvolve(x, h, mode="full")[: len(x)]
    return y.astype(np.float32)


def _primary_ambient(st: np.ndarray, sr: int):
    """Extraction centre / direct / ambiance (masques de cohérence STFT)."""
    nfft, hop = 2048, 512
    _, _, Z = signal.stft(st, sr, nperseg=nfft, noverlap=nfft - hop)
    L, R = Z[0], Z[1]
    from scipy.ndimage import uniform_filter

    def sm(a):
        return uniform_filter(a.real, (1, 7)) + 1j * uniform_filter(a.imag, (1, 7)) \
            if np.iscomplexobj(a) else uniform_filter(a, (1, 7))
    pLL, pRR = sm(np.abs(L) ** 2), sm(np.abs(R) ** 2)
    pLR = sm(L * np.conj(R))
    coh = np.abs(pLR) / np.sqrt(pLL * pRR + 1e-20)            # 0..1
    sim = 2 * np.abs(pLR) / (pLL + pRR + 1e-20)                # similarité panoramique
    c_mask = np.clip(sim, 0, 1) ** 2 * coh
    amb_mask = np.sqrt(np.clip(1 - coh, 0, 1))
    Cz = c_mask * (L + R) / 2
    Lz, Rz = L - Cz, R - Cz
    La, Ra = amb_mask * Lz, amb_mask * Rz
    Ld, Rd = Lz - La, Rz - Ra

    def inv(z):
        _, y = signal.istft(z, sr, nperseg=nfft, noverlap=nfft - hop)
        y = y[: st.shape[1]]
        return np.pad(y, (0, st.shape[1] - len(y))).astype(np.float32)
    return inv(Cz), inv(Ld), inv(Rd), inv(La), inv(Ra)


def _render(layout, sr, L, R, C, amb_l, amb_r, lfe_src, height_l, height_r, wide_l=None, wide_r=None):
    n = len(L)
    names = LAYOUTS[layout]
    ch = {k: np.zeros(n, np.float32) for k in names}
    lfe = _lp(lfe_src, sr, 120.0, 8) * 0.7
    if layout == "Mono":
        ch["C"] = (L + R) * 0.5 + C + (amb_l + amb_r) * 0.35
    elif layout in ("Stéréo", "2.0", "2.1"):
        ch["L"] = L + C * 0.7071 + amb_l
        ch["R"] = R + C * 0.7071 + amb_r
        if layout == "2.1":
            ch["L"] = _hp(ch["L"], sr, 80, 2)
            ch["R"] = _hp(ch["R"], sr, 80, 2)
            ch["LFE"] = lfe
    else:
        ch["L"], ch["R"], ch["C"], ch["LFE"] = L, R, C, lfe
        if layout == "5.1":
            ch["Ls"], ch["Rs"] = amb_l * 0.85, amb_r * 0.85
        else:
            ch["Lss"], ch["Rss"] = amb_l * 0.7, amb_r * 0.7
            ch["Lrs"] = _decorrelate(amb_l, sr, 11, 18) * 0.6
            ch["Rrs"] = _decorrelate(amb_r, sr, 12, 19) * 0.6
        if layout in ("7.1.6", "9.1.6"):
            hl, hr = _hp(height_l, sr, 250, 2), _hp(height_r, sr, 250, 2)
            ch["Ltf"], ch["Rtf"] = hl * 0.5, hr * 0.5
            ch["Ltm"] = _decorrelate(hl, sr, 21, 7) * 0.4
            ch["Rtm"] = _decorrelate(hr, sr, 22, 8) * 0.4
            ch["Ltr"] = _decorrelate(hl, sr, 23, 15) * 0.4
            ch["Rtr"] = _decorrelate(hr, sr, 24, 16) * 0.4
        if layout == "9.1.6":
            wl = wide_l if wide_l is not None else (L + amb_l) * 0.5
            wr = wide_r if wide_r is not None else (R + amb_r) * 0.5
            ch["Lw"], ch["Rw"] = wl * 0.6, wr * 0.6
    return np.stack([ch[k] for k in names]).astype(np.float32)


def upmix(audio: np.ndarray, sr: int, layout: str, stems: dict | None = None) -> np.ndarray:
    """Convertit un master (mono/stéréo) vers le format demandé."""
    from .audio_io import to_stereo

    if layout == "Mono":
        return audio.mean(axis=0, keepdims=True).astype(np.float32) if audio.shape[0] > 1 else audio
    if layout in ("Stéréo", "2.0"):
        return to_stereo(audio)
    st = to_stereo(audio)
    n = st.shape[1]

    if stems:  # --- upmix objet à partir des stems -------------------------
        z = np.zeros(n, np.float32)
        def g(k):
            s = stems.get(k)
            if s is None:
                return z, z
            s = to_stereo(s)[:, :n]
            if s.shape[1] < n:
                s = np.pad(s, ((0, 0), (0, n - s.shape[1])))
            return s[0], s[1]
        vl, vr = g("vocals"); dl, dr = g("drums"); bl, br = g("bass")
        gl, gr = g("guitar"); pl, pr = g("piano"); sl, sr_ = g("strings")
        al, ar = g("pads"); yl, yr = g("synth"); ol, or_ = g("other")
        # placement objet (le niveau est recalé sur le master à l'export)
        L = dl + bl * 0.6 + gl * 0.8 + pl * 0.8 + sl * 0.5 + yl * 0.6 + ol * 0.7
        R = dr + br * 0.6 + gr * 0.8 + pr * 0.8 + sr_ * 0.5 + yr * 0.6 + or_ * 0.7
        C = (vl + vr) / 2 * 1.0 + (bl + br) / 2 * 0.4
        vs = (vl - vr) / 2
        L, R = L + vs, R - vs
        amb_l = sl * 0.6 + al * 0.8 + gl * 0.2 + pl * 0.2 + yl * 0.4 + ol * 0.3
        amb_r = sr_ * 0.6 + ar * 0.8 + gr * 0.2 + pr * 0.2 + yr * 0.4 + or_ * 0.3
        amb_l = _decorrelate(amb_l, sr, 1, 5); amb_r = _decorrelate(amb_r, sr, 2, 6)
        height_l, height_r = al * 0.7 + sl * 0.5 + dl * 0.15, ar * 0.7 + sr_ * 0.5 + dr * 0.15
        lfe_src = (bl + br) / 2 + (dl + dr) / 2 * 0.5
        wide_l, wide_r = gl * 0.4 + yl * 0.4 + sl * 0.3, gr * 0.4 + yr * 0.4 + sr_ * 0.3
        out = _render(layout, sr, L, R, C, amb_l, amb_r, lfe_src, height_l, height_r, wide_l, wide_r)
    else:  # --- upmix spectral ----------------------------------------------
        C, Ld, Rd, La, Ra = _primary_ambient(st, sr)
        amb_l = _decorrelate(La, sr, 3, 10)
        amb_r = _decorrelate(Ra, sr, 4, 11)
        out = _render(layout, sr, Ld + La * 0.3, Rd + Ra * 0.3, C, amb_l, amb_r,
                      st.mean(axis=0), La + Ld * 0.1, Ra + Rd * 0.1)

    # Préservation du niveau : ramène la crête au niveau du master
    ref_pk = float(np.max(np.abs(st))) + 1e-9
    pk = float(np.max(np.abs(out))) + 1e-9
    if pk > ref_pk:
        out *= ref_pk / pk
    return out.astype(np.float32)
