"""Séparation de sources (stems).

Étape 1 — IA Demucs v4 « htdemucs_6s » (Meta AI, Hybrid Transformer) :
          voix, batterie, basse, guitare, piano, autres.
Étape 2 — Séparation étendue REMASTRA : la piste « autres » est décomposée
          en cordes, nappes (pads) et synthés par masques temps-fréquence
          (harmonique/percussif, persistance temporelle, timbre).
Les 8 stems se somment exactement au mixage d'origine.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .audio_io import resample, to_stereo

STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "strings", "pads", "synth"]
STEM_LABELS = {
    "vocals": "Voix", "drums": "Batterie", "bass": "Basse", "guitar": "Guitare",
    "piano": "Piano", "strings": "Cordes", "pads": "Nappes / Pads", "synth": "Synthés",
    "other": "Autres", "background": "Fond",
}
STEM_COLORS = {
    "vocals": "#FF4FD8", "drums": "#FFB020", "bass": "#7C5CFF", "guitar": "#FF6B4A",
    "piano": "#35E0A1", "strings": "#FFD85C", "pads": "#4FC3FF", "synth": "#B8FF4F",
    "other": "#9AA4B2",
}
QUALITY = {"Rapide": 0, "Haute qualité": 1, "Ultra (lent)": 4}
MODELS = ["Demucs v4 — 8 stems (rapide)", "MVSep Mega BS-RoFormer — 53 stems"]


def stem_label(name: str) -> str:
    if name in STEM_LABELS:
        return STEM_LABELS[name]
    from . import mega
    return mega.label(name)


def stem_color(name: str) -> str:
    if name in STEM_COLORS:
        return STEM_COLORS[name]
    from . import mega
    return mega.color(name)

_MODELS: dict = {}


def _noop(*_a, **_k):
    pass


def demucs_available() -> bool:
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_model(name: str, log=_noop):
    if name not in _MODELS:
        from demucs.pretrained import get_model

        log(f"Chargement du modèle IA {name} (téléchargé au premier lancement)…")
        m = get_model(name)
        m.eval()
        _MODELS[name] = m
    return _MODELS[name]


class _ProgressTqdm:
    """Remplace tqdm dans demucs.apply pour récupérer la progression."""

    def __init__(self, cb):
        self.cb = cb

    def tqdm(self, iterable, *a, **k):
        items = list(iterable)
        n = max(1, len(items))
        for i, it in enumerate(items):
            yield it
            self.cb((i + 1) / n)


def run_demucs(audio: np.ndarray, sr: int, model_name: str = "htdemucs_6s",
               shifts: int = 1, progress=_noop, log=_noop) -> dict[str, np.ndarray]:
    import demucs.apply as dapply
    import torch

    model = _get_model(model_name, log)
    msr = model.samplerate
    x = resample(to_stereo(audio), sr, msr)
    wav = torch.from_numpy(np.ascontiguousarray(x))
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    wav = (wav - mu) / sd
    dev = device_name()
    log(f"Inférence Demucs sur {dev.upper()} (shifts={shifts})…")
    old = dapply.tqdm
    dapply.tqdm = _ProgressTqdm(progress)
    try:
        with torch.no_grad():
            src = dapply.apply_model(model, wav[None], device=dev, shifts=shifts,
                                     split=True, overlap=0.25, progress=True)[0]
    finally:
        dapply.tqdm = old
    src = (src * sd + mu).cpu().numpy()
    out = {}
    for name, s in zip(model.sources, src):
        out[name] = resample(s.astype(np.float32), msr, sr)[:, : audio.shape[1]]
    return out


def separate_vocals(audio: np.ndarray, sr: int, progress=_noop, log=_noop):
    parts = run_demucs(audio, sr, "htdemucs", 1, progress, log)
    voc = parts["vocals"]
    if audio.shape[0] == 1:
        voc = voc.mean(axis=0, keepdims=True)
    n = min(voc.shape[1], audio.shape[1])
    voc = voc[:, :n]
    return {"vocals": voc, "background": audio[:, :n] - voc}


# --------------------------------------------------------------------------- #
#  Séparation étendue : autres -> cordes / nappes / synthés
# --------------------------------------------------------------------------- #
def split_other(other: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    nfft, hop = 4096, 1024
    win = signal.windows.hann(nfft, sym=False)
    f, _, Z = signal.stft(other, sr, window=win, nperseg=nfft, noverlap=nfft - hop)
    M = np.abs(Z).mean(axis=0) + 1e-9                     # (bins, frames)
    fr = sr / hop
    H = ndimage.median_filter(M, size=(1, max(3, int(0.25 * fr) | 1)))
    P = ndimage.median_filter(M, size=(17, 1))
    harm = H ** 2 / (H ** 2 + P ** 2 + 1e-18)             # 1 = tonal
    # Persistance : les nappes tiennent de longues notes à attaque lente
    L = ndimage.minimum_filter1d(ndimage.uniform_filter1d(M, int(1.5 * fr) + 1, axis=1),
                                 int(1.0 * fr) + 1, axis=1)
    sustain = np.clip(L / M, 0, 1) ** 1.2
    # Pondération timbrale des cordes (180 Hz – 5 kHz, cloche log)
    lf = np.log2(np.maximum(f, 20) / 950.0)
    w_band = np.exp(-0.5 * (lf / 1.35) ** 2)[:, None]
    # Brillance / attaque rapide -> synthés
    flux = np.maximum(np.diff(M, axis=1, prepend=M[:, :1]), 0) / M
    bright = np.clip((f[:, None] - 2500) / 6000, 0, 1)

    w_pad = harm * sustain * (0.6 + 0.4 * (1 - bright))
    w_str = harm * (1 - sustain) * w_band * (1 - 0.5 * np.clip(flux, 0, 1))
    w_syn = (1 - harm) * 0.6 + harm * (1 - sustain) * (1 - w_band) + 0.4 * bright * harm
    W = np.stack([w_str, w_pad, w_syn]) ** 2 + 1e-12      # masques de Wiener
    W = ndimage.uniform_filter(W, size=(1, 3, 3))
    W /= W.sum(axis=0, keepdims=True)
    out = {}
    for name, w in zip(["strings", "pads", "synth"], W):
        _, y = signal.istft(Z * w[None], sr, window=win, nperseg=nfft, noverlap=nfft - hop)
        out[name] = y[:, : other.shape[1]].astype(np.float32)
        if out[name].shape[1] < other.shape[1]:
            out[name] = np.pad(out[name], ((0, 0), (0, other.shape[1] - out[name].shape[1])))
    # correction résiduelle : somme exacte = autres
    resid = other - sum(out.values())
    out["synth"] += resid
    return out


def separate(audio: np.ndarray, sr: int, quality: str = "Haute qualité",
             extended: bool = True, progress=_noop, log=_noop) -> dict[str, np.ndarray]:
    """Retourne un dict de stems (stéréo, même longueur/sr que l'entrée)."""
    if not demucs_available():
        raise RuntimeError(
            "Le moteur IA Demucs n'est pas installé.\n"
            "Lancez install.bat (ou : pip install torch demucs).")
    shifts = QUALITY.get(quality, 1)
    parts = run_demucs(audio, sr, "htdemucs_6s", shifts,
                       lambda p: progress(0.9 * p), log)
    if extended:
        log("Séparation étendue : cordes / nappes / synthés…")
        parts.update(split_other(parts.pop("other"), sr))
    progress(1.0)
    order = [k for k in STEM_ORDER if k in parts] + [k for k in parts if k not in STEM_ORDER]
    return {k: parts[k] for k in order}
