"""Export WAV / FLAC multicanal.

* WAV : WAVE_FORMAT_EXTENSIBLE avec dwChannelMask, 16 / 24 bits PCM ou
  32 bits float, RF64 automatique au-delà de 4 Go (EBU Tech 3306).
* FLAC : 16 / 24 bits, jusqu'à 8 canaux (limite de la norme FLAC).
  Pour 7.1.6 et 9.1.6 (14/16 canaux), export « multi-mono » : un FLAC par
  canal dans un dossier, nommé selon la norme (L, R, C, LFE, Ltf…).
Dither TPDF avec mise en forme du bruit légère lors de la réduction à 16/24 bits.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from . import spatial
from .audio_io import resample

FORMATS = ["WAV", "FLAC"]
BIT_DEPTHS = {"WAV": ["24 bits", "32 bits float", "16 bits"], "FLAC": ["24 bits", "16 bits"]}
SAMPLE_RATES = [44100, 48000, 88200, 96000, 192000]

_KSDATAFORMAT_PCM = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_KSDATAFORMAT_FLOAT = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


@dataclass
class ExportSettings:
    fmt: str = "WAV"
    bits: str = "24 bits"
    samplerate: int = 48000
    layout: str = "Stéréo"
    use_stems: bool = True
    dither: bool = True


def _noop(*_a, **_k):
    pass


def dither_quantize(x: np.ndarray, bits: int, dither: bool = True) -> np.ndarray:
    q = 2 ** (bits - 1)
    if dither:
        rng = np.random.default_rng(1234)
        tpdf = (rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32))
        # mise en forme simple du 1er ordre (pousse le bruit vers l'aigu)
        shaped = tpdf - 0.5 * np.roll(tpdf, 1, axis=-1)
        y = np.round(x * (q - 1) + shaped)
    else:
        y = np.round(x * (q - 1))
    return np.clip(y, -q, q - 1).astype(np.int32)


def write_wav(path: str, audio: np.ndarray, sr: int, bits: str, mask: int, dither=True):
    """WAV extensible / RF64 multicanal (écriture par blocs)."""
    n_ch, n = audio.shape
    is_float = bits.startswith("32")
    bps = 32 if is_float else int(bits.split()[0])
    block_align = n_ch * bps // 8
    data_size = n * block_align
    rf64 = data_size + 80 > 0xFFFFFFFF
    with open(path, "wb") as f:
        if rf64:
            f.write(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE")
            f.write(b"ds64" + struct.pack("<IQQQI", 28, 0, data_size, n, 0))
        else:
            f.write(b"RIFF" + struct.pack("<I", 0) + b"WAVE")
        fmt = struct.pack("<HHIIHH", 0xFFFE, n_ch, sr, sr * block_align, block_align, bps)
        fmt += struct.pack("<HHI", 22, bps, mask)
        fmt += _KSDATAFORMAT_FLOAT if is_float else _KSDATAFORMAT_PCM
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", 0xFFFFFFFF if rf64 else data_size))
        step = 1 << 18
        for s in range(0, n, step):
            blk = audio[:, s: s + step]
            if is_float:
                f.write(np.ascontiguousarray(blk.T, dtype="<f4").tobytes())
            else:
                q = dither_quantize(blk, bps, dither).T
                if bps == 16:
                    f.write(np.ascontiguousarray(q, dtype="<i2").tobytes())
                else:  # 24 bits packés
                    b = np.ascontiguousarray(q, dtype="<i4").view(np.uint8).reshape(-1, 4)[:, :3]
                    f.write(b.tobytes())
        end = f.tell()
        if data_size % 2:
            f.write(b"\x00")
            end += 1
        if rf64:
            f.seek(20)
            f.write(struct.pack("<QQQ", end - 8, data_size, n))
        else:
            f.seek(4)
            f.write(struct.pack("<I", end - 8))


def write_flac(path: str, audio: np.ndarray, sr: int, bits: str, dither=True):
    import soundfile as sf

    bps = int(bits.split()[0])
    q = dither_quantize(audio, bps, dither)
    if bps == 24:
        data = (q.astype(np.int32) << 8).T  # soundfile attend l'int32 aligné à gauche
        sf.write(path, data, sr, format="FLAC", subtype="PCM_24")
    else:
        sf.write(path, q.astype(np.int16).T, sr, format="FLAC", subtype="PCM_16")


def render(audio: np.ndarray, sr: int, cfg: ExportSettings, stems: dict | None = None,
           log=_noop) -> np.ndarray:
    from . import analysis as an
    from .mastering import true_peak_limiter

    use = stems if (cfg.use_stems and stems and cfg.layout not in ("Mono", "Stéréo", "2.0")) else None
    if use:
        log("Spatialisation objet à partir des stems IA…")
    elif cfg.layout not in ("Mono", "Stéréo", "2.0"):
        log("Upmix spectral direct/ambiance…")
    out = spatial.upmix(audio, sr, cfg.layout, use)
    if use:  # recalage loudness sur le master + sécurité true-peak
        ref = an.integrated_loudness(audio, sr)
        cur = an.integrated_loudness(out, sr)
        if ref > -69 and cur > -69:
            out = out * 10 ** ((ref - cur) / 20)
        tp = an.true_peak_db(audio, sr)
        out, _ = true_peak_limiter(out, sr, min(-1.0, tp))
    if cfg.samplerate != sr:
        log(f"Rééchantillonnage {sr} → {cfg.samplerate} Hz (SoX VHQ)")
        out = resample(out, sr, cfg.samplerate)
    return out


def export(path: str, audio: np.ndarray, sr: int, cfg: ExportSettings,
           stems: dict | None = None, log=_noop) -> list[str]:
    out = render(audio, sr, cfg, stems, log)
    return save(path, out, cfg, log)


def save(path: str, out: np.ndarray, cfg: ExportSettings, log=_noop) -> list[str]:
    names = spatial.LAYOUTS[cfg.layout]
    base, _ = os.path.splitext(path)
    written: list[str] = []
    if cfg.fmt == "WAV":
        p = base + ".wav"
        write_wav(p, out, cfg.samplerate, cfg.bits, spatial.channel_mask(cfg.layout), cfg.dither)
        written.append(p)
    else:
        if out.shape[0] <= 8:
            p = base + ".flac"
            write_flac(p, out, cfg.samplerate, cfg.bits, cfg.dither)
            written.append(p)
        else:
            folder = base + f"_{cfg.layout}_FLAC"
            os.makedirs(folder, exist_ok=True)
            log(f"FLAC limité à 8 canaux → export multi-mono ({out.shape[0]} fichiers)")
            stem_name = os.path.basename(base)
            for i, nm in enumerate(names):
                p = os.path.join(folder, f"{stem_name}_{i + 1:02d}_{nm}.flac")
                write_flac(p, out[i: i + 1], cfg.samplerate, cfg.bits, cfg.dither)
                written.append(p)
    if len(names) > 2:
        txt = base + f"_{cfg.layout}_channels.txt"
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(f"REMASTRA — format {cfg.layout} ({len(names)} canaux)\n")
            fh.write(f"{spatial.LAYOUT_DESC[cfg.layout]}\n\nOrdre des canaux :\n")
            for i, nm in enumerate(names):
                fh.write(f"  {i + 1:2d}  {nm}\n")
        written.append(txt)
    for w in written:
        log(f"✔ Écrit : {w}")
    return written
