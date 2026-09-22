"""Chargement audio (tous formats) et rééchantillonnage haute qualité."""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

AUDIO_FILTER = (
    "Audio / Vidéo (*.wav *.flac *.mp3 *.ogg *.opus *.m4a *.aac *.aif *.aiff "
    "*.wma *.mp4 *.mov *.mkv *.avi *.webm);;Tous les fichiers (*.*)"
)


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        from shutil import which

        return which("ffmpeg")


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Charge un fichier audio/vidéo -> (audio[canaux, n], samplerate)."""
    if sf is not None:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return np.ascontiguousarray(data.T), int(sr)
        except Exception:
            pass
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError("Format non supporté et FFmpeg introuvable.")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", path, "-vn", "-acodec", "pcm_f32le", tmp],
            check=True, creationflags=flags,
        )
        data, sr = sf.read(tmp, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Rééchantillonnage qualité mastering (soxr VHQ, sinon polyphase)."""
    if sr_in == sr_out:
        return audio
    try:
        import soxr

        out = soxr.resample(audio.T, sr_in, sr_out, quality="VHQ")
        return np.ascontiguousarray(out.T, dtype=np.float32)
    except Exception:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(audio, sr_out // g, sr_in // g, axis=1).astype(np.float32)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] == 2:
        return audio
    # Downmix ITU-R BS.775 (L R C LFE Ls Rs ...)
    L, R = audio[0], audio[1]
    C = audio[2] if audio.shape[0] > 2 else 0
    extra_l = audio[4] if audio.shape[0] > 4 else 0
    extra_r = audio[5] if audio.shape[0] > 5 else 0
    g = 0.7071
    out = np.stack([L + g * C + g * extra_l, R + g * C + g * extra_r])
    peak = np.max(np.abs(out)) + 1e-12
    return (out / max(1.0, peak)).astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0, keepdims=True).astype(np.float32)


def write_preview_wav(audio: np.ndarray, sr: int, path: str) -> str:
    """Fichier de pré-écoute (stéréo 16 bits)."""
    st = to_stereo(audio)
    peak = float(np.max(np.abs(st))) if st.size else 0.0
    if peak > 1.0:
        st = st / peak
    sf.write(path, st.T, sr, subtype="PCM_16")
    return path
