"""Séparation 53 stems — MVSep Mega BS-RoFormer (ZFTurbo / MVSep, licence MIT).

Modèle : mvsep_mega_model_bs_roformer_53_stems_v1.ckpt (≈ 1,4 Go, 681 M paramètres)
Source : https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21

Particularités du modèle :
  * les stems se recouvrent (ex. « vocal » contient « lead-vocal » + « back-vocal »,
    « drums » contient « kick », « snare », « hh »…) : ils ne se somment pas au mix ;
  * REMASTRA construit donc un jeu « principal » non redondant (voix, batterie,
    basse, guitare, piano, cordes, synthé + résidu « autres ») qui, lui, se somme
    exactement au mix d'origine. Les stems de détail restent disponibles à part.
  * GPU NVIDIA avec 16 Go de VRAM recommandé (mode « Économe » pour 8–12 Go).
"""
from __future__ import annotations

import colorsys
import gc
import os
import tempfile
import urllib.request

import numpy as np

from .audio_io import resample, to_stereo

CKPT_NAME = "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
CKPT_URL = ("https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.21/" + CKPT_NAME)
CKPT_SIZE = 1_368_919_887
SAMPLE_RATE = 44100

INSTRUMENTS = [
    "accordion", "acoustic-guitar", "back-vocal", "banjo", "bass", "bassoon", "bells",
    "bowed_strings", "brass", "cello", "clarinet", "congas", "digital-piano", "dobro",
    "double-bass", "drums", "electric-guitar", "flute", "french-horn", "glockenspiel",
    "guitar", "harmonica", "harp", "harpsichord", "hh", "keys", "kick", "lead-vocal",
    "mandolin", "marimba", "oboe", "organ", "percussion", "piano", "saxophone", "sitar",
    "snare", "strings", "synth", "tambourine", "timpani", "toms", "triangle", "trombone",
    "trumpet", "tuba", "ukulele", "viola", "violin", "vocal", "wind", "wind-chimes", "woodwind",
]

LABELS_FR = {
    "accordion": "Accordéon", "acoustic-guitar": "Guitare acoustique", "back-vocal": "Chœurs",
    "banjo": "Banjo", "bass": "Basse", "bassoon": "Basson", "bells": "Cloches",
    "bowed_strings": "Cordes frottées", "brass": "Cuivres", "cello": "Violoncelle",
    "clarinet": "Clarinette", "congas": "Congas", "digital-piano": "Piano numérique",
    "dobro": "Dobro", "double-bass": "Contrebasse", "drums": "Batterie",
    "electric-guitar": "Guitare électrique", "flute": "Flûte", "french-horn": "Cor",
    "glockenspiel": "Glockenspiel", "guitar": "Guitare", "harmonica": "Harmonica",
    "harp": "Harpe", "harpsichord": "Clavecin", "hh": "Charleston (hi-hat)", "keys": "Claviers",
    "kick": "Grosse caisse", "lead-vocal": "Voix lead", "mandolin": "Mandoline",
    "marimba": "Marimba", "oboe": "Hautbois", "organ": "Orgue", "percussion": "Percussions",
    "piano": "Piano", "saxophone": "Saxophone", "sitar": "Sitar", "snare": "Caisse claire",
    "strings": "Cordes", "synth": "Synthé", "tambourine": "Tambourin", "timpani": "Timbales",
    "toms": "Toms", "triangle": "Triangle", "trombone": "Trombone", "trumpet": "Trompette",
    "tuba": "Tuba", "ukulele": "Ukulélé", "viola": "Alto", "violin": "Violon", "vocal": "Voix",
    "wind": "Vents", "wind-chimes": "Carillon à vent", "woodwind": "Bois",
    "other": "Autres (résidu)",
}

# Jeu principal non redondant (+ « other » = résidu calculé) : se somme au mix.
MAIN_SET = ["vocal", "drums", "bass", "guitar", "piano", "strings", "synth"]
# Correspondance vers les groupes utilisés par la spatialisation 5.1 → 9.1.6
SPATIAL_MAP = {"vocal": "vocals", "drums": "drums", "bass": "bass", "guitar": "guitar",
               "piano": "piano", "strings": "strings", "synth": "synth", "other": "other"}

MODEL_CONFIG = dict(
    dim=256, depth=12, stereo=True, num_stems=53, time_transformer_depth=1,
    freq_transformer_depth=1, linear_transformer_depth=0,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, flash_attn=True,
    dim_freqs_in=1025, stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048,
    stft_normalized=False, mask_estimator_depth=2, multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256), multi_stft_hop_size=147,
    multi_stft_normalized=False, mlp_expansion_factor=2, use_torch_checkpoint=False,
    skip_connection=False,
)

MEMORY_MODES = {
    "Standard — segments de 20 s (GPU ≥ 16 Go)": 882000,
    "Économe — segments de 10 s (GPU 10–12 Go)": 441000,
    "Minimal — segments de 5 s (GPU 6–8 Go / CPU)": 220500,
}

_MODEL = {"path": None, "net": None, "device": None}


def _noop(*_a, **_k):
    pass


def label(name: str) -> str:
    return LABELS_FR.get(name, name)


def color(name: str) -> str:
    if name == "other":
        return "#9AA4B2"
    i = INSTRUMENTS.index(name) if name in INSTRUMENTS else 0
    h = (i * 0.618034) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.62, 0.85)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


# --------------------------------------------------------------------------- #
#  Emplacement / téléchargement du modèle
# --------------------------------------------------------------------------- #
def models_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "REMASTRA", "models")
    os.makedirs(d, exist_ok=True)
    return d


def default_ckpt_path() -> str:
    return os.path.join(models_dir(), CKPT_NAME)


def find_ckpt(preferred: str | None = None) -> str | None:
    for p in (preferred, default_ckpt_path(),
              os.path.join(os.getcwd(), "models", CKPT_NAME), os.path.join(os.getcwd(), CKPT_NAME)):
        if p and os.path.isfile(p) and os.path.getsize(p) > 100_000_000:
            return p
    return None


def available() -> bool:
    try:
        import beartype  # noqa: F401
        import einops  # noqa: F401
        import rotary_embedding_torch  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def download(progress=_noop, log=_noop, dest: str | None = None) -> str:
    dest = dest or default_ckpt_path()
    tmp = dest + ".part"
    log(f"Téléchargement du modèle MVSep Mega 53 stems (≈ {CKPT_SIZE / 1e9:.1f} Go)…")
    req = urllib.request.Request(CKPT_URL, headers={"User-Agent": "REMASTRA/1.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or CKPT_SIZE)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            progress(done / total)
    if os.path.getsize(tmp) < 100_000_000:
        os.remove(tmp)
        raise RuntimeError("Téléchargement incomplet du modèle.")
    os.replace(tmp, dest)
    log(f"Modèle enregistré : {dest}")
    return dest


# --------------------------------------------------------------------------- #
#  Inférence
# --------------------------------------------------------------------------- #
def _load(path: str, log=_noop):
    import torch

    from .stems import device_name
    from .models.bs_roformer import BSRoformer

    dev = device_name()
    if _MODEL["net"] is not None and _MODEL["path"] == path and _MODEL["device"] == dev:
        return _MODEL["net"], dev
    log("Chargement du modèle BS-RoFormer 53 stems (681 M paramètres)…")
    _MODEL.update(path=None, net=None, device=None)   # libère l'ancien modèle
    gc.collect()
    net = BSRoformer(**MODEL_CONFIG)
    if dev == "cuda":
        net.half()           # poids FP16 (comme le checkpoint) : VRAM divisée par 2
    try:                     # mmap : pas de 2e copie des poids en mémoire vive
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    del sd
    gc.collect()
    net.eval().to(dev)
    _MODEL.update(path=path, net=net, device=dev)
    return net, dev


def separate(audio: np.ndarray, sr: int, ckpt: str, memory_mode: str = "",
             hide_silent: bool = True, silence_db: float = -40.0,
             progress=_noop, log=_noop) -> dict[str, np.ndarray]:
    """Sépare en 53 stems. Retourne {nom: audio stéréo (2, n)} au sr d'entrée."""
    import torch

    if not available():
        raise RuntimeError("Dépendances manquantes : pip install torch einops beartype "
                           "rotary-embedding-torch")
    if not ckpt or not os.path.isfile(ckpt):
        raise RuntimeError("Modèle 53 stems introuvable. Téléchargez-le ou choisissez le "
                           f"fichier {CKPT_NAME}.")
    net, dev = _load(ckpt, log)
    C = MEMORY_MODES.get(memory_mode, 441000 if dev == "cpu" else 882000)
    n_in = audio.shape[1]
    mix = resample(to_stereo(audio), sr, SAMPLE_RATE).astype(np.float32)
    n = mix.shape[1]
    N = 2
    step = C // N
    fade = C // 10
    border = C - step
    if n > 2 * border:
        mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")
    else:
        border = 0
    L = mix.shape[1]
    S = len(INSTRUMENTS)
    win = np.ones(C, np.float32)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)

    nbytes = S * 2 * L * 4
    tmpdir = None
    if nbytes > 1.5e9:  # gros morceau : accumulation sur disque (mémoire vive préservée)
        tmpdir = tempfile.mkdtemp(prefix="remastra_mega_")
        acc = np.lib.format.open_memmap(os.path.join(tmpdir, "acc.npy"), "w+", np.float32, (S, 2, L))
    else:
        acc = np.zeros((S, 2, L), np.float32)
    cnt = np.zeros(L, np.float32)

    starts = list(range(0, L, step))
    log(f"Inférence BS-RoFormer sur {dev.upper()} — {len(starts)} segments de {C / SAMPLE_RATE:.0f} s…")
    use_amp = dev == "cuda"
    with torch.inference_mode():
        for k, i in enumerate(starts):
            part = mix[:, i:i + C]
            ln = part.shape[1]
            if ln < C:
                mode = "reflect" if ln > C // 2 + 1 else "constant"
                part = np.pad(part, ((0, 0), (0, C - ln)), mode=mode)
            x = torch.from_numpy(np.ascontiguousarray(part))[None].to(dev)
            if dev == "cuda":
                x = x.half()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                y = net(x)[0].float().cpu().numpy()          # (53, 2, C)
            w = win.copy()
            if i == 0:
                w[:fade] = 1
            if i + C >= L:
                w[-fade:] = 1
            acc[:, :, i:i + ln] += y[:, :, :ln] * w[:ln]
            cnt[i:i + ln] += w[:ln]
            progress(0.92 * (k + 1) / len(starts))
            if i + C >= L:
                break

    log("Assemblage des stems…")
    ref = float(np.sqrt(np.mean(mix ** 2)) + 1e-9)
    out: dict[str, np.ndarray] = {}
    silent = []
    for s, name in enumerate(INSTRUMENTS):
        st = np.asarray(acc[s, :, border: L - border] if border else acc[s]) / np.maximum(
            cnt[border: L - border] if border else cnt, 1e-8)
        rms = float(np.sqrt(np.mean(st ** 2)) + 1e-12)
        if hide_silent and 20 * np.log10(rms / ref) < silence_db:
            silent.append(name)
            continue
        out[name] = resample(st.astype(np.float32), SAMPLE_RATE, sr)[:, :n_in]
    if tmpdir:
        del acc
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    # Résidu : ce que le jeu principal ne couvre pas → somme exacte = mix d'origine
    st_in = to_stereo(audio)[:, :n_in]
    main = [k for k in MAIN_SET if k in out]
    resid = st_in.copy()
    for k in main:
        resid[:, : out[k].shape[1]] -= out[k]
    out["other"] = resid.astype(np.float32)
    log(f"{len(out) - 1} instruments détectés"
        + (f", {len(silent)} absents masqués" if silent else ""))
    progress(1.0)
    ordered = [k for k in MAIN_SET if k in out] + ["other"] + \
        sorted([k for k in out if k not in MAIN_SET and k != "other"], key=lambda k: label(k))
    return {k: out[k] for k in ordered}


def is_mega_set(stems: dict) -> bool:
    return any(k in stems for k in ("vocal", "lead-vocal", "kick", "violin", "hh"))


def default_active(stems: dict) -> set[str]:
    """Stems actifs par défaut dans le mix (jeu principal non redondant)."""
    if not is_mega_set(stems):
        return set(stems)
    return {k for k in stems if k in MAIN_SET or k == "other"}


def spatial_groups(stems: dict) -> dict:
    """Stems → groupes de spatialisation (uniquement le jeu principal non redondant)."""
    if not is_mega_set(stems):
        return stems
    return {SPATIAL_MAP[k]: v for k, v in stems.items() if k in SPATIAL_MAP}
