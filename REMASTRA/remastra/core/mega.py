"""Séparation par BS-RoFormer — plusieurs modèles (licence MIT / usage libre).

Modèles disponibles (clé de registre → fichier) :
  * "mega53"       → mvsep_mega_model_bs_roformer_53_stems_v1.ckpt (≈ 1,4 Go, 681 M paramètres)
                      Source : https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21
                      53 instruments qui se recouvrent (ex. « vocal » contient « lead-vocal » +
                      « back-vocal »). REMASTRA construit un jeu « principal » non redondant
                      (voix, batterie, basse, guitare, piano, cordes, synthé + résidu) qui se
                      somme exactement au mix d'origine.
  * "male_female"  → bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt (≈ 330 Mo)
                      Modèle par aufr33 : sépare la voix en deux stems, voix d'homme et voix de
                      femme, avec un résidu instrumental calculé.
  * "vocals_1296"  → model_bs_roformer_ep_368_sdr_12.9628.ckpt (≈ 610 Mo, SDR 12.96 dB)
                      Modèle « BS-RoFormer-Viperx-1296 » : séparation voix / instrumental haute
                      précision, très utilisé (UVR).

Le code réseau (models/bs_roformer) est générique : il charge n'importe quel checkpoint
BS-RoFormer dès lors que la configuration (dim, depth, num_stems, freqs_per_bands…) est
correcte, ce que fournit REGISTRY[model_id]["model_config"] pour chaque modèle.
"""
from __future__ import annotations

import colorsys
import gc
import os
import tempfile
import urllib.request

import numpy as np

from .audio_io import resample, to_stereo

SAMPLE_RATE = 44100

# --------------------------------------------------------------------------- #
#  Modèle « mega53 » — MVSep Mega BS-RoFormer, 53 instruments
# --------------------------------------------------------------------------- #
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

_MEGA53_CONFIG = dict(
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

# --------------------------------------------------------------------------- #
#  Modèle « male_female » — aufr33, séparation voix homme / voix femme
# --------------------------------------------------------------------------- #
_MALE_FEMALE_CONFIG = dict(
    dim=384, depth=8, stereo=True, num_stems=2, time_transformer_depth=1,
    freq_transformer_depth=1, linear_transformer_depth=0,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, flash_attn=True,
    dim_freqs_in=1025, stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
    stft_normalized=False, mask_estimator_depth=2, multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256), multi_stft_hop_size=147,
    multi_stft_normalized=False,
)

# --------------------------------------------------------------------------- #
#  Modèle « vocals_1296 » — BS-RoFormer-Viperx-1296, voix / instrumental
# --------------------------------------------------------------------------- #
_VOCALS_1296_CONFIG = dict(
    dim=512, depth=12, stereo=True, num_stems=1, time_transformer_depth=1,
    freq_transformer_depth=1, linear_transformer_depth=0,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, flash_attn=True,
    dim_freqs_in=1025, stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
    stft_normalized=False, mask_estimator_depth=2, multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256), multi_stft_hop_size=147,
    multi_stft_normalized=False,
)

STANDARD = "Standard — segments de 20 s (GPU ≥ 16 Go)"
ECONOMY = "Économe — segments de 10 s (GPU 10–12 Go)"
MINIMAL = "Minimal — segments de 5 s (GPU 6–8 Go / CPU)"
STANDARD_S = "Standard — segments de 8 s (GPU ≥ 8 Go)"
ECONOMY_S = "Économe — segments de 4 s (GPU 4–6 Go)"
MINIMAL_S = "Minimal — segments de 2 s (GPU / CPU)"

MEMORY_MODES = {STANDARD: 882000, ECONOMY: 441000, MINIMAL: 220500}

# --------------------------------------------------------------------------- #
#  Registre des modèles BS-RoFormer
# --------------------------------------------------------------------------- #
REGISTRY = {
    "mega53": dict(
        key="mega53",
        label_fr="MVSep Mega BS-RoFormer — 53 stems",
        ckpt_name="mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
        ckpt_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/"
                 "download/v1.0.21/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
        ckpt_size=1_368_919_887,
        instruments=INSTRUMENTS,
        labels_fr=LABELS_FR,
        model_config=_MEGA53_CONFIG,
        memory_modes=MEMORY_MODES,
        residual_mode="main_set",
        main_set=MAIN_SET,
        spatial_map=SPATIAL_MAP,
        note_fr="Les 53 stems se recouvrent (ex. « Voix » = voix lead + chœurs). REMASTRA "
                "active par défaut un jeu principal (voix, batterie, basse, guitare, piano, "
                "cordes, synthé + résidu) qui se somme exactement au mix ; les instruments "
                "détaillés sont en mute. GPU NVIDIA 16 Go recommandé, très lent sur CPU.",
    ),
    "male_female": dict(
        key="male_female",
        label_fr="BS-RoFormer — Voix homme / femme (aufr33)",
        ckpt_name="bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
        ckpt_url="https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/"
                 "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
        ckpt_size=330_000_000,
        instruments=["male", "female"],
        labels_fr={"male": "Voix (homme)", "female": "Voix (femme)",
                   "other": "Instrumental (résidu)"},
        model_config=_MALE_FEMALE_CONFIG,
        memory_modes={STANDARD_S: 352800, ECONOMY_S: 176400, MINIMAL_S: 88200},
        residual_mode="sum",
        residual_name="other",
        spatial_map={"male": "vocals", "female": "vocals", "other": "other"},
        note_fr="Sépare la voix en deux stems, voix d'homme et voix de femme, plus un résidu "
                "instrumental calculé (la somme redonne exactement le mix). Idéal pour isoler "
                "un duo, un chœur mixte ou un featuring. Fonctionne mieux sur une piste déjà "
                "isolée (voix) ou un mix où la voix domine.",
    ),
    "vocals_1296": dict(
        key="vocals_1296",
        label_fr="BS-RoFormer-1296 — Voix / instrumental (viperx)",
        ckpt_name="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        ckpt_url="https://github.com/TRvlvr/model_repo/releases/download/"
                 "all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        ckpt_size=639_317_465,
        instruments=["vocal"],
        labels_fr={"vocal": "Voix", "other": "Instrumental (résidu)"},
        model_config=_VOCALS_1296_CONFIG,
        memory_modes={STANDARD_S: 352800, ECONOMY_S: 176400, MINIMAL_S: 88200},
        residual_mode="sum",
        residual_name="other",
        spatial_map={"vocal": "vocals", "other": "other"},
        note_fr="Modèle de référence pour la séparation voix / instrumental (SDR 12,96 dB), "
                "très utilisé par la communauté UVR. Le résidu instrumental se somme "
                "exactement au mix. Excellent choix par défaut pour isoler une voix lead.",
    ),
}

# Ordre d'affichage dans le sélecteur de modèle (Demucs est géré par stems.py)
MODEL_IDS = ["mega53", "male_female", "vocals_1296"]

# --------------------------------------------------------------------------- #
#  Alias rétro-compatibles (ancien API à un seul modèle)
# --------------------------------------------------------------------------- #
CKPT_NAME = REGISTRY["mega53"]["ckpt_name"]
CKPT_URL = REGISTRY["mega53"]["ckpt_url"]
CKPT_SIZE = REGISTRY["mega53"]["ckpt_size"]
MODEL_CONFIG = REGISTRY["mega53"]["model_config"]

_MODEL = {"model_id": None, "path": None, "net": None, "device": None}


def _noop(*_a, **_k):
    pass


def models(ids=None):
    """Liste ordonnée des infos de modèles (pour construire le sélecteur GUI)."""
    return [REGISTRY[k] for k in (ids or MODEL_IDS)]


def model_label(model_id: str) -> str:
    return REGISTRY[model_id]["label_fr"]


def label(name: str, model_id: str = "mega53") -> str:
    return REGISTRY.get(model_id, REGISTRY["mega53"])["labels_fr"].get(name, name)


def color(name: str, model_id: str = "mega53") -> str:
    if name in ("other", "instrumental"):
        return "#9AA4B2"
    instruments = REGISTRY.get(model_id, REGISTRY["mega53"])["instruments"]
    i = instruments.index(name) if name in instruments else 0
    h = (i * 0.618034) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.62, 0.85)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


# --------------------------------------------------------------------------- #
#  Emplacement / téléchargement des modèles
# --------------------------------------------------------------------------- #
def models_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "REMASTRA", "models")
    os.makedirs(d, exist_ok=True)
    return d


def default_ckpt_path(model_id: str = "mega53") -> str:
    return os.path.join(models_dir(), REGISTRY[model_id]["ckpt_name"])


def find_ckpt(model_id: str = "mega53", preferred: str | None = None) -> str | None:
    name = REGISTRY[model_id]["ckpt_name"]
    for p in (preferred, default_ckpt_path(model_id),
              os.path.join(os.getcwd(), "models", name), os.path.join(os.getcwd(), name)):
        if p and os.path.isfile(p) and os.path.getsize(p) > 50_000_000:
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


def download(model_id: str = "mega53", progress=_noop, log=_noop, dest: str | None = None) -> str:
    info = REGISTRY[model_id]
    dest = dest or default_ckpt_path(model_id)
    tmp = dest + ".part"
    log(f"Téléchargement du modèle {info['label_fr']} (≈ {info['ckpt_size'] / 1e9:.2f} Go)…")
    req = urllib.request.Request(info["ckpt_url"], headers={"User-Agent": "REMASTRA/1.5"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or info["ckpt_size"])
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            progress(done / total)
    if os.path.getsize(tmp) < 50_000_000:
        os.remove(tmp)
        raise RuntimeError("Téléchargement incomplet du modèle.")
    os.replace(tmp, dest)
    log(f"Modèle enregistré : {dest}")
    return dest


# Wrappers rétro-compatibles pour le modèle 53 stems
def download_legacy(progress=_noop, log=_noop, dest: str | None = None) -> str:
    return download("mega53", progress, log, dest)


# --------------------------------------------------------------------------- #
#  Inférence
# --------------------------------------------------------------------------- #
def _load(model_id: str, path: str, log=_noop):
    import torch

    from .stems import device_name
    from .models.bs_roformer import BSRoformer

    info = REGISTRY[model_id]
    dev = device_name()
    if (_MODEL["net"] is not None and _MODEL["path"] == path
            and _MODEL["model_id"] == model_id and _MODEL["device"] == dev):
        return _MODEL["net"], dev
    log(f"Chargement du modèle {info['label_fr']}…")
    _MODEL.update(model_id=None, path=None, net=None, device=None)   # libère l'ancien modèle
    gc.collect()
    net = BSRoformer(**info["model_config"])
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
    _MODEL.update(model_id=model_id, path=path, net=net, device=dev)
    return net, dev


def separate(audio: np.ndarray, sr: int, ckpt: str, memory_mode: str = "",
             hide_silent: bool = True, silence_db: float = -40.0,
             progress=_noop, log=_noop, model_id: str = "mega53") -> dict[str, np.ndarray]:
    """Sépare avec le modèle BS-RoFormer `model_id`. Retourne {nom: audio stéréo (2, n)}."""
    import torch

    info = REGISTRY.get(model_id, REGISTRY["mega53"])
    instruments = info["instruments"]
    if not available():
        raise RuntimeError("Dépendances manquantes : pip install torch einops beartype "
                           "rotary-embedding-torch")
    if not ckpt or not os.path.isfile(ckpt):
        raise RuntimeError(f"Modèle « {info['label_fr']} » introuvable. Téléchargez-le ou "
                           f"choisissez le fichier {info['ckpt_name']}.")
    net, dev = _load(model_id, ckpt, log)
    modes = info["memory_modes"]
    default_c = next(iter(modes.values()))
    C = modes.get(memory_mode, default_c if dev != "cpu" else min(modes.values()))
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
    S = len(instruments)
    win = np.ones(C, np.float32)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)

    nbytes = S * 2 * L * 4
    tmpdir = None
    if nbytes > 1.5e9:  # gros morceau : accumulation sur disque (mémoire vive préservée)
        tmpdir = tempfile.mkdtemp(prefix="remastra_roformer_")
        acc = np.lib.format.open_memmap(os.path.join(tmpdir, "acc.npy"), "w+", np.float32, (S, 2, L))
    else:
        acc = np.zeros((S, 2, L), np.float32)
    cnt = np.zeros(L, np.float32)

    starts = list(range(0, L, step))
    log(f"Inférence BS-RoFormer ({info['label_fr']}) sur {dev.upper()} — "
        f"{len(starts)} segments de {C / SAMPLE_RATE:.0f} s…")
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
                y = net(x)[0].float().cpu().numpy()          # (S, 2, C)
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
    for s, name in enumerate(instruments):
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

    st_in = to_stereo(audio)[:, :n_in]
    if info["residual_mode"] == "main_set":
        # Résidu : ce que le jeu principal ne couvre pas → somme exacte = mix d'origine
        main = [k for k in info["main_set"] if k in out]
        resid = st_in.copy()
        for k in main:
            resid[:, : out[k].shape[1]] -= out[k]
        out["other"] = resid.astype(np.float32)
        ordered = [k for k in info["main_set"] if k in out] + ["other"] + \
            sorted([k for k in out if k not in info["main_set"] and k != "other"],
                   key=lambda k: label(k, model_id))
    else:
        # Résidu générique : mix - somme des stems produits (voix / instrumental, homme / femme…)
        resid_name = info.get("residual_name", "other")
        resid = st_in.copy()
        for k in out:
            resid[:, : out[k].shape[1]] -= out[k]
        out[resid_name] = resid.astype(np.float32)
        ordered = list(out.keys())

    log(f"{len(out) - 1} stem(s) détecté(s)"
        + (f", {len(silent)} absents masqués" if silent else ""))
    progress(1.0)
    return {k: out[k] for k in ordered}


def is_mega_set(stems: dict, model_id: str = "mega53") -> bool:
    if model_id != "mega53":
        return False
    return any(k in stems for k in ("lead-vocal", "back-vocal", "kick", "violin", "hh"))


def default_active(stems: dict, model_id: str = "mega53") -> set[str]:
    """Stems actifs par défaut dans le mix (jeu principal non redondant, modèle 53 stems)."""
    if not is_mega_set(stems, model_id):
        return set(stems)
    return {k for k in stems if k in MAIN_SET or k == "other"}


def spatial_groups(stems: dict, model_id: str = "mega53") -> dict:
    """Stems → groupes de spatialisation, selon la table du modèle qui les a produits."""
    info = REGISTRY.get(model_id, REGISTRY["mega53"])
    smap = info.get("spatial_map")
    if not smap:
        return stems
    out: dict[str, np.ndarray] = {}
    for k, v in stems.items():
        tgt = smap.get(k)
        if tgt is None:
            continue
        if tgt in out:
            n = min(out[tgt].shape[1], v.shape[1])
            out[tgt] = out[tgt][:, :n] + v[:, :n]
        else:
            out[tgt] = v
    return out
