"""Traductions REMASTRA (français = langue source, anglais = traduction).

* tr(texte)      : libellé d'interface (correspondance exacte).
* tr_msg(texte)  : messages dynamiques du moteur (journal, décisions de l'IA,
                   erreurs) — correspondance exacte puis motifs regex.
"""
from __future__ import annotations

import re

LANGS = {"fr": "Français", "en": "English"}
LANG = "fr"


def set_lang(lang: str):
    global LANG
    LANG = lang if lang in LANGS else "fr"


def tr(s: str) -> str:
    if LANG == "fr" or not isinstance(s, str):
        return s
    return EN.get(s, s)


def tr_msg(s: str) -> str:
    if LANG == "fr" or not isinstance(s, str):
        return s
    if s in EN:
        return EN[s]
    for rx, tpl in _PATTERNS:
        m = rx.fullmatch(s)
        if m:
            return tpl.format(*[tr(g) if g is not None else "" for g in m.groups()])
    return s


# --------------------------------------------------------------------------- #
EN: dict[str, str] = {
    # --- Navigation / barre du haut ------------------------------------------
    "WORKFLOW": "WORKFLOW",
    "①  Import && Analyse": "①  Import && Analysis",
    "②  Débruitage IA": "②  AI Denoise",
    "③  Stems IA": "③  AI Stems",
    "④  Remaster IA": "④  AI Remaster",
    "⑤  Export": "⑤  Export",
    "ÉCOUTE": "LISTEN",
    "Original": "Original",
    "Débruité": "Denoised",
    "Master": "Master",
    "Mix stems": "Stem mix",
    "MOTEURS IA": "AI ENGINES",
    "Détection…": "Detecting…",
    "Aucun fichier chargé": "No file loaded",
    "Astuce : cliquez pour vous déplacer, glissez pour sélectionner":
        "Tip: click to seek, drag to select",
    "Prêt.": "Ready.",
    "Journal ▾": "Log ▾",
    "Langue": "Language",
    # --- Import ---------------------------------------------------------------
    "Import & Analyse": "Import & Analysis",
    "Musique, doublage, voix, podcast, bande-son vidéo… REMASTRA analyse le loudness (EBU R128 / "
    "ITU-R BS.1770-4), le true-peak, la dynamique et le spectre.":
        "Music, dubbing, voice, podcasts, video soundtracks… REMASTRA analyzes loudness "
        "(EBU R128 / ITU-R BS.1770-4), true peak, dynamics and spectrum.",
    "Déposez votre fichier ici": "Drop your file here",
    "WAV · FLAC · MP3 · OGG · OPUS · M4A · AAC · AIFF · WMA · MP4 · MOV · MKV (piste audio)":
        "WAV · FLAC · MP3 · OGG · OPUS · M4A · AAC · AIFF · WMA · MP4 · MOV · MKV (audio track)",
    "Ouvrir un fichier…": "Open a file…",
    "Loudness intégré": "Integrated loudness",
    "True Peak": "True Peak",
    "Loudness Range": "Loudness Range",
    "Facteur de crête": "Crest factor",
    "Corrélation stéréo": "Stereo correlation",
    "Contenu détecté": "Detected content",
    "analyse IA": "AI analysis",
    "Spectre moyen": "Average spectrum",
    "voix": "voice",
    "musique": "music",
    "Voix": "Vocals",
    "Musique": "Music",
    "Ouvrir un fichier audio": "Open an audio file",
    "Audio / Vidéo (*.wav *.flac *.mp3 *.ogg *.opus *.m4a *.aac *.aif *.aiff *.wma *.mp4 *.mov "
    "*.mkv *.avi *.webm);;Tous les fichiers (*.*)":
        "Audio / Video (*.wav *.flac *.mp3 *.ogg *.opus *.m4a *.aac *.aif *.aiff *.wma *.mp4 *.mov "
        "*.mkv *.avi *.webm);;All files (*.*)",
    # --- Débruitage -----------------------------------------------------------
    "Débruitage IA": "AI Denoise",
    "Supprime proprement le bruit derrière la voix : souffle, ventilation, trafic, réverbération de "
    "bruit, ronflement secteur, clics. Le moteur « Isolation voix » (Demucs) retire même la musique "
    "et les ambiances.":
        "Cleanly removes the noise behind the voice: hiss, ventilation, traffic, noise reverb, mains "
        "hum, clicks. The \"Voice isolation\" engine (Demucs) even removes music and ambience.",
    "Moteur": "Engine",
    "Moteur : IA Isolation voix — Demucs": "Engine: AI Voice isolation — Demucs",
    "IA Isolation voix — Demucs": "AI Voice isolation — Demucs",
    "Fond conservé (isolation)": "Background kept (isolation)",
    "Restauration": "Restoration",
    "Filtre anti-rumble (< 60 Hz)": "Rumble filter (< 60 Hz)",
    "Anti-ronflement secteur": "Mains de-hum",
    "Anti-clic / crépitements": "De-click / crackle",
    "De-esser (sifflantes)": "De-esser (sibilance)",
    "✦  Lancer le débruitage": "✦  Run denoise",
    "Débruitage": "Denoise",
    # --- Stems ----------------------------------------------------------------
    "Séparation de stems IA": "AI Stem Separation",
    "Demucs v4 : 8 stems rapides qui se somment exactement au mix. MVSep Mega BS-RoFormer : "
    "jusqu'à 53 instruments (voix lead, chœurs, grosse caisse, caisse claire, violon, violoncelle, "
    "trompette, saxophone, orgue, harpe…). Deux modèles BS-RoFormer spécialisés voix : voix "
    "homme / femme (aufr33), et voix / instrumental haute précision (viperx).":
        "Demucs v4: 8 fast stems that add up exactly to the mix. MVSep Mega BS-RoFormer: up to 53 "
        "instruments (lead vocal, backing vocals, kick, snare, violin, cello, trumpet, saxophone, "
        "organ, harp…). Two specialized vocal BS-RoFormer models: male / female voice split "
        "(aufr33), and high-precision vocals / instrumental (viperx).",
    "Modèle": "Model",
    "Source": "Source",
    "Auto (dernière étape)": "Auto (latest step)",
    "Qualité": "Quality",
    "Rapide": "Fast",
    "Haute qualité": "High quality",
    "Ultra (lent)": "Ultra (slow)",
    "Séparation étendue (cordes / nappes / synthés)": "Extended separation (strings / pads / synths)",
    "✦  Séparer les stems": "✦  Separate stems",
    "Demucs v4 — 8 stems (rapide)": "Demucs v4 — 8 stems (fast)",
    "MVSep Mega BS-RoFormer — 53 stems": "MVSep Mega BS-RoFormer — 53 stems",
    "BS-RoFormer — Voix homme / femme (aufr33)": "BS-RoFormer — Male / female vocals (aufr33)",
    "BS-RoFormer-1296 — Voix / instrumental (viperx)": "BS-RoFormer-1296 — Vocals / instrumental (viperx)",
    "Mémoire": "Memory",
    "Standard — segments de 20 s (GPU ≥ 16 Go)": "Standard — 20 s segments (GPU ≥ 16 GB)",
    "Économe — segments de 10 s (GPU 10–12 Go)": "Economy — 10 s segments (GPU 10–12 GB)",
    "Minimal — segments de 5 s (GPU 6–8 Go / CPU)": "Minimal — 5 s segments (GPU 6–8 GB / CPU)",
    "Standard — segments de 8 s (GPU ≥ 8 Go)": "Standard — 8 s segments (GPU ≥ 8 GB)",
    "Économe — segments de 4 s (GPU 4–6 Go)": "Economy — 4 s segments (GPU 4–6 GB)",
    "Minimal — segments de 2 s (GPU / CPU)": "Minimal — 2 s segments (GPU / CPU)",
    "Masquer les instruments absents du morceau": "Hide instruments absent from the track",
    "Fichier": "File",
    "⬇ Télécharger le modèle (1,4 Go)": "⬇ Download the model (1.4 GB)",
    "⬇ Télécharger le modèle ({s:.2f} Go)": "⬇ Download the model ({s:.2f} GB)",
    "Choisir un .ckpt…": "Choose a .ckpt…",
    "Les 53 stems se recouvrent (ex. « Voix » = voix lead + chœurs). REMASTRA active par défaut un "
    "jeu principal (voix, batterie, basse, guitare, piano, cordes, synthé + résidu) qui se somme "
    "exactement au mix ; les instruments détaillés sont en mute. GPU NVIDIA 16 Go recommandé, très "
    "lent sur CPU.":
        "The 53 stems overlap (e.g. \"Vocals\" = lead vocal + backing vocals). By default REMASTRA "
        "enables a main set (vocals, drums, bass, guitar, piano, strings, synth + residual) that "
        "adds up exactly to the mix; detailed instruments are muted. NVIDIA GPU with 16 GB "
        "recommended, very slow on CPU.",
    "Sépare la voix en deux stems, voix d'homme et voix de femme, plus un résidu "
    "instrumental calculé (la somme redonne exactement le mix). Idéal pour isoler "
    "un duo, un chœur mixte ou un featuring. Fonctionne mieux sur une piste déjà "
    "isolée (voix) ou un mix où la voix domine.":
        "Splits the voice into two stems, male and female, plus a computed instrumental "
        "residual (the sum adds up exactly to the mix). Ideal for isolating a duet, a mixed "
        "choir or a featured vocal. Works best on an already-isolated vocal track or a mix "
        "where the voice dominates.",
    "Modèle de référence pour la séparation voix / instrumental (SDR 12,96 dB), "
    "très utilisé par la communauté UVR. Le résidu instrumental se somme "
    "exactement au mix. Excellent choix par défaut pour isoler une voix lead.":
        "Benchmark model for vocals / instrumental separation (SDR 12.96 dB), widely used by "
        "the UVR community. The instrumental residual adds up exactly to the mix. An "
        "excellent default choice for isolating a lead vocal.",
    "Aucun stem pour l'instant.": "No stems yet.",
    "Écouter le mix des stems": "Play the stem mix",
    "STEMS PRINCIPAUX — se somment au mix d'origine": "MAIN STEMS — add up to the original mix",
    "Exporter": "Export",
    "Modèle MVSep Mega 53 stems": "MVSep Mega 53-stem model",
    "Checkpoint (*.ckpt *.pth *.pt)": "Checkpoint (*.ckpt *.pth *.pt)",
    "Téléchargement du modèle 53 stems": "Downloading the 53-stem model",
    "Téléchargement du modèle {m}": "Downloading the {m} model",
    "Le modèle MVSep Mega 53 stems (1,4 Go) n'est pas encore installé.\n\nLe télécharger "
    "maintenant ?": "The MVSep Mega 53-stem model (1.4 GB) is not installed yet.\n\nDownload it now?",
    "Le modèle {m} ({s:.2f} Go) n'est pas encore installé.\n\nLe télécharger maintenant ?":
        "The {m} model ({s:.2f} GB) is not installed yet.\n\nDownload it now?",
    "Séparation 53 stems (BS-RoFormer)": "53-stem separation (BS-RoFormer)",
    "Séparation ({m})": "Separation ({m})",
    "Séparation des stems": "Stem separation",
    "Exporter le stem": "Export stem",
    "Voix (homme)": "Male vocals", "Voix (femme)": "Female vocals",
    "Instrumental (résidu)": "Instrumental (residual)",
    # Stems Demucs
    "Batterie": "Drums", "Basse": "Bass", "Guitare": "Guitar", "Piano": "Piano",
    "Cordes": "Strings", "Nappes / Pads": "Pads", "Synthés": "Synths", "Autres": "Other",
    "Fond": "Background",
    # Stems 53
    "Accordéon": "Accordion", "Guitare acoustique": "Acoustic guitar", "Chœurs": "Backing vocals",
    "Banjo": "Banjo", "Basson": "Bassoon", "Cloches": "Bells", "Cordes frottées": "Bowed strings",
    "Cuivres": "Brass", "Violoncelle": "Cello", "Clarinette": "Clarinet", "Congas": "Congas",
    "Piano numérique": "Digital piano", "Dobro": "Dobro", "Contrebasse": "Double bass",
    "Guitare électrique": "Electric guitar", "Flûte": "Flute", "Cor": "French horn",
    "Glockenspiel": "Glockenspiel", "Harmonica": "Harmonica", "Harpe": "Harp",
    "Clavecin": "Harpsichord", "Charleston (hi-hat)": "Hi-hat", "Claviers": "Keys",
    "Grosse caisse": "Kick", "Voix lead": "Lead vocal", "Mandoline": "Mandolin",
    "Marimba": "Marimba", "Hautbois": "Oboe", "Orgue": "Organ", "Percussions": "Percussion",
    "Saxophone": "Saxophone", "Sitar": "Sitar", "Caisse claire": "Snare", "Synthé": "Synth",
    "Tambourin": "Tambourine", "Timbales": "Timpani", "Toms": "Toms", "Triangle": "Triangle",
    "Trombone": "Trombone", "Trompette": "Trumpet", "Tuba": "Tuba", "Ukulélé": "Ukulele",
    "Alto": "Viola", "Violon": "Violin", "Vents": "Wind", "Carillon à vent": "Wind chimes",
    "Bois": "Woodwinds", "Autres (résidu)": "Other (residual)",
    # --- Remaster -------------------------------------------------------------
    "Remastérisation IA": "AI Remastering",
    "Analyse intelligente → EQ corrective à phase linéaire, compression multibande adaptative, "
    "excitateur harmonique, image stéréo M/S, glue, normalisation loudness et limiteur true-peak x4.":
        "Smart analysis → linear-phase corrective EQ, adaptive multiband compression, harmonic "
        "exciter, M/S stereo imaging, glue, loudness normalization and 4× true-peak limiter.",
    "Profil": "Profile",
    "Cible loudness": "Loudness target",
    "LUFS perso": "Custom LUFS",
    "Plafond true-peak": "True-peak ceiling",
    "Caractère": "Character",
    "Correction EQ IA": "AI EQ correction",
    "Compression": "Compression",
    "Excitateur / Air": "Exciter / Air",
    "Largeur stéréo": "Stereo width",
    "Basses mono sous": "Mono bass below",
    "Charger une référence…": "Load a reference…",
    "Mastering par référence : REMASTRA reproduit l'équilibre tonal, le loudness et la largeur "
    "d'un morceau de référence.":
        "Reference mastering: REMASTRA matches the tonal balance, loudness and width of a "
        "reference track.",
    "Aucune référence": "No reference",
    "Référence : ": "Reference: ",
    "Morceau de référence": "Reference track",
    "✦  Remastériser avec l'IA": "✦  Remaster with AI",
    "Loudness": "Loudness", "LRA": "LRA", "Crête": "Crest",
    "Spectre avant / après": "Spectrum before / after",
    "Décisions du moteur IA": "AI engine decisions",
    "Personnalisé…": "Custom…",
    "Auto (IA)": "Auto (AI)",
    "Pop / Variété": "Pop",
    "Rock / Metal": "Rock / Metal",
    "Hip-Hop / Trap": "Hip-Hop / Trap",
    "Électro / EDM": "Electronic / EDM",
    "Jazz / Acoustique": "Jazz / Acoustic",
    "Classique / Orchestral": "Classical / Orchestral",
    "Cinéma / Bande originale": "Cinema / Soundtrack",
    "Doublage / Dialogue": "Dubbing / Dialogue",
    "Podcast / Voix": "Podcast / Voice",
    "Profil (auto)": "Profile (auto)",
    "Streaming — Spotify/YouTube (-14 LUFS)": "Streaming — Spotify/YouTube (-14 LUFS)",
    "Apple Music (-16 LUFS)": "Apple Music (-16 LUFS)",
    "CD / Club (-9 LUFS)": "CD / Club (-9 LUFS)",
    "Broadcast EBU R128 (-23 LUFS)": "Broadcast EBU R128 (-23 LUFS)",
    "Cinéma / ATSC A/85 (-24 LKFS)": "Cinema / ATSC A/85 (-24 LKFS)",
    "Podcast (-16 LUFS)": "Podcast (-16 LUFS)",
    "Sans normalisation": "No normalization",
    "Neutre": "Neutral", "Chaleureux": "Warm", "Brillant": "Bright",
    "Vintage analogique": "Analog vintage", "Punchy": "Punchy",
    "avant": "before",
    # --- Export ---------------------------------------------------------------
    "WAV (PCM 16/24 bits, 32 bits float, RF64 > 4 Go) ou FLAC sans perte (16/24 bits). Du mono "
    "jusqu'au Dolby Atmos 9.1.6 et DTS:X avec spatialisation IA, plus un repli Binaural pour "
    "l'écoute au casque.":
        "WAV (16/24-bit PCM, 32-bit float, RF64 > 4 GB) or lossless FLAC (16/24-bit). From mono up "
        "to Dolby Atmos 9.1.6 and DTS:X with AI spatialization, plus a Binaural fold-down for "
        "headphone listening.",
    "Master (Remaster IA)": "Master (AI Remaster)",
    "Voix / audio débruité": "Denoised voice / audio",
    "Mix des stems": "Stem mix",
    "Stems séparés (1 fichier par stem)": "Separate stems (1 file per stem)",
    "Format": "Format",
    "Résolution": "Bit depth",
    "Fréquence": "Sample rate",
    "24 bits": "24-bit", "16 bits": "16-bit", "32 bits float": "32-bit float",
    "Dither TPDF + noise shaping": "TPDF dither + noise shaping",
    "Canaux": "Channels",
    "Mono": "Mono", "Stéréo": "Stereo",
    "Spatialisation objet à partir des stems IA (si disponibles)":
        "Object spatialization from the AI stems (if available)",
    "⬇  Exporter": "⬇  Export",
    "Configuration d'écoute": "Speaker layout",
    "  —  FLAC est limité à 8 canaux : export multi-mono (1 FLAC par canal).":
        "  —  FLAC is limited to 8 channels: multi-mono export (1 FLAC per channel).",
    "1 canal — C": "1 channel — C",
    "2 canaux — master stéréo L/R": "2 channels — L/R stereo master",
    "2 canaux — L/R discrets (broadcast, pistes séparées compatibles)":
        "2 channels — discrete L/R (broadcast, split-track compatible)",
    "3 canaux — L R + LFE (caisson)": "3 channels — L R + LFE (subwoofer)",
    "6 canaux — L R C LFE Ls Rs": "6 channels — L R C LFE Ls Rs",
    "8 canaux — L R C LFE Lrs Rrs Lss Rss": "8 channels — L R C LFE Lrs Rrs Lss Rss",
    "14 canaux — 7.1 + 6 hauteurs (Dolby Atmos bed)": "14 channels — 7.1 + 6 heights (Dolby Atmos bed)",
    "16 canaux — 7.1.6 + Wides (Dolby Atmos bed étendu)":
        "16 channels — 7.1.6 + Wides (extended Dolby Atmos bed)",
    "12 canaux — 7.1 + 4 hauteurs (bed immersif objet DTS:X)":
        "12 channels — 7.1 + 4 heights (DTS:X object-based immersive bed)",
    "2 canaux — repli binaural casque (indices ITD/ILD, spectraux et de décorrélation "
    "simulant profondeur et hauteur ; pas de convolution HRTF mesurée)":
        "2 channels — binaural headphone fold-down (ITD/ILD, spectral and decorrelation "
        "cues simulating depth and height; not measured-HRTF convolution)",
    "Séparez d'abord les stems (étape ③).": "Separate the stems first (step ③).",
    "Dossier de destination des stems": "Destination folder for the stems",
    "Export des stems": "Stem export",
    "Cette source n'est pas encore disponible. Lancez l'étape correspondante ou choisissez une "
    "autre source.": "This source is not available yet. Run the matching step or choose another source.",
    "Exporter": "Export",
    "REMASTRA — Export terminé": "REMASTRA — Export complete",
    # --- Widgets --------------------------------------------------------------
    "Glissez un fichier audio ou vidéo ici": "Drag an audio or video file here",
    "Spectre — lancez une analyse": "Spectrum — run an analysis",
    "Avant": "Before", "Après": "After", "EQ IA": "AI EQ",
    "▲ AVANT": "▲ FRONT", "Horizontal": "Horizontal", "Hauteur": "Height",
    # --- Messages moteur (exacts) ---------------------------------------------
    "Format non supporté et FFmpeg introuvable.": "Unsupported format and FFmpeg not found.",
    "Filtre anti-rumble (HPF 60 Hz, 24 dB/oct)": "Rumble filter (HPF 60 Hz, 24 dB/oct)",
    "Aucun ronflement secteur détecté": "No mains hum detected",
    "Anti-clic": "De-click",
    "Demucs : isolation de la voix…": "Demucs: isolating the voice…",
    "De-esser": "De-esser",
    "Spatialisation objet à partir des stems IA…": "Object spatialization from the AI stems…",
    "Upmix spectral direct/ambiance…": "Spectral direct/ambience upmix…",
    "Analyse du signal source…": "Analyzing the source signal…",
    "Source déjà dense : compression allégée": "Source already dense: lighter compression",
    "Aigus pauvres détectés (source ancienne ?) : excitateur renforcé":
        "Dull highs detected (old source?): exciter boosted",
    "Image très étroite : pas d'élargissement artificiel (quasi-mono)":
        "Very narrow image: no artificial widening (near-mono)",
    "Compresseur de bus « glue » 2:1 (attaque 30 ms)": "\"Glue\" bus compressor 2:1 (30 ms attack)",
    "Téléchargement incomplet du modèle.": "Incomplete model download.",
    "Chargement du modèle BS-RoFormer 53 stems (681 M paramètres)…":
        "Loading the BS-RoFormer 53-stem model (681M parameters)…",
    "Assemblage des stems…": "Assembling stems…",
    "Séparation étendue : cordes / nappes / synthés…": "Extended separation: strings / pads / synths…",
    "Le moteur IA Demucs n'est pas installé.\nLancez install.bat (ou : pip install torch demucs).":
        "The Demucs AI engine is not installed.\nRun install.bat (or: pip install torch demucs).",
    "Dépendances manquantes : pip install torch einops beartype rotary-embedding-torch":
        "Missing dependencies: pip install torch einops beartype rotary-embedding-torch",
    "aucune": "none",
    "grave": "low", "médium": "mid", "aigu": "high",
    "Traitement": "Processing",
    "Sélection : {a} → {b}": "Selection: {a} → {b}",
    "Profil : sélection {a}s – {b}s": "Profile: selection {a}s – {b}s",
    "{t} terminé ✔": "{t} complete ✔",
    "Erreur : ": "Error: ",
    "Calcul : {d}": "Compute: {d}",
    "Go": "GB",
    "canal(aux)": "channel(s)",
    "Modèle absent — {f}": "Model missing — {f}",
    "INSTRUMENTS DÉTAILLÉS ({n}) — se recouvrent avec les stems principaux, en mute par défaut":
        "DETAILED INSTRUMENTS ({n}) — overlap with the main stems, muted by default",
    "{n} fichier(s) exporté(s) :": "{n} file(s) exported:",
    "Choisissez votre langue": "Choose your language",
    "Initialisation des moteurs IA…": "Initializing AI engines…",
    "REMASTRA — format {l} ({n} canaux)": "REMASTRA — {l} format ({n} channels)",
    "Ordre des canaux :": "Channel order:",
}

# Motifs dynamiques (regex FR → gabarit EN ; les groupes sont re-traduits avec tr())
_PATTERNS = [(re.compile(a, re.S), b) for a, b in [
    (r"Anti-ronflement (\d+) Hz \+ 8 harmoniques", "De-hum {0} Hz + 8 harmonics"),
    (r"Rééchantillonnage (\d+) → (\d+) Hz \(SoX VHQ\)", "Resampling {0} → {1} Hz (SoX VHQ)"),
    (r"Rééchantillonnage (\d+) → (\d+) Hz pour le traitement", "Resampling {0} → {1} Hz for processing"),
    (r"FLAC limité à 8 canaux → export multi-mono \((\d+) fichiers\)",
     "FLAC limited to 8 channels → multi-mono export ({0} files)"),
    (r"✔ Écrit : (.*)", "✔ Written: {0}"),
    (r"Limiteur true-peak : gain (\S+) dB, réduction max (\S+) dB",
     "True-peak limiter: gain {0} dB, max reduction {1} dB"),
    (r"Contenu détecté : (\S+) → profil « (.+) »", "Detected content: {0} → \"{1}\" profile"),
    (r"Référence : (\S+) LUFS, corrélation (\S+)", "Reference: {0} LUFS, correlation {1}"),
    (r"Cible loudness : (.+), plafond (\S+) dBTP", "Loudness target: {0}, ceiling {1} dBTP"),
    (r"EQ phase linéaire : (.*)", "Linear-phase EQ: {0}"),
    (r"Compression multibande : intensité (\d+)% \(3 bandes 120 Hz / 5 kHz\)",
     "Multiband compression: {0}% amount (3 bands 120 Hz / 5 kHz)"),
    (r"Saturation analogique « (.+) »", "Analog saturation \"{0}\""),
    (r"Stéréo M/S : largeur (\d+)%, basses mono < (\d+) Hz",
     "M/S stereo: width {0}%, mono bass < {1} Hz"),
    (r"Résultat : (\S+) LUFS, (\S+) dBTP, LRA (\S+) LU", "Result: {0} LUFS, {1} dBTP, LRA {2} LU"),
    (r"Téléchargement du modèle MVSep Mega 53 stems \(≈ (\S+) Go\)…",
     "Downloading the MVSep Mega 53-stem model (≈ {0} GB)…"),
    (r"Téléchargement du modèle (.+) \(≈ (\S+) Go\)…", "Downloading the {0} model (≈ {1} GB)…"),
    (r"Modèle enregistré : (.*)", "Model saved: {0}"),
    (r"Modèle 53 stems introuvable\. (.*)", "53-stem model not found. Download it or choose the .ckpt file."),
    (r"Modèle « (.+) » introuvable\. Téléchargez-le ou choisissez le fichier (\S+)\.",
     "Model \"{0}\" not found. Download it or choose the {1} file."),
    (r"Chargement du modèle (.+)…", "Loading the {0} model…"),
    (r"Inférence BS-RoFormer sur (\S+) — (\d+) segments de (\d+) s…",
     "BS-RoFormer inference on {0} — {1} segments of {2} s…"),
    (r"Inférence BS-RoFormer \((.+)\) sur (\S+) — (\d+) segments de (\d+) s…",
     "BS-RoFormer inference ({0}) on {1} — {2} segments of {3} s…"),
    (r"(\d+) instruments détectés, (\d+) absents masqués", "{0} instruments detected, {1} absent hidden"),
    (r"(\d+) instruments détectés", "{0} instruments detected"),
    (r"(\d+) stem\(s\) détecté\(s\), (\d+) absents masqués", "{0} stem(s) detected, {1} absent hidden"),
    (r"(\d+) stem\(s\) détecté\(s\)", "{0} stem(s) detected"),
    (r"Chargement du modèle IA (\S+) \(téléchargé au premier lancement\)…",
     "Loading AI model {0} (downloaded on first use)…"),
    (r"Inférence Demucs sur (\S+) \(shifts=(\d+)\)…", "Demucs inference on {0} (shifts={1})…"),
    (r"── (.+) ──", "── {0} ──"),
    (r"(.+) terminé ✔", "{0} complete ✔"),
    (r"Chargement de (.+)", "Loading {0}"),
    (r"Export du stem (.+)", "Exporting stem {0}"),
    (r"Erreur : (.*)", "Error: {0}"),
    (r"« (.+) » n'est pas encore disponible\.", "\"{0}\" is not available yet."),
]]
