<p align="center">
  <img src="remastra/assets/splash.png" alt="REMASTRA" width="720">
</p>

<p align="center">
  <b>AI Denoise · 8 AI Stems · AI Remaster · WAV/FLAC export from mono to 9.1.6</b><br>
  <img src="https://img.shields.io/badge/platform-Windows%2010%2F11-2BD9FE">
  <img src="https://img.shields.io/badge/python-3.10–3.12-784BA0">
  <img src="https://img.shields.io/badge/license-MIT-FF3CAC">
</p>

# REMASTRA — AI Audio Remaster Studio

Audio remastering software for music, dubbing, voice and video soundtracks.

| AI Remaster | AI Stems | Multichannel export |
|---|---|---|
| ![](docs/screenshot-remaster.png) | ![](docs/screenshot-stems.png) | ![](docs/screenshot-export.png) |

> The user interface is currently in French.

## Getting started (Windows 10/11, 64-bit)

1. Download `REMASTRA-vX.Y.Z-windows-x64.zip` from the **[Releases](../../releases)** page and extract it anywhere (e.g. `C:\REMASTRA`).
2. Double-click **`Remastra.exe`**.
3. On first launch, the launcher offers to install the required components: Python 3.11 (through winget if it is missing), the user interface, PyTorch (the NVIDIA GPU build if a card is detected, otherwise the CPU build), Demucs and DeepFilterNet. The AI models are downloaded at the end.
   You will need about 3–6 GB of disk space and an Internet connection.
4. After that, the application opens directly, with no console window.

You can also drag an audio file onto `Remastra.exe` to open it directly.

> Standalone build (no Python needed): after the first install, run `build_standalone.bat`. PyInstaller creates `dist\REMASTRA\REMASTRA.exe`.

## Workflow

| Step | What it does |
|---|---|
| ① Import & Analysis | Opens WAV, FLAC, MP3, OGG, OPUS, M4A, AAC, AIFF, WMA, MP4, MOV and MKV. Measures loudness (EBU R128 / ITU-R BS.1770-4), 4× true peak, LRA, crest factor, stereo correlation and the spectrum. Detects whether the content is voice or music. |
| ② AI Denoise | **DeepFilterNet 3**: neural speech enhancement at 48 kHz. **Demucs voice isolation**: removes music, ambience and crowd noise behind the voice, with an adjustable background level. **Spectral Pro**: Ephraim-Malah MMSE log-spectral estimator with speech presence probability; it can learn the noise profile from a region you select with the mouse. Restoration tools: rumble filter, 50/60 Hz de-hum (8 harmonics), de-click, de-esser. |
| ③ AI Stems | 8 stems: vocals, drums, bass, guitar, piano, strings, pads, synths. Each stem has mute/solo, a gain control and its own export. |
| ④ AI Remaster | Analysis and automatic choice among 10 profiles, linear-phase corrective EQ (8k-tap FIR), adaptive multiband compression, harmonic exciter, analog saturation, M/S stereo imaging with mono bass, glue compression, loudness normalization and a true-peak limiter. Every decision the engine makes is listed. Reference-track mastering is also available. |
| ⑤ Export | **WAV**: 16/24-bit PCM or 32-bit float, WAVE_FORMAT_EXTENSIBLE with a channel mask, automatic RF64 above 4 GB. **FLAC**: 16/24-bit. Channel layouts: **Mono, Stereo, 2.0, 2.1, 5.1, 7.1, 7.1.6, 9.1.6**. Spatialization uses the AI stems as objects, or a spectral direct/ambience upmix. TPDF dither and SoX VHQ resampling (44.1–192 kHz). |

Instant A/B listening (Original / Denoised / Master / Stem mix): click the buttons at the top of the window. The space bar starts and pauses playback.

## Technical notes

* **Stems**: Demucs v4 `htdemucs_6s` separates 6 sources. The strings, pads and synths stems are then extracted from Demucs's "other" track with time-frequency masks (harmonic/percussive, sustain, timbre). They are an **estimate** and are less precise than the 5 stems produced directly by the network. The 8 stems still add up exactly to the original mix.
* **FLAC**: the format is limited to 8 channels. For 7.1.6 (14 channels) and 9.1.6 (16 channels), REMASTRA therefore exports **multi-mono** FLAC: one file per channel, named `01_L`, `02_R`… The WAV export keeps all channels in a single file.
* **Channel order**: standard WAVE order up to 7.1, then Dolby Atmos order for layouts with height channels (`L R C LFE Lrs Rrs Lss Rss [Lw Rw] Ltf Rtf Ltm Rtm Ltr Rtr`). Every multichannel export comes with a `_channels.txt` file.
* **GPU**: NVIDIA (CUDA) cards are used automatically. On CPU, stem separation takes about 1–3× the track length in "High quality" mode.
* If an AI engine is missing or its model cannot be downloaded, REMASTRA falls back to the Spectral Pro engine. Engine status is shown in the bottom-left corner.

## Project structure

```
REMASTRA/
├── Remastra.exe          Windows launcher (shipped in releases)
├── install.bat           component installer
├── build_standalone.bat  standalone PyInstaller build
├── requirements*.txt
├── remastra/
│   ├── core/  audio_io, analysis, denoise, stems, mastering, spatial, export
│   ├── gui/   app, main_window, widgets, theme, workers
│   └── assets/ icon.ico, icon.png, splash.png
└── launcher/  C source of the launcher (MinGW-w64)
```

## Development

```bash
git clone <repo-url> && cd REMASTRA
pip install -r requirements.txt torch torchaudio -r requirements-ai.txt
python -m remastra
```

* Build the launcher: `sh launcher/build_launcher.sh` (Linux, MinGW-w64) or `launcher\build_launcher.bat` (Windows, MinGW).
* Publish a release: `git tag v1.0.0 && git push origin v1.0.0`. The GitHub Actions workflow compiles `Remastra.exe` and creates the release with the Windows archive and `RELEASE_NOTES.md`.

## Credits

* [Demucs](https://github.com/facebookresearch/demucs) (Meta AI): source separation
* [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet): speech denoising
* [PySide6 / Qt](https://www.qt.io/qt-for-python), [pedalboard](https://github.com/spotify/pedalboard), [python-soxr](https://github.com/dofuuz/python-soxr), NumPy, SciPy

## License

MIT, see [LICENSE](LICENSE). The Demucs (MIT) and DeepFilterNet (MIT/Apache-2.0) models are downloaded from their official repositories.
