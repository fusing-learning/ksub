# ksub — YouTube → Multilingual Subtitled Video

One command takes a YouTube URL (or a local video you prepare) and produces:

1. An editable Korean `.srt`
2. An editable Chinese (Simplified) `.srt` (default; English via `--langs en`)
3. A single `.mkv` with all subtitle tracks soft-muxed (toggleable in VLC / MX Player)

---

## What the pipeline does

`ksub` runs five independent stages. Each stage reads files from disk and writes new files, so you can re-run any step after hand-editing subtitles.

```
YouTube URL  (or local source.mkv)
      ↓
① Download      → source.mkv + audio.wav
      ↓
② Transcribe    → words.json
      ↓
③ Segment       → ko.srt
      ↓
④ Translate     → zh.srt (and/or en.srt)
      ↓
⑤ Mux           → subtitled.mkv (inside workdir; source.mkv removed after mux)
```

### Stage 1: Download

- Fetches the video title and creates a workdir: `<out>/<sanitized_title>/`
- Downloads video as `source.mkv` via `yt-dlp`
- Extracts mono 16 kHz `audio.wav` for speech recognition

**Tools:** `yt-dlp`, `ffmpeg`

### Stage 2: Transcribe

- Runs Korean speech recognition with **word-level timestamps**
- Writes `words.json` (flat list of `{word, start, end}`)
- Uses **mlx-whisper** on Apple Silicon (downloads ~3 GB model on first run)
- Falls back to **whisper.cpp** if mlx-whisper is unavailable

### Stage 3: Segment

- Converts word timestamps into broadcast-style Korean subtitles
- Splits on line length (max 2 lines × 21 chars), sentence ends, pauses, and max duration (7 s)
- Adjusts timing for reading speed and min gaps between cues
- Writes `ko.srt`

**No API key needed.**

### Stage 4: Translate

- Reads `ko.srt`, sends Korean text to Gemini (default) or Ollama in batches
- Preserves cue count, timestamps, and line breaks exactly
- Re-wraps translated text to line-length limits
- Writes `zh.srt` and/or `en.srt`

**Requires `GEMINI_API_KEY`** (or a running Ollama instance).

### Stage 5: Mux

- Combines `source.mkv` + subtitle files into one file
- **No video re-encode** — copy only, finishes in seconds
- Track order is always: Korean → Chinese → English
- Output: `<workdir>/subtitled.mkv` (everything in one folder)
- Deletes `source.mkv` after mux to avoid two full-video copies (use `--keep-source` to retain it)

---

## Prerequisites

Install external tools via Homebrew:

```bash
brew install yt-dlp ffmpeg
```

Optional fallback ASR (if mlx-whisper is unavailable):

```bash
brew install whisper-cpp
mkdir -p ~/.cache/whisper
curl -L -o ~/.cache/whisper/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
export WHISPER_CPP_MODEL=~/.cache/whisper/ggml-large-v3.bin
```

## Install

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

After install, `ksub` is available as a terminal command.

## Gemini API key

Translation uses the Gemini Flash API. Set your key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

---

## Usage: YouTube URL

End-to-end (Korean + Chinese subtitles, MKV output):

```bash
ksub "https://www.youtube.com/watch?v=EXAMPLE"
```

English only:

```bash
ksub "https://www.youtube.com/watch?v=EXAMPLE" --langs en
```

Both Chinese and English:

```bash
ksub "https://www.youtube.com/watch?v=EXAMPLE" --langs en,zh
```

Korean subtitles only — no translation, no API key:

```bash
ksub "https://www.youtube.com/watch?v=EXAMPLE" --langs ""
```

Output MKV as MP4 instead:

```bash
ksub "https://www.youtube.com/watch?v=EXAMPLE" --mp4
```

Full pipeline without muxing (stops after translation):

```bash
ksub "https://www.youtube.com/watch?v=EXAMPLE" --no-mux
```

---

## Usage: Local MP4 (or any video file)

There is no `--input` flag in v1. Prepare a workdir manually, then skip the download stage.

### Step 1 — Create workdir and remux your file

The pipeline expects a file named `source.mkv` inside the workdir. Remux your MP4 without re-encoding:

```bash
mkdir -p ~/Videos/my_show
ffmpeg -i /path/to/your/video.mp4 -c copy ~/Videos/my_show/source.mkv
```

### Step 2 — Run the pipeline

```bash
ksub --skip-download --workdir ~/Videos/my_show --out ~/Videos
```

No YouTube URL needed. The pipeline will transcribe → segment → translate → mux automatically.

For MP4 output:

```bash
ksub --skip-download --workdir ~/Videos/my_show --out ~/Videos --mp4
```

### Step-by-step (local file)

```bash
ksub --transcribe-only --workdir ~/Videos/my_show
ksub --segment-only  --workdir ~/Videos/my_show
ksub --translate-only --workdir ~/Videos/my_show
ksub --mux-only      --workdir ~/Videos/my_show --out ~/Videos
```

### Already have a Korean SRT?

Put `source.mkv` and your `ko.srt` in the workdir, then:

```bash
ksub --translate-only --workdir ~/Videos/my_show
ksub --mux-only --workdir ~/Videos/my_show --out ~/Videos
```

---

## Workdir layout

After a full run, the workdir looks like this:

```
<out>/<title>/
  <title>.mkv      ← final video (VLC-friendly name)
  <title>.ko.srt   ← Korean (VLC auto-loads alongside the .mkv)
  <title>.zh.srt   ← Chinese
  <title>.en.srt   ← English (if requested)
  audio.wav        ← intermediate (gitignored)
  words.json       ← intermediate
```

VLC matches external subs when the basename matches: `Episode.mkv` + `Episode.zh.srt`. Soft subs are also embedded in the `.mkv` for track switching.

`source.mkv` exists only until mux completes (deleted by default). Re-mux uses `<title>.mkv` as the video source.

---

## Stage flags

| Flag | Runs | Requires |
|------|------|----------|
| `--download-only` | Download video + extract audio | YouTube URL |
| `--transcribe-only` | ASR → `words.json` | `audio.wav` (or `source.mkv` to extract from) |
| `--segment-only` | `words.json` → `ko.srt` | `words.json` |
| `--translate-only` | `ko.srt` → `zh.srt` / `en.srt` | `ko.srt` |
| `--mux-only` | Combine video + SRTs | `source.mkv`, `ko.srt`, `--workdir` |
| `--no-mux` | Full pipeline minus mux | YouTube URL (or `--skip-download`) |

Other useful flags:

| Flag | Effect |
|------|--------|
| `--force` | Re-run a stage even when output already exists |
| `--skip-download` | Skip `yt-dlp`; use existing `source.mkv` in workdir |
| `--workdir <dir>` | Use an explicit work directory instead of auto-naming |
| `--out <dir>` | Output directory (default: current directory) |
| `--langs zh\|en\|en,zh\|""` | Target languages (`""` = skip translation) |
| `--mp4` | Output `subtitled.mp4` instead of `subtitled.mkv` |
| `--keep-source` | Keep `source.mkv` after mux (default: delete to save space) |
| `--engine-translate gemini\|ollama` | Override translation engine |

Stages are **idempotent** — if output already exists, the stage is skipped unless `--force` is passed.

---

## Edit workflow

1. Run the full pipeline once.
2. Open `ko.srt` or `zh.srt` in any text editor and fix lines.
3. Re-mux to bake changes into the video:

```bash
ksub --mux-only --workdir <path/to/workdir> --out <output-dir>
```

To re-generate subtitles from scratch (e.g. after fixing ASR):

```bash
ksub --segment-only --workdir <dir> --force
ksub --translate-only --workdir <dir> --force
ksub --mux-only --workdir <dir> --force
```

---

## Warnings during a run

Warnings on stderr are **not errors** — the pipeline still completes.

| Warning | Meaning |
|---------|---------|
| `cue(s) exceed reading speed` | Fast dialogue with no gap before the next cue; display time can't be extended further. Spot-check those lines in VLC. |
| `translated cue(s) exceed reading speed` | Chinese text is longer than Korean and hits the same gap limit. |
| `long token (N chars): ...` | Whisper merged words without a space. Safe to fix manually in `ko.srt`. |

Variety shows and rapid banter trigger more of these. Hand-edit the flagged cues if any feel too fast on screen.

---

## Playback on Samsung Android

**Samsung's stock player and Google Photos ignore muxed soft subs — use VLC or MX Player.**

VLC for Android and MX Player handle MKV with embedded SRT tracks and show a subtitle-track selector. Same-named `.srt` files in the same folder also work in VLC/MX without muxing.

---

## Quick reference

| I want to… | Command |
|------------|---------|
| Subtitle a YouTube video | `ksub "<URL>"` |
| Subtitle a local MP4 | Remux to `source.mkv`, then `ksub --skip-download --workdir <dir>` |
| Add English instead of Chinese | `ksub "<URL>" --langs en` |
| Korean only, no API key | `ksub "<URL>" --langs ""` or `--segment-only` |
| Re-mux after editing SRTs | `ksub --mux-only --workdir <dir>` |
| Translate an existing `ko.srt` | `ksub --translate-only --workdir <dir>` |
| Re-run one stage | Add `--force` to that stage's flag |