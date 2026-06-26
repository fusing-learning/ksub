"""Stage: Transcribe audio to word-level timestamps."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ksub import config


def check_prerequisites() -> str:
    """Return engine name: 'mlx' or 'whisper_cpp'. Exit if neither available."""
    try:
        import mlx_whisper  # noqa: F401
        return "mlx"
    except ImportError:
        pass

    result = subprocess.run(["which", "whisper-cli"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return "whisper_cpp"

    print(
        "error: no ASR engine available — install mlx-whisper (pip install mlx-whisper) "
        "or whisper-cpp (brew install whisper-cpp)",
        file=sys.stderr,
    )
    raise SystemExit(1)


def resolve_whisper_cpp_model() -> Path:
    """Resolve whisper.cpp model path."""
    env_path = os.environ.get("WHISPER_CPP_MODEL")
    if env_path:
        path = Path(os.path.expanduser(env_path))
        if path.is_file():
            return path

    path = Path(os.path.expanduser(config.WHISPER_CPP_MODEL))
    if path.is_file():
        return path

    print(
        "error: whisper.cpp model not found. Install with:\n"
        "  mkdir -p ~/.cache/whisper\n"
        "  curl -L -o ~/.cache/whisper/ggml-large-v3.bin \\\n"
        "    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin\n"
        "  export WHISPER_CPP_MODEL=~/.cache/whisper/ggml-large-v3.bin",
        file=sys.stderr,
    )
    raise SystemExit(1)


def adapt_mlx_segments(segments: list[dict]) -> list[dict]:
    """Flatten mlx-whisper segments into normalized words.json entries."""
    words: list[dict] = []
    for segment in segments:
        for entry in segment.get("words", []):
            word = entry.get("word", "")
            if word:
                word = word.lstrip()
            if not word or not word.strip():
                continue
            start = float(entry["start"])
            end = entry.get("end")
            if end is None:
                end = start + 0.01
            else:
                end = float(end)
            if start >= end:
                continue
            words.append({"word": word, "start": start, "end": end})
    words.sort(key=lambda w: w["start"])
    return words


def adapt_whisper_cpp_json(data: dict) -> list[dict]:
    """Convert whisper.cpp --output-json-full to words.json."""
    words: list[dict] = []
    for segment in data.get("transcription", []):
        entries = segment.get("words")
        if entries is None:
            entries = [
                t for t in segment.get("tokens", [])
                if not t.get("is_special", False)
            ]
        for entry in entries:
            word = entry.get("word") or entry.get("text", "")
            if word:
                word = word.lstrip()
            if not word or not word.strip():
                continue
            start = float(entry["t0"]) / 100.0
            end = float(entry["t1"]) / 100.0
            if start >= end:
                continue
            words.append({"word": word, "start": start, "end": end})
    words.sort(key=lambda w: w["start"])
    return words


def validate_words(words: list[dict]) -> None:
    """Validate words.json before segmentation."""
    if len(words) < 1:
        print("error: words.json has no entries", file=sys.stderr)
        raise SystemExit(1)
    for i, w in enumerate(words):
        try:
            float(w["start"])
            float(w["end"])
        except (KeyError, TypeError, ValueError):
            print(f"error: words.json entry {i} has invalid start/end", file=sys.stderr)
            raise SystemExit(1)


def extract_audio(mkv_path: Path, wav_path: Path) -> None:
    """Extract 16 kHz mono WAV from MKV (transcribe-stage convenience)."""
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(mkv_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
            "-y",
        ],
        check=True,
        capture_output=True,
    )


def _transcribe_mlx(wav_path: Path) -> list[dict]:
    import mlx_whisper

    print("Note: mlx-whisper may download ~3 GB model on first run (whisper-large-v3-mlx).")
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="ko",
        word_timestamps=True,
    )
    return adapt_mlx_segments(result["segments"])


def _transcribe_whisper_cpp(wav_path: Path, workdir: Path) -> list[dict]:
    model_path = resolve_whisper_cpp_model()
    output_prefix = workdir / "transcript_raw"
    subprocess.run(
        [
            "whisper-cli",
            "-m",
            str(model_path),
            "-l",
            "ko",
            "-f",
            str(wav_path),
            "--output-json-full",
            "-ml",
            "1",
            "-of",
            str(output_prefix),
        ],
        check=True,
        capture_output=True,
    )
    json_path = Path(f"{output_prefix}.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return adapt_whisper_cpp_json(data)


def _file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_transcribe(workdir: Path, *, force: bool = False) -> None:
    """Run the transcribe stage: audio.wav → words.json."""
    words_path = workdir / "words.json"
    wav_path = workdir / "audio.wav"
    if _file_nonempty(words_path) and not force:
        return

    if not _file_nonempty(wav_path):
        from ksub.paths import resolve_video_input

        video_path = resolve_video_input(workdir)
        if video_path is not None:
            extract_audio(video_path, wav_path)
        else:
            print(f"error: expected {wav_path} or a video file in {workdir}", file=sys.stderr)
            raise SystemExit(1)

    engine = check_prerequisites()
    if engine == "mlx":
        words = _transcribe_mlx(wav_path)
    else:
        words = _transcribe_whisper_cpp(wav_path, workdir)

    validate_words(words)
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)