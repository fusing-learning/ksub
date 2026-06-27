"""Stage: Transcribe audio to word-level timestamps."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ksub import config
from ksub.fs import extract_audio, file_nonempty
from ksub.paths import audio_wav_path, resolve_video_input, words_json_path
from ksub.types import Word, WordValidationError, normalize_words, validate_words


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
        "error: no ASR engine available — install mlx-whisper (pip install ksub[mlx]) "
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


def adapt_mlx_segments(segments: list[dict]) -> list[Word]:
    """Flatten mlx-whisper segments into normalized words.json entries."""
    words: list[Word] = []
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


def adapt_whisper_cpp_json(data: dict) -> list[Word]:
    """Convert whisper.cpp --output-json-full to words.json."""
    words: list[Word] = []
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


def _transcribe_mlx(wav_path: Path) -> list[Word]:
    import mlx_whisper

    print("Note: mlx-whisper may download ~3 GB model on first run (whisper-large-v3-mlx).")
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="ko",
        word_timestamps=True,
    )
    return adapt_mlx_segments(result["segments"])


def _transcribe_whisper_cpp(wav_path: Path, workdir: Path) -> list[Word]:
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


def run_transcribe(workdir: Path, *, force: bool = False) -> None:
    """Run the transcribe stage: audio.wav → words.json."""
    words_path = words_json_path(workdir)
    wav_path = audio_wav_path(workdir)
    if file_nonempty(words_path) and not force:
        return

    if not file_nonempty(wav_path):
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

    try:
        normalize_words(words)
        validate_words(words)
    except WordValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
