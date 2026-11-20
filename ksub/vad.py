"""Silero VAD: drop non-speech (applause, BGM, silence) before Whisper."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import urllib.request
import wave
from pathlib import Path

from ksub import config
from ksub.types import Word

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # Silero v5 at 16 kHz
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)

SpeechSpan = tuple[float, float]


def speech_spans_from_probs(
    probs: list[float],
    *,
    window_s: float,
    audio_duration: float,
    threshold: float,
    neg_threshold: float,
    min_speech_s: float,
    min_silence_s: float,
) -> list[SpeechSpan]:
    """Turn per-window speech probabilities into [start, end) spans in seconds."""
    triggered = False
    start = 0.0
    temp_end: float | None = None
    spans: list[SpeechSpan] = []

    for i, prob in enumerate(probs):
        t = i * window_s
        if prob >= threshold:
            temp_end = None
            if not triggered:
                triggered = True
                start = t
            continue
        if triggered and prob < neg_threshold:
            if temp_end is None:
                temp_end = t
            if t - temp_end >= min_silence_s:
                if temp_end - start >= min_speech_s:
                    spans.append((start, temp_end))
                triggered = False
                temp_end = None

    if triggered:
        end = audio_duration
        if end - start >= min_speech_s:
            spans.append((start, end))

    return spans


def pad_and_merge_spans(
    spans: list[SpeechSpan],
    *,
    audio_duration: float,
    pad_s: float,
    merge_gap_s: float,
) -> list[SpeechSpan]:
    """Pad each span, clip to the file, and merge overlaps / tiny gaps."""
    if not spans:
        return []

    padded = [
        (max(0.0, start - pad_s), min(audio_duration, end + pad_s))
        for start, end in spans
    ]
    merged = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap_s:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def flatten_clip_timestamps(spans: list[SpeechSpan]) -> list[float]:
    """mlx-whisper clip_timestamps: start, end, start, end, ..."""
    out: list[float] = []
    for start, end in spans:
        out.append(start)
        out.append(end)
    return out


def remap_concat_words(words: list[Word], spans: list[SpeechSpan]) -> list[Word]:
    """Map timestamps from concatenated speech audio back onto the original timeline."""
    remapped: list[Word] = []
    for entry in words:
        start = _concat_time_to_original(entry["start"], spans)
        end = _concat_time_to_original(entry["end"], spans)
        if start >= end:
            continue
        remapped.append({"word": entry["word"], "start": start, "end": end})
    remapped.sort(key=lambda w: w["start"])
    return remapped


def _concat_time_to_original(t: float, spans: list[SpeechSpan]) -> float:
    elapsed = 0.0
    for start, end in spans:
        duration = end - start
        if t <= elapsed + duration:
            return start + (t - elapsed)
        elapsed += duration
    if not spans:
        return t
    return spans[-1][1]


def read_wav_float32(path: Path) -> tuple[object, float]:
    """Load 16 kHz mono PCM16 WAV as float32 in [-1, 1]. Returns (array, duration_s)."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        n_frames = wav.getnframes()
        pcm = wav.readframes(n_frames)

    if channels != 1 or rate != SAMPLE_RATE or width != 2:
        raise ValueError(
            f"{path} must be 16 kHz mono PCM16 (got {rate} Hz, {channels} ch, {width * 8}-bit)"
        )

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / SAMPLE_RATE
    return audio, duration


def write_speech_wav(src: Path, dest: Path, spans: list[SpeechSpan]) -> None:
    """Write only the speech spans from src into dest, concatenated."""
    with wave.open(str(src), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        n_frames = wav.getnframes()
        pcm = wav.readframes(n_frames)

    if channels != 1 or rate != SAMPLE_RATE or width != 2:
        raise ValueError(
            f"{src} must be 16 kHz mono PCM16 (got {rate} Hz, {channels} ch, {width * 8}-bit)"
        )

    frame_bytes = width
    chunks: list[bytes] = []
    for start, end in spans:
        start_i = max(0, int(start * rate))
        end_i = min(n_frames, int(end * rate))
        if end_i > start_i:
            chunks.append(pcm[start_i * frame_bytes : end_i * frame_bytes])

    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(b"".join(chunks))


def vad_deps_available() -> bool:
    """True when numpy and onnxruntime are importable (the ``[vad]`` extra)."""
    return all(importlib.util.find_spec(mod) is not None for mod in ("numpy", "onnxruntime"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_vad_model() -> Path:
    """Return a local Silero VAD ONNX path, downloading on first use."""
    env_path = os.environ.get("KSUB_VAD_MODEL")
    if env_path:
        path = Path(os.path.expanduser(env_path))
        if path.is_file():
            return path
        print(f"error: KSUB_VAD_MODEL is not a file: {path}", file=sys.stderr)
        raise SystemExit(1)

    cache = Path(os.path.expanduser(config.VAD_MODEL_CACHE))
    if cache.is_file():
        if _sha256(cache) == config.VAD_MODEL_SHA256:
            return cache
        print("Cached Silero VAD model does not match the pinned version; re-downloading")

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".onnx.download")
    print(f"Downloading Silero VAD model to {cache}")
    try:
        urllib.request.urlretrieve(config.VAD_MODEL_URL, tmp)
        actual = _sha256(tmp)
        if actual != config.VAD_MODEL_SHA256:
            raise ValueError(
                f"checksum mismatch (expected {config.VAD_MODEL_SHA256}, got {actual})"
            )
        tmp.replace(cache)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(
            f"error: failed to download Silero VAD model: {exc}\n"
            f"  {config.VAD_MODEL_URL}\n"
            "  or set KSUB_VAD_MODEL to a local silero_vad.onnx\n"
            "  or pass --no-vad to transcribe the full audio",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return cache


def collect_speech_spans(wav_path: Path) -> list[SpeechSpan]:
    """Detect speech regions in a 16 kHz mono WAV."""
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        print(
            "error: VAD requires numpy and onnxruntime — pip install 'ksub[vad]'\n"
            "  (or pass --no-vad to transcribe the full audio, including non-speech)",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    audio, duration = read_wav_float32(wav_path)
    model_path = resolve_vad_model()
    probs = _speech_probs(audio, model_path, onnxruntime, np)
    window_s = WINDOW_SAMPLES / SAMPLE_RATE
    spans = speech_spans_from_probs(
        probs,
        window_s=window_s,
        audio_duration=duration,
        threshold=config.VAD_THRESHOLD,
        neg_threshold=config.VAD_NEG_THRESHOLD,
        min_speech_s=config.VAD_MIN_SPEECH_MS / 1000.0,
        min_silence_s=config.VAD_MIN_SILENCE_MS / 1000.0,
    )
    return pad_and_merge_spans(
        spans,
        audio_duration=duration,
        pad_s=config.VAD_PAD_MS / 1000.0,
        merge_gap_s=config.VAD_MERGE_GAP_MS / 1000.0,
    )


def _speech_probs(audio, model_path: Path, onnxruntime, np) -> list[float]:
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 4
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)

    n = int(audio.shape[0])
    pad = (WINDOW_SAMPLES - (n % WINDOW_SAMPLES)) % WINDOW_SAMPLES
    if pad:
        audio = np.pad(audio, (0, pad))

    probs: list[float] = []
    for i in range(0, len(audio), WINDOW_SAMPLES):
        chunk = audio[i : i + WINDOW_SAMPLES].reshape(1, WINDOW_SAMPLES)
        x = np.concatenate([context, chunk], axis=1)
        out, state = session.run(
            ["output", "stateN"],
            {"input": x, "state": state, "sr": sr},
        )
        context = x[:, -CONTEXT_SAMPLES:]
        probs.append(float(out.reshape(-1)[0]))
    return probs
