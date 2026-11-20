"""Tests for Silero VAD span math (no ONNX / no model download)."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from ksub import config, vad

from ksub.vad import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    flatten_clip_timestamps,
    pad_and_merge_spans,
    remap_concat_words,
    speech_spans_from_probs,
    write_speech_wav,
)

WINDOW_S = WINDOW_SAMPLES / SAMPLE_RATE


def test_speech_spans_from_probs_silence_only():
    probs = [0.1] * 20
    spans = speech_spans_from_probs(
        probs,
        window_s=WINDOW_S,
        audio_duration=20 * WINDOW_S,
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_s=0.05,
        min_silence_s=0.1,
    )
    assert spans == []


def test_speech_spans_from_probs_middle_region():
    # 4 windows silence, 10 speech, 10 silence
    probs = [0.1] * 4 + [0.9] * 10 + [0.1] * 10
    spans = speech_spans_from_probs(
        probs,
        window_s=WINDOW_S,
        audio_duration=24 * WINDOW_S,
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_s=0.05,
        min_silence_s=3 * WINDOW_S,
    )
    assert len(spans) == 1
    start, end = spans[0]
    assert start == 4 * WINDOW_S
    assert end == 14 * WINDOW_S


def test_speech_spans_hysteresis_keeps_brief_dip():
    # Dip to 0.4 is below threshold but above neg_threshold — stay in speech.
    probs = [0.1] * 2 + [0.9] * 4 + [0.4] + [0.9] * 4 + [0.1] * 8
    spans = speech_spans_from_probs(
        probs,
        window_s=WINDOW_S,
        audio_duration=len(probs) * WINDOW_S,
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_s=0.05,
        min_silence_s=3 * WINDOW_S,
    )
    assert len(spans) == 1
    assert spans[0][0] == 2 * WINDOW_S
    assert spans[0][1] == 11 * WINDOW_S


def test_speech_spans_splits_on_min_silence():
    probs = [0.9] * 5 + [0.1] * 8 + [0.9] * 5 + [0.1] * 4
    spans = speech_spans_from_probs(
        probs,
        window_s=WINDOW_S,
        audio_duration=len(probs) * WINDOW_S,
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_s=0.05,
        min_silence_s=3 * WINDOW_S,
    )
    assert len(spans) == 2
    assert spans[0] == (0.0, 5 * WINDOW_S)
    assert spans[1][0] == 13 * WINDOW_S


def test_speech_spans_drops_short_blip():
    probs = [0.1] * 4 + [0.9] + [0.1] * 10
    spans = speech_spans_from_probs(
        probs,
        window_s=WINDOW_S,
        audio_duration=len(probs) * WINDOW_S,
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_s=0.1,
        min_silence_s=WINDOW_S,
    )
    assert spans == []


def test_pad_and_merge_clips_and_merges_tiny_gap():
    spans = [(1.0, 2.0), (2.15, 3.0), (5.0, 6.0)]
    merged = pad_and_merge_spans(
        spans,
        audio_duration=6.0,
        pad_s=0.1,
        merge_gap_s=0.2,
    )
    assert merged == [(0.9, 3.1), (4.9, 6.0)]


def test_pad_and_merge_empty():
    assert pad_and_merge_spans([], audio_duration=10.0, pad_s=0.1, merge_gap_s=0.2) == []


def test_flatten_clip_timestamps():
    assert flatten_clip_timestamps([(1.0, 2.5), (4.0, 5.0)]) == [1.0, 2.5, 4.0, 5.0]


def test_remap_concat_words_second_span():
    spans = [(10.0, 20.0), (40.0, 50.0)]
    words = remap_concat_words(
        [
            {"word": "하나", "start": 1.0, "end": 2.0},
            {"word": "둘", "start": 11.0, "end": 12.5},
        ],
        spans,
    )
    assert words == [
        {"word": "하나", "start": 11.0, "end": 12.0},
        {"word": "둘", "start": 41.0, "end": 42.5},
    ]


def _write_silence_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    n_frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * n_frames)


def test_write_speech_wav_concatenates_spans(tmp_path: Path):
    src = tmp_path / "full.wav"
    dest = tmp_path / "speech.wav"
    _write_silence_wav(src, 2.0)
    write_speech_wav(src, dest, [(0.5, 1.0), (1.5, 2.0)])

    with wave.open(str(dest), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 16000  # 0.5s + 0.5s


def _fake_download(payload: bytes):
    def fetch(url, dest):
        Path(dest).write_bytes(payload)
    return fetch


def test_resolve_vad_model_rejects_checksum_mismatch(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("KSUB_VAD_MODEL", raising=False)
    cache = tmp_path / "silero_vad.onnx"
    monkeypatch.setattr(config, "VAD_MODEL_CACHE", str(cache))
    with patch("urllib.request.urlretrieve", _fake_download(b"tampered")):
        with pytest.raises(SystemExit):
            vad.resolve_vad_model()
    assert not cache.exists()
    assert not cache.with_suffix(".onnx.download").exists()
    assert "checksum mismatch" in capsys.readouterr().err


def test_resolve_vad_model_redownloads_stale_cache(tmp_path: Path, monkeypatch):
    import hashlib

    monkeypatch.delenv("KSUB_VAD_MODEL", raising=False)
    good = b"pinned model bytes"
    monkeypatch.setattr(config, "VAD_MODEL_SHA256", hashlib.sha256(good).hexdigest())
    cache = tmp_path / "silero_vad.onnx"
    cache.write_bytes(b"old model")
    monkeypatch.setattr(config, "VAD_MODEL_CACHE", str(cache))
    with patch("urllib.request.urlretrieve", _fake_download(good)):
        assert vad.resolve_vad_model() == cache
    assert cache.read_bytes() == good
