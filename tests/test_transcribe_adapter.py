"""Tests for whisper.cpp and mlx-whisper JSON adapters."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ksub import transcribe
from ksub.transcribe import adapt_mlx_segments, adapt_whisper_cpp_json
from ksub.types import WordValidationError, normalize_words, validate_words


@pytest.fixture(autouse=True)
def _vad_deps_present():
    """Run-transcribe tests mock VAD itself; don't depend on onnxruntime being installed."""
    with patch("ksub.transcribe.vad_deps_available", return_value=True):
        yield

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_words_coerces_string_timestamps():
    words = [{"word": "안녕", "start": "0.5", "end": "1.5"}]
    normalize_words(words)
    validate_words(words)
    assert words[0]["start"] == 0.5
    assert words[0]["end"] == 1.5
    assert isinstance(words[0]["start"], float)
    assert isinstance(words[0]["end"], float)


def test_validate_words_rejects_missing_word_text():
    with pytest.raises(WordValidationError):
        validate_words([{"start": 0.0, "end": 1.0}])


def test_validate_words_rejects_empty_word_text():
    with pytest.raises(WordValidationError):
        validate_words([{"word": "  ", "start": 0.0, "end": 1.0}])


def test_validate_words_rejects_non_numeric_timestamp():
    with pytest.raises(WordValidationError):
        normalize_words([{"word": "안녕", "start": "abc", "end": 1.0}])


def test_validate_words_rejects_non_positive_duration():
    with pytest.raises(WordValidationError):
        validate_words([{"word": "안녕", "start": 1.0, "end": 1.0}])


def test_whisper_cpp_adapter():
    with open(FIXTURES / "whisper_cpp_sample.json", encoding="utf-8") as f:
        data = json.load(f)

    words = adapt_whisper_cpp_json(data)
    validate_words(words)

    assert len(words) == 3
    assert words[0] == {"word": "안녕하세요", "start": 0.0, "end": 1.2}
    assert words[1] == {"word": "여러분", "start": 1.3, "end": 2.5}
    assert words[2] == {"word": "반갑습니다", "start": 3.0, "end": 5.0}


def test_mlx_adapter_normalization():
    with open(FIXTURES / "mlx_sample.json", encoding="utf-8") as f:
        data = json.load(f)

    words = adapt_mlx_segments(data["segments"])
    validate_words(words)

    assert words == [
        {"word": "안녕하세요", "start": 0.0, "end": 1.2},
        {"word": "여러분", "start": 1.3, "end": 2.5},
        {"word": "missing_end", "start": 4.0, "end": 4.01},
    ]


def test_adapters_produce_same_schema():
    with open(FIXTURES / "whisper_cpp_sample.json", encoding="utf-8") as f:
        cpp_data = json.load(f)
    with open(FIXTURES / "mlx_sample.json", encoding="utf-8") as f:
        mlx_data = json.load(f)

    cpp_words = adapt_whisper_cpp_json(cpp_data)
    mlx_words = adapt_mlx_segments(mlx_data["segments"])

    for words in (cpp_words, mlx_words):
        assert isinstance(words, list)
        for entry in words:
            assert set(entry.keys()) == {"word", "start", "end"}
            assert isinstance(entry["word"], str) and entry["word"]
            assert isinstance(entry["start"], float)
            assert isinstance(entry["end"], float)
            assert entry["start"] < entry["end"]


def test_run_transcribe_passes_vad_clips_to_mlx(tmp_path: Path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 64)

    words = [{"word": "안녕", "start": 1.0, "end": 1.5}]
    with patch("ksub.transcribe.collect_speech_spans", return_value=[(1.0, 2.0), (4.0, 5.0)]):
        with patch("ksub.transcribe.check_prerequisites", return_value="mlx"):
            with patch("ksub.transcribe._transcribe_mlx", return_value=words) as mlx:
                transcribe.run_transcribe(tmp_path, force=True, vad=True)

    mlx.assert_called_once_with(wav, clip_timestamps=[1.0, 2.0, 4.0, 5.0])
    dumped = json.loads((tmp_path / "words.json").read_text(encoding="utf-8"))
    assert dumped == words


def test_run_transcribe_no_vad_skips_speech_spans(tmp_path: Path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 64)

    words = [{"word": "안녕", "start": 0.0, "end": 0.5}]
    with patch("ksub.transcribe.collect_speech_spans") as collect:
        with patch("ksub.transcribe.check_prerequisites", return_value="mlx"):
            with patch("ksub.transcribe._transcribe_mlx", return_value=words) as mlx:
                transcribe.run_transcribe(tmp_path, force=True, vad=False)

    collect.assert_not_called()
    mlx.assert_called_once_with(wav, clip_timestamps=None)


def test_run_transcribe_empty_vad_falls_back_to_full_audio(tmp_path: Path, capsys):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 64)

    words = [{"word": "안녕", "start": 0.0, "end": 0.5}]
    with patch("ksub.transcribe.collect_speech_spans", return_value=[]):
        with patch("ksub.transcribe.check_prerequisites", return_value="mlx"):
            with patch("ksub.transcribe._transcribe_mlx", return_value=words) as mlx:
                transcribe.run_transcribe(tmp_path, force=True, vad=True)

    mlx.assert_called_once_with(wav, clip_timestamps=None)
    assert "VAD found no speech" in capsys.readouterr().err


def test_run_transcribe_passes_spans_to_whisper_cpp(tmp_path: Path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 64)

    words = [{"word": "안녕", "start": 11.0, "end": 12.0}]
    spans = [(10.0, 20.0), (40.0, 50.0)]
    with patch("ksub.transcribe.collect_speech_spans", return_value=spans):
        with patch("ksub.transcribe.check_prerequisites", return_value="whisper_cpp"):
            with patch("ksub.transcribe._transcribe_whisper_cpp", return_value=words) as cpp:
                transcribe.run_transcribe(tmp_path, force=True, vad=True)

    cpp.assert_called_once_with(wav, tmp_path, speech_spans=spans)


def test_run_transcribe_missing_vad_deps_falls_back(tmp_path: Path, capsys):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 64)

    words = [{"word": "안녕", "start": 0.0, "end": 0.5}]
    with patch("ksub.transcribe.vad_deps_available", return_value=False):
        with patch("ksub.transcribe.collect_speech_spans") as collect:
            with patch("ksub.transcribe.check_prerequisites", return_value="whisper_cpp"):
                with patch("ksub.transcribe._transcribe_whisper_cpp", return_value=words) as cpp:
                    transcribe.run_transcribe(tmp_path, force=True, vad=True)

    collect.assert_not_called()
    cpp.assert_called_once_with(wav, tmp_path, speech_spans=None)
    assert "VAD skipped" in capsys.readouterr().err


def test_run_transcribe_checks_engine_before_vad(tmp_path: Path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 64)

    with patch("ksub.transcribe.collect_speech_spans") as collect:
        with patch("ksub.transcribe.check_prerequisites", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                transcribe.run_transcribe(tmp_path, force=True, vad=True)

    collect.assert_not_called()
