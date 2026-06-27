"""Tests for whisper.cpp and mlx-whisper JSON adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ksub.transcribe import adapt_mlx_segments, adapt_whisper_cpp_json
from ksub.types import WordValidationError, normalize_words, validate_words

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
