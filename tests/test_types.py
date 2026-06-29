"""Tests for words.json cleanup helpers."""

from __future__ import annotations

from ksub.types import clean_whisper_hallucinations, sanitize_cue_text


def test_sanitize_cue_text_collapses_stuck_like_cue():
    like = "like" * 50
    text, changed = sanitize_cue_text(like)
    assert changed is True
    assert text == "…"


def test_sanitize_cue_text_keeps_normal_cue():
    text, changed = sanitize_cue_text("안녕하세요")
    assert changed is False
    assert text == "안녕하세요"


def test_clean_whisper_hallucinations_removes_stuck_like_token():
    like = "like" * 50
    words = [
        {"word": "안녕", "start": 0.0, "end": 0.5},
        {"word": like, "start": 10.0, "end": 11.0},
        {"word": "다음", "start": 20.0, "end": 20.5},
    ]
    warnings = clean_whisper_hallucinations(words)

    assert len(words) == 2
    assert words[0]["word"] == "안녕"
    assert words[1]["word"] == "다음"
    assert len(warnings) == 1
    assert "removed repetitive hallucination" in warnings[0]
    assert "9.0s until next speech" in warnings[0]


def test_clean_whisper_hallucinations_keeps_korean_laughter():
    words = [
        {"word": "ㅋ" * 60, "start": 0.0, "end": 1.0},
        {"word": "진짜", "start": 1.1, "end": 1.5},
    ]
    warnings = clean_whisper_hallucinations(words)

    assert len(words) == 2
    assert warnings == []


def test_clean_whisper_hallucinations_keeps_short_repetition():
    words = [{"word": "like" * 3, "start": 0.0, "end": 0.5}]
    warnings = clean_whisper_hallucinations(words)

    assert len(words) == 1
    assert warnings == []