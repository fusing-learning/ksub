"""Unit tests for subtitle segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import srt

from ksub import config
from ksub.segment import (
    len_latin,
    len_raw,
    raw_cues_to_srt,
    segment_words,
    wrap_text_lines,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_len_raw_and_len_latin():
    assert len_raw("hello") == 5
    assert len_raw("안녕") == 2
    assert len_latin("hello") == 5
    assert len_latin("안녕") == 4


def test_long_run_on_sentence_splits():
    """A very long run-on sentence should split into multiple cues."""
    words = []
    t = 0.0
    for i in range(30):
        words.append({"word": f"단어{i}", "start": t, "end": t + 0.3})
        t += 0.35

    cues = segment_words(words)
    assert len(cues) > 1
    for cue in cues:
        assert cue.end - cue.start >= config.MIN_CUE_DURATION - 0.01 or cue == cues[-1]
        assert cue.end - cue.start <= config.MAX_CUE_DURATION + 0.01
        assert cue.text.count("\n") <= 1
        for line in cue.text.split("\n"):
            assert len_raw(line) <= config.MAX_CHARS_PER_LINE_CJK


def test_pause_mid_sentence_splits():
    """A long pause mid-sentence should trigger a cue break."""
    words = [
        {"word": "첫번째", "start": 0.0, "end": 0.5},
        {"word": "부분", "start": 0.5, "end": 1.0},
        {"word": "둘째", "start": 2.0, "end": 2.5},
        {"word": "부분", "start": 2.5, "end": 3.0},
    ]
    cues = segment_words(words)
    assert len(cues) == 2


def test_rapid_fire_short_words():
    """Rapid short words should accumulate until a trigger fires."""
    words = [
        {"word": "a", "start": 0.0, "end": 0.2},
        {"word": "b", "start": 0.25, "end": 0.45},
        {"word": "c", "start": 0.5, "end": 0.7},
        {"word": "d", "start": 0.75, "end": 0.95},
        {"word": "e", "start": 1.0, "end": 1.2},
    ]
    cues = segment_words(words)
    assert len(cues) >= 1
    assert " ".join(w["word"] for w in words) in cues[0].text or len(cues) > 1


def test_sentence_end_breaks_after_min_duration():
    words = [
        {"word": "안녕하세요.", "start": 0.0, "end": 1.5},
        {"word": "다음", "start": 1.6, "end": 2.0},
        {"word": "문장", "start": 2.0, "end": 2.5},
    ]
    cues = segment_words(words)
    assert len(cues) >= 2


def test_wrap_text_two_lines():
    text = " ".join(["가나"] * 12)
    wrapped = wrap_text_lines(text, max_per_line=21)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(len_raw(line) <= 21 for line in lines)


def test_reading_speed_extend_or_split():
    """Dense speech should meet reading speed via extend or split."""
    words = []
    t = 0.0
    for i in range(12):
        words.append({"word": f"말{i}", "start": t, "end": t + 0.15})
        t += 0.16

    cues = segment_words(words)
    for cue in cues:
        dur = cue.end - cue.start
        if dur > 0:
            assert len_latin(cue.text) / dur <= config.READING_SPEED_CJK + 0.1


def test_srt_output_format():
    with open(FIXTURES / "words_sample.json", encoding="utf-8") as f:
        words = json.load(f)

    cues = segment_words(words)
    subtitles = raw_cues_to_srt(cues)
    composed = srt.compose(subtitles)

    assert "00:00:00,000" in composed
    parsed = list(srt.parse(composed))
    assert len(parsed) == len(subtitles)
    assert parsed[0].index == 1