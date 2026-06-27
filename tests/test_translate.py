"""Tests for translation ID validation and newline preservation."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import srt

from ksub import config
from ksub.translate import (
    MAX_RETRIES,
    TranslationSession,
    _ids_match,
    translate_cues,
    translate_srt,
)


def test_ids_match():
    batch = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    good = [{"id": 2, "translation": "b"}, {"id": 1, "translation": "a"}]
    bad = [{"id": 1, "translation": "a"}]
    assert _ids_match(batch, good)
    assert not _ids_match(batch, bad)


def test_translate_cues_preserves_newlines(tmp_path: Path):
    ko_path = tmp_path / "ko.srt"
    cues = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="첫 줄\n둘째 줄"),
        srt.Subtitle(index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="단일"),
    ]
    with open(ko_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(cues))

    session = TranslationSession(engine="gemini")

    def fake_gemini(sess, batch, system_prompt, context=""):
        return [
            {
                "id": item["id"],
                "translation": item["text"]
                .replace("첫 줄", "第一行")
                .replace("둘째 줄", "第二行")
                .replace("단일", "单行"),
            }
            for item in batch
        ]

    with patch("ksub.translate._call_gemini", side_effect=fake_gemini):
        with patch("ksub.translate.check_gemini_prerequisites"):
            out_path = tmp_path / "zh.srt"
            translate_srt(
                str(ko_path),
                config.get_lang("zh"),
                str(out_path),
                session,
            )

    with open(out_path, encoding="utf-8") as f:
        result = list(srt.parse(f.read()))

    assert len(result) == 2
    assert result[0].index == 1
    assert result[0].start == cues[0].start
    assert result[0].end >= cues[0].end
    assert "\n" in result[0].content
    assert result[1].content


def test_translate_cues_retries_on_id_mismatch():
    batch = [{"id": 1, "text": "안녕"}]
    session = TranslationSession(engine="gemini")
    calls = []

    def fake_call(sess, batch, system_prompt, context):
        calls.append(1)
        if len(calls) == 1:
            return [{"id": 99, "translation": "wrong"}]
        return [{"id": 1, "translation": "你好"}]

    with patch("ksub.translate._call_engine", side_effect=fake_call):
        with patch("ksub.translate.check_gemini_prerequisites"):
            result = translate_cues(batch, config.get_lang("zh").system_prompt, session)

    assert result == [{"id": 1, "translation": "你好"}]
    assert len(calls) == 2


def test_translate_cues_reports_json_failure_after_retries(capsys):
    batch = [{"id": 1, "text": "안녕"}]
    session = TranslationSession(engine="gemini")
    calls = []

    def fake_call(sess, batch, system_prompt, context):
        calls.append(1)
        raise json.JSONDecodeError("bad json", "not json", 0)

    with patch("ksub.translate._call_engine", side_effect=fake_call):
        with patch("ksub.translate.check_gemini_prerequisites"):
            with pytest.raises(SystemExit):
                translate_cues(batch, config.get_lang("zh").system_prompt, session)

    captured = capsys.readouterr()
    assert len(calls) == MAX_RETRIES
    assert "translation JSON parse failed" in captured.err
    assert "translation ID mismatch" not in captured.err
