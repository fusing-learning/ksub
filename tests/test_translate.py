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
    _call_engine_with_api_retry,
    _ids_match,
    _is_transient_api_error,
    _loads_translation_json,
    _parse_gemini_json,
    translate_cues,
    translate_srt,
)


def test_is_transient_api_error_detects_gemini_server_error():
    try:
        from google.genai import errors
    except ImportError:
        pytest.skip("google-genai not installed")

    exc = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}},
    )
    assert _is_transient_api_error(exc)


def test_call_engine_with_api_retry_recovers_from_transient_error():
    try:
        from google.genai import errors
    except ImportError:
        pytest.skip("google-genai not installed")

    batch = [{"id": 1, "text": "안녕"}]
    session = TranslationSession(engine="gemini")
    calls = []

    def fake_call(sess, batch, system_prompt, context):
        calls.append(1)
        if len(calls) == 1:
            raise errors.ServerError(
                503,
                {
                    "error": {
                        "code": 503,
                        "message": "high demand",
                        "status": "UNAVAILABLE",
                    }
                },
            )
        return [{"id": 1, "translation": "你好"}]

    with patch("ksub.translate._call_engine", side_effect=fake_call):
        with patch("ksub.translate.time.sleep"):
            result = _call_engine_with_api_retry(
                session, batch, "prompt", "", batch_num=1
            )

    assert result == [{"id": 1, "translation": "你好"}]
    assert len(calls) == 2


def test_loads_translation_json_strips_markdown_fence():
    payload = '```json\n[{"id": 1, "translation": "你好"}]\n```'
    assert _loads_translation_json(payload) == [{"id": 1, "translation": "你好"}]


def test_parse_gemini_json_uses_parsed_field():
    resp = type("Resp", (), {"parsed": [{"id": 1, "translation": "你好"}], "text": None})()
    assert _parse_gemini_json(resp) == [{"id": 1, "translation": "你好"}]


def test_parse_gemini_json_falls_back_to_text():
    payload = '[{"id": 1, "translation": "你好"}]'
    resp = type("Resp", (), {"parsed": None, "text": payload})()
    assert _parse_gemini_json(resp) == [{"id": 1, "translation": "你好"}]


def test_parse_gemini_json_raises_when_text_missing():
    resp = type(
        "Resp",
        (),
        {
            "parsed": None,
            "text": None,
            "candidates": [],
            "prompt_feedback": None,
        },
    )()
    with pytest.raises(json.JSONDecodeError, match="no text"):
        _parse_gemini_json(resp)


def test_call_gemini_uses_minimal_thinking():
    session = TranslationSession(engine="gemini")
    captured: dict = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Resp",
                (),
                {
                    "parsed": [{"id": 1, "translation": "你好"}],
                    "text": None,
                    "usage_metadata": None,
                },
            )()

    session._gemini_client = type("Client", (), {"models": FakeModels()})()

    from ksub.translate import _call_gemini

    result = _call_gemini(session, [{"id": 1, "text": "안녕"}], "prompt")
    assert result == [{"id": 1, "translation": "你好"}]
    assert captured["config"]["thinking_config"] == {
        "thinking_level": config.GEMINI_THINKING_LEVEL,
    }


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
