"""Shared typed contracts between pipeline stages."""

from __future__ import annotations

import re
from typing import TypedDict

from ksub import config

_REPETITIVE_LATIN = re.compile(
    r"^([a-zA-Z]{2,12}?)(?:\1){"
    + str(config.REPETITIVE_HALLUCINATION_MIN_REPEATS - 1)
    + r",}",
)


class Word(TypedDict):
    word: str
    start: float
    end: float


class CueInput(TypedDict):
    id: int
    text: str


class TranslationResult(TypedDict):
    id: int
    translation: str


class WordValidationError(ValueError):
    """Raised when words.json entries do not match the expected contract."""


def normalize_words(words: list[Word]) -> None:
    """Normalize words.json timestamp values in place."""
    for i, w in enumerate(words):
        try:
            w["start"] = float(w["start"])
            w["end"] = float(w["end"])
        except (KeyError, TypeError, ValueError):
            raise WordValidationError(
                f"words.json entry {i} has invalid start/end"
            ) from None


def _repetitive_hallucination_pattern(word: str) -> str | None:
    """Return the repeated Latin pattern if word looks like a stuck ASR loop."""
    stripped = word.strip()
    if len(stripped) < config.REPETITIVE_HALLUCINATION_MIN_CHARS:
        return None

    match = re.match(r"^([^\w]*)(.+)$", stripped, re.UNICODE)
    core = match.group(2) if match else stripped
    found = _REPETITIVE_LATIN.match(core) or _REPETITIVE_LATIN.match(core.lower())
    if found is None:
        return None
    return found.group(1)


def sanitize_cue_text(text: str) -> tuple[str, bool]:
    """Collapse stuck ASR loops in subtitle text to a short ellipsis."""
    pattern = _repetitive_hallucination_pattern(text)
    if pattern is None:
        return text, False

    match = re.match(r"^([^\w]*)(.+)$", text.strip(), re.UNICODE)
    prefix = match.group(1) if match else ""
    return f"{prefix}…", True


def clean_whisper_hallucinations(words: list[Word]) -> list[str]:
    """Drop stuck Latin repetition loops; return human-readable warnings."""
    warnings: list[str] = []
    kept: list[Word] = []

    for i, entry in enumerate(words):
        pattern = _repetitive_hallucination_pattern(entry["word"])
        if pattern is None:
            kept.append(entry)
            continue

        gap_after = (
            words[i + 1]["start"] - entry["end"] if i + 1 < len(words) else None
        )
        msg = (
            f"removed repetitive hallucination at {entry['start']:.1f}s "
            f"({len(entry['word'])} chars, pattern {pattern!r})"
        )
        if (
            gap_after is not None
            and gap_after >= config.REPETITIVE_HALLUCINATION_GAP_WARN_SECONDS
        ):
            msg += (
                f"; {gap_after:.1f}s until next speech — "
                "some dialogue may be missing"
            )
        warnings.append(msg)

    words[:] = kept
    return warnings


def validate_words(words: list[Word]) -> None:
    """Validate words.json before segmentation."""
    if len(words) < 1:
        raise WordValidationError("words.json has no entries")
    for i, w in enumerate(words):
        try:
            start = w["start"]
            end = w["end"]
        except (KeyError, TypeError):
            raise WordValidationError(
                f"words.json entry {i} has invalid start/end"
            ) from None
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise WordValidationError(
                f"words.json entry {i} has invalid start/end"
            )
        if start >= end:
            raise WordValidationError(
                f"words.json entry {i} has invalid start/end"
            )
        word = w.get("word")
        if not isinstance(word, str) or not word.strip():
            raise WordValidationError(f"words.json entry {i} missing 'word' text")
