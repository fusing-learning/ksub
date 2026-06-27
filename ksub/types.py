"""Shared typed contracts between pipeline stages."""

from __future__ import annotations

from typing import TypedDict


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
