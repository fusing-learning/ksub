"""Stage: Segment word timestamps into professional subtitle cues."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import srt

from ksub import config
from ksub.fs import file_nonempty
from ksub.paths import remove_legacy_srt, resolve_srt_path, srt_path, words_json_path
from ksub.types import Word, WordValidationError, normalize_words, validate_words

SENTENCE_END = {".", "?", "!", "。", "？", "！"}


def len_raw(text: str) -> int:
    """Raw character count — every character counts as 1."""
    return len(text)


def len_latin(text: str) -> int:
    """Latin-equivalent count for reading-speed checks."""
    total = 0
    for ch in text:
        total += 2 if _is_cjk_wide(ch) else 1
    return total


def _is_cjk_wide(ch: str) -> bool:
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return True
    if 0x4E00 <= code <= 0x9FFF:
        return True
    if 0x3400 <= code <= 0x4DBF:
        return True
    if 0x3000 <= code <= 0x303F:
        return True
    if 0xFF00 <= code <= 0xFFEF:
        return True
    if 0x3130 <= code <= 0x318F:
        return True
    if 0x1100 <= code <= 0x11FF:
        return True
    return False


@dataclass
class RawCue:
    words: list[Word]
    text: str
    start: float
    end: float


def _join_words(words: list[Word]) -> str:
    return " ".join(w["word"] for w in words)


def _ends_sentence(word: str) -> bool:
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_END


def segment_words(
    words: list[Word],
    *,
    max_chars_per_line: int = config.MAX_CHARS_PER_LINE_CJK,
    max_chars_per_cue: int | None = None,
    reading_speed: float = config.READING_SPEED_CJK,
) -> list[RawCue]:
    """Greedy timestamp-aware segmentation into raw cues."""
    if max_chars_per_cue is None:
        max_chars_per_cue = max_chars_per_line * 2

    cues: list[RawCue] = []
    current: list[Word] = []

    for i, word_entry in enumerate(words):
        if current:
            test_text = _join_words(current + [word_entry])
            if len_raw(test_text) > max_chars_per_cue:
                cues.append(_make_raw_cue(current))
                current = []

        current.append(word_entry)
        cue_end = current[-1]["end"]
        cue_start = current[0]["start"]
        duration = cue_end - cue_start

        should_close = False
        if _ends_sentence(word_entry["word"]) and duration >= config.MIN_CUE_DURATION_BEFORE_SENTENCE_BREAK:
            should_close = True
        if i + 1 < len(words):
            gap = words[i + 1]["start"] - word_entry["end"]
            if gap > config.PAUSE_GAP_SECONDS:
                should_close = True
        if duration >= config.MAX_CUE_DURATION:
            should_close = True

        if should_close:
            cues.append(_make_raw_cue(current))
            current = []

    if current:
        cues.append(_make_raw_cue(current))

    wrapped = [_wrap_cue_text(c, max_chars_per_line) for c in cues]
    return _timing_fix(wrapped, reading_speed, max_chars_per_line)


def _make_raw_cue(words: list[Word]) -> RawCue:
    text = _join_words(words)
    return RawCue(
        words=list(words),
        text=text,
        start=words[0]["start"],
        end=words[-1]["end"],
    )


def _wrap_cue_text(cue: RawCue, max_per_line: int) -> RawCue:
    text = wrap_text_lines(cue.text, max_per_line)
    return RawCue(words=cue.words, text=text, start=cue.start, end=cue.end)


def wrap_text_lines(text: str, max_per_line: int) -> str:
    """Break text into at most two balanced lines at spaces."""
    if "\n" in text:
        lines = text.split("\n")
        if len(lines) <= 2 and all(len_raw(l) <= max_per_line for l in lines):
            return text

    tokens = text.split(" ")
    for token in tokens:
        if len(token) > config.LONG_TOKEN_WARN_CHARS:
            print(
                f"warning: long token ({len(token)} chars): {token[:30]}...",
                file=sys.stderr,
            )

    if len_raw(text) <= max_per_line:
        return text

    best_split: int | None = None
    best_diff = float("inf")
    for split_at in range(1, len(tokens)):
        line1 = " ".join(tokens[:split_at])
        line2 = " ".join(tokens[split_at:])
        if len_raw(line1) > max_per_line or len_raw(line2) > max_per_line:
            continue
        diff = abs(len_raw(line1) - len_raw(line2))
        if diff < best_diff:
            best_diff = diff
            best_split = split_at

    if best_split is not None:
        return " ".join(tokens[:best_split]) + "\n" + " ".join(tokens[best_split:])

    mid = max(1, len(tokens) // 2)
    return " ".join(tokens[:mid]) + "\n" + " ".join(tokens[mid:])


def _reading_speed(cue: RawCue) -> float:
    duration = cue.end - cue.start
    if duration <= 0:
        return float("inf")
    return len_latin(cue.text) / duration


def _needed_duration(cue: RawCue, reading_speed: float) -> float:
    return len_latin(cue.text) / reading_speed


def _max_end_for_cue(cues: list[RawCue], index: int) -> float:
    if index + 1 < len(cues):
        return cues[index + 1].start - config.MIN_CUE_GAP
    return cues[index].start + config.MAX_CUE_DURATION


def _apply_min_duration(cues: list[RawCue]) -> None:
    for i, cue in enumerate(cues):
        if cue.end - cue.start < config.MIN_CUE_DURATION:
            new_end = min(cue.start + config.MIN_CUE_DURATION, _max_end_for_cue(cues, i))
            cue.end = max(cue.end, new_end)


def _enforce_min_gaps(cues: list[RawCue]) -> None:
    for i in range(len(cues) - 1):
        max_end = cues[i + 1].start - config.MIN_CUE_GAP
        if cues[i].end > max_end:
            min_end = cues[i].start + config.MIN_CUE_DURATION
            if min_end > max_end:
                continue
            cues[i].end = max_end
        if cues[i].end <= cues[i].start:
            cues[i].end = cues[i].start + 0.01


def _try_extend_reading_speed(cues: list[RawCue], index: int, reading_speed: float) -> bool:
    cue = cues[index]
    if _reading_speed(cue) <= reading_speed:
        return True
    needed_end = cue.start + _needed_duration(cue, reading_speed)
    max_end = min(_max_end_for_cue(cues, index), cue.start + config.MAX_CUE_DURATION)
    new_end = min(needed_end, max_end)
    if new_end > cue.end:
        cue.end = new_end
    return _reading_speed(cue) <= reading_speed


def _half_can_meet_reading_speed(
    half: RawCue,
    max_end: float,
    reading_speed: float,
) -> bool:
    trial = RawCue(words=half.words, text=half.text, start=half.start, end=half.end)
    cap = min(max_end, trial.start + config.MAX_CUE_DURATION)
    needed_end = trial.start + _needed_duration(trial, reading_speed)
    trial.end = min(max(needed_end, trial.end, trial.start + config.MIN_CUE_DURATION), cap)
    return _reading_speed(trial) <= reading_speed


def _best_split_index(
    cue: RawCue,
    reading_speed: float,
    max_end: float,
) -> int | None:
    if len(cue.words) < 2:
        return None

    best_idx: int | None = None
    best_score = float("inf")
    for split_at in range(1, len(cue.words)):
        left = _make_raw_cue(cue.words[:split_at])
        right = _make_raw_cue(cue.words[split_at:])
        left_max = right.start - config.MIN_CUE_GAP
        right_max = max_end
        if not _half_can_meet_reading_speed(left, left_max, reading_speed):
            continue
        if not _half_can_meet_reading_speed(right, right_max, reading_speed):
            continue
        left_needed = _needed_duration(left, reading_speed)
        right_needed = _needed_duration(right, reading_speed)
        left_natural = left.end - left.start
        right_natural = right.end - right.start
        score = max(0, left_needed - left_natural) + max(0, right_needed - right_natural)
        if score < best_score:
            best_score = score
            best_idx = split_at
    return best_idx


def _split_cue(
    cue: RawCue,
    split_at: int,
    max_per_line: int,
) -> tuple[RawCue, RawCue]:
    left = _wrap_cue_text(_make_raw_cue(cue.words[:split_at]), max_per_line)
    right = _wrap_cue_text(_make_raw_cue(cue.words[split_at:]), max_per_line)
    return left, right


def _timing_fix(
    cues: list[RawCue],
    reading_speed: float,
    max_per_line: int,
) -> list[RawCue]:
    if not cues:
        return cues

    result = [
        RawCue(words=c.words, text=c.text, start=c.start, end=c.end) for c in cues
    ]
    _apply_min_duration(result)
    _enforce_min_gaps(result)

    unresolved: list[int] = []
    i = 0
    safety = 0
    max_ops = len(result) * 20

    while i < len(result):
        safety += 1
        if safety > max_ops:
            break

        cue = result[i]
        if cue.end - cue.start > config.MAX_CUE_DURATION:
            cue.end = cue.start + config.MAX_CUE_DURATION

        if _reading_speed(cue) <= reading_speed:
            i += 1
            continue

        extended = _try_extend_reading_speed(result, i, reading_speed)
        if extended:
            i += 1
            continue

        max_end = min(_max_end_for_cue(result, i), cue.start + config.MAX_CUE_DURATION)
        split_at = _best_split_index(cue, reading_speed, max_end)
        if split_at is not None:
            left, right = _split_cue(cue, split_at, max_per_line)
            result[i : i + 1] = [left, right]
            _apply_min_duration(result)
            continue

        unresolved.append(i + 1)
        i += 1

    _enforce_min_gaps(result)

    if unresolved:
        sample = ", ".join(str(n) for n in unresolved[:5])
        suffix = f" (and {len(unresolved) - 5} more)" if len(unresolved) > 5 else ""
        print(
            f"warning: {len(unresolved)} cue(s) exceed reading speed after split/extend "
            f"(e.g. {sample}{suffix})",
            file=sys.stderr,
        )

    return result


def raw_cues_to_srt(cues: list[RawCue]) -> list[srt.Subtitle]:
    """Convert raw cues to srt.Subtitle objects."""
    subtitles: list[srt.Subtitle] = []
    for idx, cue in enumerate(cues, start=1):
        subtitles.append(
            srt.Subtitle(
                index=idx,
                start=timedelta(seconds=cue.start),
                end=timedelta(seconds=cue.end),
                content=cue.text,
            )
        )
    return subtitles


def extend_cue_for_reading_speed(
    text: str,
    start: float,
    end: float,
    next_start: float | None,
    reading_speed: float,
) -> tuple[float, bool]:
    """Extend cue end toward next cue if reading speed exceeded.

    Returns (new_end, still_exceeds).
    """
    duration = end - start
    if duration <= 0:
        return end, False
    speed = len_latin(text) / duration
    if speed <= reading_speed:
        return end, False
    needed = len_latin(text) / reading_speed
    new_end = start + needed
    if next_start is not None:
        new_end = min(new_end, next_start - config.MIN_CUE_GAP)
    if new_end > end:
        new_duration = new_end - start
        if new_duration > 0 and len_latin(text) / new_duration <= reading_speed:
            return new_end, False
        return new_end, True
    return end, True


def run_segment(workdir: Path, *, force: bool = False) -> None:
    """Run the segment stage: words.json → ko.srt."""
    words_path = words_json_path(workdir)
    ko_path = resolve_srt_path(workdir, "ko")

    if file_nonempty(ko_path) and not force:
        return

    if not file_nonempty(words_path):
        print(f"error: expected {words_path}", file=sys.stderr)
        raise SystemExit(1)

    with open(words_path, encoding="utf-8") as f:
        words: list[Word] = json.load(f)

    try:
        normalize_words(words)
        validate_words(words)
    except WordValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    cues = segment_words(words)
    subtitles = raw_cues_to_srt(cues)

    out_path = srt_path(workdir, "ko")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))
    remove_legacy_srt(workdir, "ko")
