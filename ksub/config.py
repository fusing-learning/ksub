"""All tuneable constants for the ksub pipeline."""

from __future__ import annotations

from dataclasses import dataclass

# Translation engine
TRANSLATION_ENGINE = "gemini"  # "gemini" | "ollama"
GEMINI_MODEL = "gemini-3.5-flash-lite"
# Gemini 3.x uses thinking levels, not budgets. "minimal" keeps JSON output
# reliable and thinking-token cost (billed as output) near zero.
GEMINI_THINKING_LEVEL = "minimal"
GEMINI_RATE_INPUT = 0.30  # USD/1M input tokens (estimate only)
GEMINI_RATE_OUTPUT = 2.50  # USD/1M output tokens (estimate only)

# Ollama (used only when TRANSLATION_ENGINE = "ollama")
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# whisper.cpp fallback model
WHISPER_CPP_MODEL = "~/.cache/whisper/ggml-large-v3.bin"

# Subtitle line limits
MAX_CHARS_PER_LINE = 42  # Latin (en.srt)
MAX_CHARS_PER_LINE_CJK = 21  # Korean/Chinese (ko.srt, zh.srt)

# Cue timing
MAX_CUE_DURATION = 7.0
MIN_CUE_DURATION = 1.0
MIN_CUE_GAP = 0.080

# Reading speed (Latin-equivalent chars/sec)
READING_SPEED_LATIN = 17
READING_SPEED_CJK = 8.5

# Segmentation thresholds
MIN_CUE_DURATION_BEFORE_SENTENCE_BREAK = 1.0
PAUSE_GAP_SECONDS = 0.5
LONG_TOKEN_WARN_CHARS = 20

# Whisper repetitive hallucinations (e.g. stuck "likelikelike..." on applause/noise)
REPETITIVE_HALLUCINATION_MIN_CHARS = 40
REPETITIVE_HALLUCINATION_MIN_REPEATS = 8
REPETITIVE_HALLUCINATION_GAP_WARN_SECONDS = 3.0

# Translation batching
BATCH_SIZE = 80
TRANSLATION_CONTEXT_LINES = 5

# Gemini transient errors (503 high demand, 429 rate limit)
TRANSLATION_API_MAX_RETRIES = 6
TRANSLATION_API_RETRY_BASE_SECONDS = 5
TRANSLATION_API_RETRY_MAX_SECONDS = 60


@dataclass(frozen=True)
class Lang:
    """Language profile for mux metadata, translation prompts, and re-wrap."""

    cli: str
    max_chars_per_line: int
    reading_speed: float
    ffmpeg_lang: str
    ffmpeg_title: str
    # Translation system prompt; empty for the source language (never translated).
    system_prompt: str = ""


_ZH_PROMPT = (
    "Translate the Korean subtitle lines to Simplified Chinese (简体中文), "
    "mainland standard usage. "
    'Return ONLY a JSON array of objects: [{"id": N, "translation": "..."}, ...]. '
    "Match the input IDs exactly. Natural conversational register suitable for subtitles. "
    "Preserve line breaks (\\n) exactly. "
    "No commentary, no markdown fences, no merging or splitting of lines."
)
_EN_PROMPT = (
    "Translate the Korean subtitle lines to English. "
    'Return ONLY a JSON array of objects: [{"id": N, "translation": "..."}, ...]. '
    "Match the input IDs exactly. Natural conversational register suitable for subtitles. "
    "Preserve line breaks (\\n) exactly. "
    "No commentary, no markdown fences, no merging or splitting of lines."
)

# Insertion order is the mux track order (source first, then targets).
LANGS: dict[str, Lang] = {
    "ko": Lang(
        cli="ko",
        max_chars_per_line=MAX_CHARS_PER_LINE_CJK,
        reading_speed=READING_SPEED_CJK,
        ffmpeg_lang="kor",
        ffmpeg_title="한국어",
    ),
    "zh": Lang(
        cli="zh",
        max_chars_per_line=MAX_CHARS_PER_LINE_CJK,
        reading_speed=READING_SPEED_CJK,
        ffmpeg_lang="zho",
        ffmpeg_title="中文",
        system_prompt=_ZH_PROMPT,
    ),
    "en": Lang(
        cli="en",
        max_chars_per_line=MAX_CHARS_PER_LINE,
        reading_speed=READING_SPEED_LATIN,
        ffmpeg_lang="eng",
        ffmpeg_title="English",
        system_prompt=_EN_PROMPT,
    ),
}

# Source language is the first entry; every other entry is a translation target.
SOURCE_LANG = next(iter(LANGS))
TRANSLATION_LANGS = tuple(cli for cli in LANGS if cli != SOURCE_LANG)
MUX_TRACK_ORDER = tuple(LANGS)


def get_lang(cli: str) -> Lang:
    """Look up a language profile by CLI code (ko, zh, en)."""
    try:
        return LANGS[cli]
    except KeyError:
        raise ValueError(f"unsupported language: {cli}") from None
