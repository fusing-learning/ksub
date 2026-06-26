"""All tuneable constants for the ksub pipeline."""

# Translation engine
TRANSLATION_ENGINE = "gemini"  # "gemini" | "ollama"
GEMINI_MODEL = "gemini-2.5-flash"
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

# Translation batching
BATCH_SIZE = 80
TRANSLATION_CONTEXT_LINES = 5