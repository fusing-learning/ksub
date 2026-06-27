"""Stage: Translate Korean subtitles to target languages."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import srt

from ksub import config
from ksub.fs import file_nonempty
from ksub.paths import remove_legacy_srt, resolve_srt_path, srt_path
from ksub.segment import (
    extend_cue_for_reading_speed,
    len_raw,
    wrap_text_lines,
)
from ksub.types import CueInput, TranslationResult

MAX_RETRIES = 2


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0


@dataclass
class TranslationSession:
    """Per-run translation state (client, token usage, cost summary)."""

    engine: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_line: str | None = None
    prerequisites_checked: bool = False
    _gemini_client: object | None = field(default=None, repr=False)

    def gemini_client(self):
        if self._gemini_client is None:
            import os
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise SystemExit("GEMINI_API_KEY is not set — required for translation.")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client


def check_gemini_prerequisites() -> None:
    """Verify google-genai is importable."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        print(
            "error: google-genai not installed — pip install ksub[gemini]",
            file=sys.stderr,
        )
        raise SystemExit(1)


def check_ollama_prerequisites() -> None:
    """Verify Ollama is running and model is available."""
    url = f"{config.OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        print(f"error: Ollama not reachable at {config.OLLAMA_BASE_URL}", file=sys.stderr)
        raise SystemExit(1)

    models = [m.get("name", "") for m in data.get("models", [])]
    model_base = config.OLLAMA_MODEL.split(":")[0]
    if not any(m.startswith(model_base) or m == config.OLLAMA_MODEL for m in models):
        print(f"error: Ollama model not found — run: ollama pull {config.OLLAMA_MODEL}", file=sys.stderr)
        raise SystemExit(1)


def _build_user_content(batch: list[CueInput], context: str) -> str:
    user_content = ""
    if context:
        user_content += (
            "Previous context (do NOT translate — for register reference only):\n"
            f"{context}\n\n"
        )
    user_content += f"Translate these cues:\n{json.dumps(batch, ensure_ascii=False)}"
    return user_content


def _read_srt(path: Path) -> list[srt.Subtitle]:
    with open(path, encoding="utf-8-sig") as f:
        return list(srt.parse(f.read()))


def _write_srt(path: Path, cues: list[srt.Subtitle]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(cues))


def _ids_match(batch: list[CueInput], result: list[TranslationResult]) -> bool:
    input_ids = {item["id"] for item in batch}
    try:
        output_ids = {item["id"] for item in result}
    except (KeyError, TypeError):
        return False
    return input_ids == output_ids


def _is_rate_limited(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 503):
        return True
    err_str = str(exc).lower()
    return "429" in err_str or "503" in err_str


def _call_gemini(
    session: TranslationSession,
    batch: list[CueInput],
    system_prompt: str,
    context: str = "",
) -> list[TranslationResult]:
    user_content = _build_user_content(batch, context)

    resp = session.gemini_client().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[{"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_content}]}],
        config={"response_mime_type": "application/json"},
    )

    usage = resp.usage_metadata
    if usage is not None:
        session.token_usage.input += usage.prompt_token_count or 0
        session.token_usage.output += usage.candidates_token_count or 0

    return json.loads(resp.text)


def _call_ollama(
    batch: list[CueInput],
    system_prompt: str,
    context: str = "",
) -> list[TranslationResult]:
    user_content = _build_user_content(batch, context)

    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "format": "json",
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())

    content = data["message"]["content"]
    return json.loads(content)


def _check_engine_prerequisites(session: TranslationSession) -> None:
    if session.prerequisites_checked:
        return
    if session.engine == "gemini":
        check_gemini_prerequisites()
    elif session.engine == "ollama":
        check_ollama_prerequisites()
    else:
        print(f"error: unknown translation engine: {session.engine}", file=sys.stderr)
        raise SystemExit(1)
    session.prerequisites_checked = True


def _call_engine(
    session: TranslationSession,
    batch: list[CueInput],
    system_prompt: str,
    context: str,
) -> list[TranslationResult]:
    if session.engine == "ollama":
        return _call_ollama(batch, system_prompt, context)
    return _call_gemini(session, batch, system_prompt, context)


def _translate_batch_with_retry(
    session: TranslationSession,
    batch: list[CueInput],
    system_prompt: str,
    context: str,
    batch_num: int,
) -> list[TranslationResult]:
    last_error: Exception | None = None
    last_result: list[TranslationResult] | None = None
    last_json_error: json.JSONDecodeError | None = None

    for attempt in range(MAX_RETRIES):
        try:
            last_result = _call_engine(session, batch, system_prompt, context)
            if _ids_match(batch, last_result):
                return last_result
            print(
                f"Translation batch {batch_num}: ID mismatch, "
                f"retrying ({attempt + 1}/{MAX_RETRIES})",
                file=sys.stderr,
            )
        except json.JSONDecodeError as exc:
            last_json_error = exc
            print(
                f"Translation batch {batch_num}: JSON parse error, "
                f"retrying ({attempt + 1}/{MAX_RETRIES})",
                file=sys.stderr,
            )
            last_error = None
        except Exception as exc:
            if _is_rate_limited(exc) and attempt < MAX_RETRIES - 1:
                time.sleep(5)
                continue
            last_error = exc
            raise

    if last_json_error is not None and last_result is None:
        print(
            f"error: translation JSON parse failed after {MAX_RETRIES} attempts",
            file=sys.stderr,
        )
        raise SystemExit(1)

    input_ids = sorted(item["id"] for item in batch)
    try:
        output_ids = sorted(item["id"] for item in last_result) if last_result else []
    except Exception:
        output_ids = []
    print(
        f"error: translation ID mismatch — input IDs: {input_ids}, returned: {output_ids}",
        file=sys.stderr,
    )
    if last_error:
        raise last_error
    raise SystemExit(1)


def translate_cues(
    cues: list[CueInput],
    system_prompt: str,
    session: TranslationSession,
) -> list[TranslationResult]:
    """Translate cue dicts; returns [{"id": N, "translation": "..."}, ...]."""
    _check_engine_prerequisites(session)

    all_results: list[TranslationResult] = []
    context_lines: list[str] = []

    for batch_start in range(0, len(cues), config.BATCH_SIZE):
        batch = cues[batch_start : batch_start + config.BATCH_SIZE]
        batch_num = batch_start // config.BATCH_SIZE + 1
        context = "\n".join(context_lines[-config.TRANSLATION_CONTEXT_LINES :])
        result = _translate_batch_with_retry(
            session, batch, system_prompt, context, batch_num
        )
        all_results.extend(result)
        for item in result:
            context_lines.append(item["translation"])

    return all_results


def _rewrap_translation(
    text: str,
    lang: config.Lang,
    start: float,
    end: float,
    next_start: float | None,
) -> tuple[str, float, bool]:
    lines = text.split("\n")
    if len(lines) <= 2 and all(len_raw(line) <= lang.max_chars_per_line for line in lines):
        wrapped = text
    else:
        flat = text.replace("\n", " ")
        wrapped = wrap_text_lines(flat, lang.max_chars_per_line)
        if wrapped.count("\n") > 1:
            wrapped = "\n".join(wrapped.split("\n")[:2])

    new_end, exceeds = extend_cue_for_reading_speed(
        wrapped, start, end, next_start, lang.reading_speed
    )
    return wrapped, new_end, exceeds


def _finalize_cost_line(session: TranslationSession) -> None:
    if session.engine == "ollama":
        session.cost_line = (
            f"Translation: local (Ollama/{config.OLLAMA_MODEL}), no API cost"
        )
    elif session.token_usage.input or session.token_usage.output:
        cost = (
            session.token_usage.input * config.GEMINI_RATE_INPUT / 1_000_000
            + session.token_usage.output * config.GEMINI_RATE_OUTPUT / 1_000_000
        )
        session.cost_line = (
            f"Translation: {session.token_usage.input:,} input tokens, "
            f"{session.token_usage.output:,} output tokens → est. ${cost:.2f} "
            f"(rates in config; actual billing may differ)"
        )
    else:
        session.cost_line = "Translation: token usage unavailable"


def translate_srt(
    ko_srt_path: str,
    lang: config.Lang,
    out_path: str,
    session: TranslationSession,
) -> None:
    """Parse ko.srt, translate, re-wrap, and write output SRT."""
    ko_path = Path(ko_srt_path)
    out = Path(out_path)
    subtitles = _read_srt(ko_path)

    cue_dicts: list[CueInput] = [
        {"id": sub.index, "text": sub.content} for sub in subtitles
    ]
    translations = translate_cues(cue_dicts, lang.system_prompt, session)
    trans_map = {item["id"]: item["translation"] for item in translations}

    speed_warnings: list[int] = []
    for i, sub in enumerate(subtitles):
        translation = trans_map[sub.index]
        if translation.count("\n") > 1:
            print(
                f"warning: cue {sub.index} translation has more than one line break",
                file=sys.stderr,
            )

        start_sec = sub.start.total_seconds()
        end_sec = sub.end.total_seconds()
        next_start = subtitles[i + 1].start.total_seconds() if i + 1 < len(subtitles) else None

        wrapped, new_end, exceeds = _rewrap_translation(
            translation, lang, start_sec, end_sec, next_start
        )
        sub.content = wrapped
        if new_end > end_sec:
            sub.end = srt.timedelta(seconds=new_end)
        if exceeds:
            speed_warnings.append(sub.index)

    if speed_warnings:
        sample = ", ".join(str(n) for n in speed_warnings[:5])
        suffix = f" (and {len(speed_warnings) - 5} more)" if len(speed_warnings) > 5 else ""
        print(
            f"warning: {len(speed_warnings)} translated cue(s) exceed reading speed after re-wrap "
            f"(e.g. {sample}{suffix})",
            file=sys.stderr,
        )

    _write_srt(out, subtitles)


def run_translate(
    workdir: Path,
    langs: list[str],
    *,
    force: bool = False,
    engine: str | None = None,
) -> str | None:
    """Run translation for each requested CLI lang code. Returns cost summary if any ran."""
    effective_engine = engine or config.TRANSLATION_ENGINE
    session = TranslationSession(engine=effective_engine)
    ko_path = resolve_srt_path(workdir, "ko")

    if not file_nonempty(ko_path):
        print(f"error: expected {ko_path}", file=sys.stderr)
        raise SystemExit(1)

    ran_any = False
    for cli_lang in langs:
        out_path = srt_path(workdir, cli_lang)
        if file_nonempty(out_path) and not force:
            continue

        lang = config.get_lang(cli_lang)
        translate_srt(str(ko_path), lang, str(out_path), session)
        remove_legacy_srt(workdir, cli_lang)
        ran_any = True

    if ran_any:
        _finalize_cost_line(session)
        return session.cost_line
    return None
