"""Stage: Translate Korean subtitles to target languages."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import srt

from ksub import config
from ksub.paths import remove_legacy_playback_files, resolve_srt_path, srt_path
from ksub.segment import (
    extend_cue_for_reading_speed,
    len_raw,
    wrap_text_lines,
)

SYSTEM_PROMPTS = {
    "zh-CN": (
        "Translate the Korean subtitle lines to Simplified Chinese (简体中文), "
        "mainland standard usage. "
        'Return ONLY a JSON array of objects: [{"id": N, "translation": "..."}, ...]. '
        "Match the input IDs exactly. Natural conversational register suitable for subtitles. "
        "Preserve line breaks (\\n) exactly. "
        "No commentary, no markdown fences, no merging or splitting of lines."
    ),
    "en": (
        "Translate the Korean subtitle lines to English. "
        'Return ONLY a JSON array of objects: [{"id": N, "translation": "..."}, ...]. '
        "Match the input IDs exactly. Natural conversational register suitable for subtitles. "
        "Preserve line breaks (\\n) exactly. "
        "No commentary, no markdown fences, no merging or splitting of lines."
    ),
}

_client = None
_token_usage = {"input": 0, "output": 0}
_cost_line: str | None = None


def _get_client():
    """Build the Gemini client on first use."""
    global _client
    if _client is None:
        import os
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not set — required for translation (see §1).")
        _client = genai.Client(api_key=api_key)
    return _client


def check_gemini_prerequisites() -> None:
    """Verify google-genai is importable."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        print("error: google-genai not installed — pip install google-genai", file=sys.stderr)
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


def _read_srt(path: Path) -> list[srt.Subtitle]:
    with open(path, encoding="utf-8-sig") as f:
        return list(srt.parse(f.read()))


def _write_srt(path: Path, cues: list[srt.Subtitle]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(cues))


def _ids_match(batch: list[dict], result: list[dict]) -> bool:
    input_ids = {item["id"] for item in batch}
    try:
        output_ids = {item["id"] for item in result}
    except (KeyError, TypeError):
        return False
    return input_ids == output_ids


def _call_gemini(batch: list[dict], target_lang: str, context: str = "") -> list[dict]:
    system_prompt = SYSTEM_PROMPTS[target_lang]
    user_content = ""
    if context:
        user_content += f"Previous context (do NOT translate — for register reference only):\n{context}\n\n"
    user_content += f"Translate these cues:\n{json.dumps(batch, ensure_ascii=False)}"

    resp = _get_client().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[{"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_content}]}],
        config={"response_mime_type": "application/json"},
    )

    usage = resp.usage_metadata
    if usage is not None:
        _token_usage["input"] += usage.prompt_token_count or 0
        _token_usage["output"] += usage.candidates_token_count or 0

    return json.loads(resp.text)


def _call_ollama(batch: list[dict], target_lang: str, context: str = "") -> list[dict]:
    user_content = ""
    if context:
        user_content += f"Previous context (do NOT translate — for register reference only):\n{context}\n\n"
    user_content += f"Translate these cues:\n{json.dumps(batch, ensure_ascii=False)}"

    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[target_lang]},
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


def _call_engine(
    batch: list[dict],
    target_lang: str,
    context: str,
    engine: str,
) -> list[dict]:
    if engine == "ollama":
        return _call_ollama(batch, target_lang, context)
    return _call_gemini(batch, target_lang, context)


def _translate_batch_with_retry(
    batch: list[dict],
    target_lang: str,
    context: str,
    engine: str,
    batch_num: int,
) -> list[dict]:
    last_error: Exception | None = None
    last_result: list[dict] | None = None

    for attempt in range(2):
        try:
            last_result = _call_engine(batch, target_lang, context, engine)
            if _ids_match(batch, last_result):
                return last_result
            print(
                f"Translation batch {batch_num}: ID mismatch, retrying ({attempt + 1}/1)",
                file=sys.stderr,
            )
        except json.JSONDecodeError:
            print(
                f"Translation batch {batch_num}: JSON parse error, retrying ({attempt + 1}/1)",
                file=sys.stderr,
            )
            last_error = None
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "503" in err_str:
                if attempt == 0:
                    time.sleep(5)
                    continue
            last_error = exc
            raise

    input_ids = sorted(item["id"] for item in batch)
    try:
        output_ids = sorted(item["id"] for item in last_result) if last_result else []
    except Exception:
        output_ids = []
    print(f"error: translation ID mismatch — input IDs: {input_ids}, returned: {output_ids}", file=sys.stderr)
    if last_error:
        raise last_error
    raise SystemExit(1)


def translate_cues(
    cues: list[dict],
    target_lang: str,
    *,
    engine: str | None = None,
) -> list[dict]:
    """Translate cue dicts; returns [{"id": N, "translation": "..."}, ...]."""
    effective_engine = engine or config.TRANSLATION_ENGINE

    if effective_engine == "gemini":
        check_gemini_prerequisites()
    elif effective_engine == "ollama":
        check_ollama_prerequisites()
    else:
        print(f"error: unknown translation engine: {effective_engine}", file=sys.stderr)
        raise SystemExit(1)

    all_results: list[dict] = []
    context_lines: list[str] = []

    for batch_start in range(0, len(cues), config.BATCH_SIZE):
        batch = cues[batch_start : batch_start + config.BATCH_SIZE]
        batch_num = batch_start // config.BATCH_SIZE + 1
        context = "\n".join(context_lines[-config.TRANSLATION_CONTEXT_LINES :])
        result = _translate_batch_with_retry(batch, target_lang, context, effective_engine, batch_num)
        all_results.extend(result)
        for item in result:
            context_lines.append(item["translation"])

    return all_results


def _rewrap_translation(
    text: str,
    target_lang: str,
    start: float,
    end: float,
    next_start: float | None,
) -> tuple[str, float, bool]:
    if target_lang == "zh-CN":
        max_per_line = config.MAX_CHARS_PER_LINE_CJK
        reading_speed = config.READING_SPEED_CJK
    else:
        max_per_line = config.MAX_CHARS_PER_LINE
        reading_speed = config.READING_SPEED_LATIN

    lines = text.split("\n")
    if len(lines) <= 2 and all(len_raw(line) <= max_per_line for line in lines):
        wrapped = text
    else:
        flat = text.replace("\n", " ")
        wrapped = wrap_text_lines(flat, max_per_line)
        if wrapped.count("\n") > 1:
            wrapped = "\n".join(wrapped.split("\n")[:2])

    new_end, exceeds = extend_cue_for_reading_speed(
        wrapped, start, end, next_start, reading_speed
    )
    return wrapped, new_end, exceeds


def translate_srt(
    ko_srt_path: str,
    target_lang: str,
    out_path: str,
    *,
    engine: str | None = None,
) -> None:
    """Parse ko.srt, translate, re-wrap, and write output SRT."""
    global _cost_line

    ko_path = Path(ko_srt_path)
    out = Path(out_path)
    subtitles = _read_srt(ko_path)

    cue_dicts = [{"id": sub.index, "text": sub.content} for sub in subtitles]
    translations = translate_cues(cue_dicts, target_lang, engine=engine)
    trans_map = {item["id"]: item["translation"] for item in translations}

    effective_engine = engine or config.TRANSLATION_ENGINE

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
            translation, target_lang, start_sec, end_sec, next_start
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

    if effective_engine == "ollama":
        _cost_line = f"Translation: local (Ollama/{config.OLLAMA_MODEL}), no API cost"
    elif _token_usage["input"] or _token_usage["output"]:
        cost = (
            _token_usage["input"] * config.GEMINI_RATE_INPUT / 1_000_000
            + _token_usage["output"] * config.GEMINI_RATE_OUTPUT / 1_000_000
        )
        _cost_line = (
            f"Translation: {_token_usage['input']:,} input tokens, "
            f"{_token_usage['output']:,} output tokens → est. ${cost:.2f} "
            f"(rates in config; actual billing may differ)"
        )
    else:
        _cost_line = "Translation: token usage unavailable"


def get_cost_line() -> str | None:
    """Return the translation cost summary line, if translation ran."""
    return _cost_line


def reset_cost_tracking() -> None:
    """Reset token usage and cost line between runs."""
    global _cost_line
    _token_usage["input"] = 0
    _token_usage["output"] = 0
    _cost_line = None


def _lang_output_path(workdir: Path, cli_lang: str) -> Path:
    return srt_path(workdir, cli_lang)


def _cli_to_internal(cli_lang: str) -> str:
    return "zh-CN" if cli_lang == "zh" else "en"


def _file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_translate(
    workdir: Path,
    langs: list[str],
    *,
    force: bool = False,
    engine: str | None = None,
) -> None:
    """Run translation for each requested CLI lang code."""
    reset_cost_tracking()
    ko_path = resolve_srt_path(workdir, "ko")

    if not _file_nonempty(ko_path):
        print(f"error: expected {ko_path}", file=sys.stderr)
        raise SystemExit(1)

    ran_any = False
    for cli_lang in langs:
        out_path = _lang_output_path(workdir, cli_lang)
        if _file_nonempty(out_path) and not force:
            continue

        target_lang = _cli_to_internal(cli_lang)
        translate_srt(str(ko_path), target_lang, str(out_path), engine=engine)
        legacy = workdir / ("zh.srt" if cli_lang == "zh" else "en.srt")
        if legacy.is_file() and legacy != out_path:
            legacy.unlink()
        ran_any = True

    if ran_any:
        remove_legacy_playback_files(workdir)

    if not ran_any:
        reset_cost_tracking()