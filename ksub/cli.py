"""CLI entry point and stage orchestration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ksub import config, download, mux, segment, transcribe, translate
from ksub.paths import resolve_srt_path, video_output_path


@dataclass
class RunContext:
    args: argparse.Namespace
    workdir: Path
    langs: list[str]
    force: bool
    engine: str | None
    vad: bool


@dataclass
class RunResult:
    muxed_path: Path | None = None
    cost_line: str | None = None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube → multilingual subtitled video pipeline (Korean source)",
    )
    parser.add_argument("url", nargs="?", help="YouTube URL (optional for workdir-only stages)")
    parser.add_argument("--out", default=".", help="Output directory (default: current directory)")
    parser.add_argument(
        "--langs",
        default="zh",
        metavar="LANGS",
        help='Comma-separated: zh, en, or en,zh. Use "" for no translation.',
    )
    parser.add_argument("--force", action="store_true", help="Re-run stages even if outputs exist")
    parser.add_argument("--mp4", action="store_true", help="Output MP4 instead of MKV")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep source.mkv after mux (default: delete to save disk space)",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--transcribe-only", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    parser.add_argument("--no-mux", action="store_true")
    parser.add_argument("--translate-only", action="store_true")
    parser.add_argument("--mux-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--workdir", help="Explicit work directory path")
    parser.add_argument(
        "--engine-translate",
        choices=["gemini", "ollama"],
        help="Translation engine (overrides config)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Skip Silero VAD; transcribe the full audio including non-speech",
    )

    args = parser.parse_args(argv)

    stage_flags = [
        args.download_only,
        args.transcribe_only,
        args.segment_only,
        args.translate_only,
        args.mux_only,
    ]
    if sum(stage_flags) > 1:
        parser.error("only one stage-only flag may be used at a time")

    return args


def _parse_langs(langs_arg: str) -> list[str]:
    langs = [x.strip() for x in langs_arg.split(",") if x.strip()]
    for lang in langs:
        if lang not in config.TRANSLATION_LANGS:
            print(f"error: unsupported language: {lang}", file=sys.stderr)
            raise SystemExit(1)
    return langs


def _resolve_workdir(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.workdir:
        return Path(args.workdir).resolve()

    if args.url:
        title = download.fetch_title(args.url)
        sanitized = download.sanitize_title(title, args.url)
        return (out_dir / sanitized).resolve()

    print("error: --workdir required when URL is not provided", file=sys.stderr)
    raise SystemExit(1)


def _validate_url_requirement(args: argparse.Namespace, needs_download: bool) -> None:
    if needs_download and not args.url and not args.skip_download:
        print(
            "error: YouTube URL required for download "
            "(omit only with --skip-download or workdir-only stage flags)",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _resolve_context_workdir(args: argparse.Namespace, out_dir: Path) -> Path:
    is_isolated = _isolated_stage(args) is not None

    if args.mux_only and not args.workdir:
        print("error: --mux-only requires --workdir", file=sys.stderr)
        raise SystemExit(1)

    if is_isolated and args.workdir:
        return Path(args.workdir).resolve()

    _validate_url_requirement(args, needs_download=not args.skip_download)
    return _resolve_workdir(args, out_dir)


def _isolated_stage(args: argparse.Namespace) -> str | None:
    # Stage flags are the keys of ISOLATED_STAGES (single source of truth).
    for flag in ISOLATED_STAGES:
        if getattr(args, flag):
            return flag
    return None


def _print_summary(
    workdir: Path,
    *,
    muxed_path: Path | None = None,
    cost_line: str | None = None,
) -> None:
    print("Outputs:")
    print(f"  workdir:  {workdir}")

    ko_path = resolve_srt_path(workdir, "ko")
    if ko_path.is_file():
        print(f"  {ko_path.name}:   {ko_path.resolve()}")

    for cli_lang in config.TRANSLATION_LANGS:
        path = resolve_srt_path(workdir, cli_lang)
        if path.is_file():
            print(f"  {path.name}:   {path.resolve()}")

    if muxed_path and muxed_path.is_file():
        print(f"  video:    {muxed_path.resolve()}")

    if cost_line:
        print(cost_line)

    print(
        "\nSamsung's stock player and Google Photos ignore muxed soft subs — "
        "use VLC or MX Player."
    )


def _run_download(ctx: RunContext) -> RunResult:
    _validate_url_requirement(ctx.args, needs_download=True)
    download.run_download(
        ctx.workdir,
        ctx.args.url,
        force=ctx.force,
        skip_download=ctx.args.skip_download,
    )
    return RunResult()


def _run_transcribe(ctx: RunContext) -> RunResult:
    transcribe.run_transcribe(ctx.workdir, force=ctx.force, vad=ctx.vad)
    return RunResult()


def _run_segment(ctx: RunContext) -> RunResult:
    segment.run_segment(ctx.workdir, force=ctx.force)
    return RunResult()


def _run_translate(ctx: RunContext) -> RunResult:
    if not ctx.langs:
        return RunResult()
    cost_line = translate.run_translate(
        ctx.workdir,
        ctx.langs,
        force=ctx.force,
        engine=ctx.engine,
    )
    return RunResult(cost_line=cost_line)


def _resolve_muxed_path(ctx: RunContext) -> Path:
    muxed_path = mux.run_mux(
        ctx.workdir,
        ctx.langs,
        force=ctx.force,
        mp4=ctx.args.mp4,
        keep_source=ctx.args.keep_source,
    )
    if muxed_path is None:
        return video_output_path(ctx.workdir, mp4=ctx.args.mp4)
    return muxed_path


def _run_mux(ctx: RunContext) -> RunResult:
    return RunResult(muxed_path=_resolve_muxed_path(ctx))


def _run_full_pipeline(ctx: RunContext) -> RunResult:
    download.run_download(
        ctx.workdir,
        ctx.args.url,
        force=ctx.force,
        skip_download=ctx.args.skip_download,
    )
    transcribe.run_transcribe(ctx.workdir, force=ctx.force, vad=ctx.vad)
    segment.run_segment(ctx.workdir, force=ctx.force)

    cost_line = None
    if ctx.langs:
        cost_line = translate.run_translate(
            ctx.workdir,
            ctx.langs,
            force=ctx.force,
            engine=ctx.engine,
        )

    muxed_path = _resolve_muxed_path(ctx) if not ctx.args.no_mux else None

    return RunResult(muxed_path=muxed_path, cost_line=cost_line)


ISOLATED_STAGES: dict[str, Callable[[RunContext], RunResult]] = {
    "download_only": _run_download,
    "transcribe_only": _run_transcribe,
    "segment_only": _run_segment,
    "translate_only": _run_translate,
    "mux_only": _run_mux,
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out).resolve()
    print(f"Output directory: {out_dir}")

    requested_langs = _parse_langs(args.langs)
    workdir = _resolve_context_workdir(args, out_dir)

    ctx = RunContext(
        args=args,
        workdir=workdir,
        langs=requested_langs,
        force=args.force,
        engine=args.engine_translate,
        vad=not args.no_vad,
    )

    stage = _isolated_stage(args)
    if stage is not None:
        result = ISOLATED_STAGES[stage](ctx)
    else:
        result = _run_full_pipeline(ctx)

    _print_summary(
        workdir,
        muxed_path=result.muxed_path,
        cost_line=result.cost_line,
    )


if __name__ == "__main__":
    main()
