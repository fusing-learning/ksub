"""CLI entry point and stage orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ksub import download, mux, segment, transcribe, translate
from ksub.paths import resolve_srt_path, srt_path, video_output_path


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


def _print_summary(
    workdir: Path,
    langs: list[str],
    *,
    muxed_path: Path | None = None,
    translation_ran: bool = False,
) -> None:
    print("Outputs:")
    print(f"  workdir:  {workdir}")

    ko_path = resolve_srt_path(workdir, "ko")
    if ko_path.is_file():
        print(f"  ko.srt:   {ko_path.resolve()}")

    zh_path = resolve_srt_path(workdir, "zh")
    if "zh" in langs or zh_path.is_file():
        if zh_path.is_file():
            print(f"  zh.srt:   {zh_path.resolve()}")

    en_path = resolve_srt_path(workdir, "en")
    if "en" in langs or en_path.is_file():
        if en_path.is_file():
            print(f"  en.srt:   {en_path.resolve()}")

    if muxed_path and muxed_path.is_file():
        print(f"  video:    {muxed_path.resolve()}")

    if translation_ran:
        cost_line = translate.get_cost_line()
        if cost_line:
            print(cost_line)

    print(
        "\nSamsung's stock player and Google Photos ignore muxed soft subs — "
        "use VLC or MX Player."
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out).resolve()
    print(f"Output directory: {out_dir}")

    requested_langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    for lang in requested_langs:
        if lang not in ("zh", "en"):
            print(f"error: unsupported language: {lang}", file=sys.stderr)
            raise SystemExit(1)

    is_isolated = any([
        args.download_only,
        args.transcribe_only,
        args.segment_only,
        args.translate_only,
        args.mux_only,
    ])

    if args.mux_only and not args.workdir:
        print("error: --mux-only requires --workdir", file=sys.stderr)
        raise SystemExit(1)

    if is_isolated and args.mux_only:
        workdir = Path(args.workdir).resolve()
    elif is_isolated and args.workdir:
        workdir = Path(args.workdir).resolve()
    else:
        _validate_url_requirement(args, needs_download=not args.skip_download)
        workdir = _resolve_workdir(args, out_dir)

    force = args.force
    engine = args.engine_translate
    translation_ran = False
    muxed_path: Path | None = None

    if args.download_only:
        _validate_url_requirement(args, needs_download=True)
        download.run_download(workdir, args.url, force=force, skip_download=args.skip_download)
        _print_summary(workdir, requested_langs)
        return

    if args.transcribe_only:
        transcribe.run_transcribe(workdir, force=force)
        _print_summary(workdir, requested_langs)
        return

    if args.segment_only:
        segment.run_segment(workdir, force=force)
        _print_summary(workdir, requested_langs)
        return

    if args.translate_only:
        if requested_langs:
            translate.run_translate(workdir, requested_langs, force=force, engine=engine)
            translation_ran = True
        _print_summary(workdir, requested_langs, translation_ran=translation_ran)
        return

    if args.mux_only:
        muxed_path = mux.run_mux(
            workdir,
            requested_langs,
            force=force,
            mp4=args.mp4,
            url=args.url,
            keep_source=args.keep_source,
        )
        if muxed_path is None:
            muxed_path = video_output_path(workdir, mp4=args.mp4)
        _print_summary(workdir, requested_langs, muxed_path=muxed_path)
        return

    # Full pipeline or --no-mux
    download.run_download(
        workdir,
        args.url,
        force=force,
        skip_download=args.skip_download,
    )
    transcribe.run_transcribe(workdir, force=force)
    segment.run_segment(workdir, force=force)

    if requested_langs:
        translate.run_translate(workdir, requested_langs, force=force, engine=engine)
        translation_ran = True

    if not args.no_mux:
        muxed_path = mux.run_mux(
            workdir,
            requested_langs,
            force=force,
            mp4=args.mp4,
            url=args.url,
            keep_source=args.keep_source,
        )
        if muxed_path is None:
            muxed_path = video_output_path(workdir, mp4=args.mp4)

    _print_summary(
        workdir,
        requested_langs,
        muxed_path=muxed_path if not args.no_mux else None,
        translation_ran=translation_ran,
    )


if __name__ == "__main__":
    main()