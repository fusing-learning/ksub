"""Stage: Mux video and subtitle tracks without re-encoding."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ksub import config
from ksub.fs import file_nonempty
from ksub.paths import (
    SOURCE_VIDEO,
    remove_legacy_srt,
    remove_legacy_video,
    resolve_srt_path,
    resolve_video_input,
    source_video_path,
    video_output_path,
)


def check_prerequisites() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("error: ffmpeg not found — install with: brew install ffmpeg", file=sys.stderr)
        raise SystemExit(1)


def _remux_temp_path(output_path: Path) -> Path:
    """Temporary sibling used when remuxing an existing final output in place."""
    return output_path.with_name(f".{output_path.stem}.remux.tmp{output_path.suffix}")


def _subprocess_stderr(exc: subprocess.CalledProcessError) -> str:
    if exc.stderr is None:
        return ""
    if isinstance(exc.stderr, bytes):
        return exc.stderr.decode(errors="replace").strip()
    return str(exc.stderr).strip()


def run_mux(
    workdir: Path,
    langs: list[str],
    *,
    force: bool = False,
    mp4: bool = False,
    keep_source: bool = False,
) -> Path | None:
    """Mux video with subtitle tracks; output <basename>.mkv beside matching .srt files."""
    check_prerequisites()

    video_path = resolve_video_input(workdir)
    ko_path = resolve_srt_path(workdir, "ko")

    if video_path is None:
        print(f"error: expected {SOURCE_VIDEO} or muxed video in {workdir}", file=sys.stderr)
        raise SystemExit(1)
    if not file_nonempty(ko_path):
        print(f"error: expected {ko_path}", file=sys.stderr)
        raise SystemExit(1)

    for cli_lang in langs:
        try:
            config.get_lang(cli_lang)
        except ValueError:
            print(f"error: unsupported mux language: {cli_lang}", file=sys.stderr)
            raise SystemExit(1)
        lang_srt = resolve_srt_path(workdir, cli_lang)
        if not file_nonempty(lang_srt):
            print(f"error: expected {lang_srt} for --langs {cli_lang}", file=sys.stderr)
            raise SystemExit(1)

    output_path = video_output_path(workdir, mp4=mp4)

    if file_nonempty(output_path) and not force:
        return None

    actual_output_path = output_path
    replacing_existing_output = video_path == output_path
    if replacing_existing_output:
        actual_output_path = _remux_temp_path(output_path)
        if actual_output_path.exists():
            actual_output_path.unlink()

    track_langs = ["ko"]
    for cli_lang in config.MUX_TRACK_ORDER[1:]:
        if cli_lang in langs:
            track_langs.append(cli_lang)

    cmd = ["ffmpeg", "-i", str(video_path), "-i", str(ko_path)]
    for cli_lang in track_langs[1:]:
        cmd.extend(["-i", str(resolve_srt_path(workdir, cli_lang))])

    cmd.extend(["-map", "0:v", "-map", "0:a"])
    for idx in range(len(track_langs)):
        cmd.extend(["-map", str(idx + 1)])

    cmd.extend(["-c:v", "copy", "-c:a", "copy"])
    if mp4:
        cmd.extend(["-c:s", "mov_text"])
    else:
        cmd.extend(["-c:s", "srt"])

    for meta_idx, cli_lang in enumerate(track_langs):
        lang = config.get_lang(cli_lang)
        cmd.extend([
            f"-metadata:s:s:{meta_idx}",
            f"language={lang.ffmpeg_lang}",
            f"-metadata:s:s:{meta_idx}",
            f"title={lang.ffmpeg_title}",
        ])

    cmd.extend([str(actual_output_path), "-y"])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = _subprocess_stderr(exc)
        if stderr:
            print(stderr, file=sys.stderr)
        else:
            print(f"error: ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        if replacing_existing_output and actual_output_path.exists():
            actual_output_path.unlink()
        raise
    except Exception:
        if replacing_existing_output and actual_output_path.exists():
            actual_output_path.unlink()
        raise

    if replacing_existing_output:
        try:
            actual_output_path.replace(output_path)
        except OSError as exc:
            print(
                f"error: failed to replace {output_path} with remuxed output: {exc}",
                file=sys.stderr,
            )
            print(f"remuxed output kept at: {actual_output_path}", file=sys.stderr)
            raise

    source_path = source_video_path(workdir)
    if (
        not keep_source
        and source_path.is_file()
        and video_path == source_path
        and output_path != source_path
    ):
        source_path.unlink()

    remove_legacy_video(workdir, mp4=mp4)
    for cli_lang in config.MUX_TRACK_ORDER:
        remove_legacy_srt(workdir, cli_lang)
    return output_path
