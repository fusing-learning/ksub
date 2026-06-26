"""Stage: Mux video and subtitle tracks without re-encoding."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ksub.download import fetch_title_from_mkv, sanitize_title
from ksub.paths import (
    SOURCE_VIDEO,
    remove_legacy_playback_files,
    resolve_srt_path,
    resolve_video_input,
    video_output_path,
)


def check_prerequisites() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("error: ffmpeg not found — install with: brew install ffmpeg", file=sys.stderr)
        raise SystemExit(1)


def resolve_mux_title(workdir: Path, url: str | None) -> str:
    """Resolve title for mux output basename."""
    mkv_path = workdir / "source.mkv"
    if url:
        from ksub.download import fetch_title

        raw = fetch_title(url)
    else:
        raw = fetch_title_from_mkv(mkv_path)
        if not raw:
            raw = workdir.name or "video_unknown"
    return sanitize_title(raw, url)


def _file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_mux(
    workdir: Path,
    langs: list[str],
    *,
    force: bool = False,
    mp4: bool = False,
    url: str | None = None,
    keep_source: bool = False,
) -> Path | None:
    """Mux video with subtitle tracks; output <basename>.mkv beside matching .srt files."""
    check_prerequisites()

    video_path = resolve_video_input(workdir)
    ko_path = resolve_srt_path(workdir, "ko")

    if video_path is None:
        print(f"error: expected {SOURCE_VIDEO} or muxed video in {workdir}", file=sys.stderr)
        raise SystemExit(1)
    if not _file_nonempty(ko_path):
        print(f"error: expected {ko_path}", file=sys.stderr)
        raise SystemExit(1)

    for lang in langs:
        if lang not in ("zh", "en"):
            print(f"error: unsupported mux language: {lang}", file=sys.stderr)
            raise SystemExit(1)
        lang_srt = resolve_srt_path(workdir, lang)
        if not _file_nonempty(lang_srt):
            print(f"error: expected {lang_srt} for --langs {lang}", file=sys.stderr)
            raise SystemExit(1)

    output_path = video_output_path(workdir, mp4=mp4)

    if _file_nonempty(output_path) and not force:
        return None

    cmd = ["ffmpeg", "-i", str(video_path), "-i", str(ko_path)]

    track_langs: list[str] = ["ko"]
    if "zh" in langs:
        cmd.extend(["-i", str(resolve_srt_path(workdir, "zh"))])
        track_langs.append("zh")
    if "en" in langs:
        cmd.extend(["-i", str(resolve_srt_path(workdir, "en"))])
        track_langs.append("en")

    cmd.extend(["-map", "0:v", "-map", "0:a"])
    for idx in range(len(track_langs)):
        cmd.extend(["-map", str(idx + 1)])

    cmd.extend(["-c:v", "copy", "-c:a", "copy"])
    if mp4:
        cmd.extend(["-c:s", "mov_text"])
    else:
        cmd.extend(["-c:s", "srt"])

    metadata = [
        ("kor", "한국어"),
        ("zho", "中文"),
        ("eng", "English"),
    ]
    meta_idx = 0
    for track in track_langs:
        if track == "ko":
            lang_code, title_text = metadata[0]
        elif track == "zh":
            lang_code, title_text = metadata[1]
        else:
            lang_code, title_text = metadata[2]
        cmd.extend([
            f"-metadata:s:s:{meta_idx}",
            f"language={lang_code}",
            f"-metadata:s:s:{meta_idx}",
            f"title={title_text}",
        ])
        meta_idx += 1

    cmd.extend([str(output_path), "-y"])

    subprocess.run(cmd, check=True, capture_output=True)

    source_path = workdir / SOURCE_VIDEO
    if (
        not keep_source
        and source_path.is_file()
        and video_path == source_path
        and output_path != source_path
    ):
        source_path.unlink()

    remove_legacy_playback_files(workdir)
    return output_path