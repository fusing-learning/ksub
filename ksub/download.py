"""Stage: Download video and extract audio."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

from ksub.fs import extract_audio, file_nonempty
from ksub.paths import audio_wav_path, source_video_path


def check_prerequisites() -> None:
    """Verify yt-dlp and ffmpeg are available."""
    for cmd in ("yt-dlp", "ffmpeg"):
        try:
            subprocess.run(
                [cmd, "--version"] if cmd == "yt-dlp" else [cmd, "-version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"error: {cmd} not found — install with: brew install {cmd}", file=sys.stderr)
            raise SystemExit(1)


def fetch_title(url: str) -> str:
    """Fetch video title via yt-dlp."""
    result = subprocess.run(
        ["yt-dlp", "--print", "title", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"error: failed to fetch title: {result.stderr}", file=sys.stderr)
        raise SystemExit(1)
    return result.stdout.strip()


def sanitize_title(title: str, url: str | None = None) -> str:
    """Sanitize a video title for use as a directory / output basename."""
    cleaned = re.sub(r"[^\w\s\-가-힣一-鿿々〆〤]", "", title, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" ", "_")
    if len(cleaned) > 60:
        cleaned = cleaned[:60]
    if not cleaned:
        if url:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
            cleaned = f"video_{url_hash}"
        else:
            cleaned = "video_unknown"
    return cleaned


def download_video(url: str, mkv_path: Path) -> None:
    """Download video with two-format fallback."""
    primary = [
        "yt-dlp",
        "-f",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mkv",
        "-o",
        str(mkv_path),
        url,
    ]
    result = subprocess.run(primary, capture_output=True, text=True)
    if result.returncode == 0:
        return

    fallback = [
        "yt-dlp",
        "-f",
        "best",
        "--merge-output-format",
        "mkv",
        "-o",
        str(mkv_path),
        url,
    ]
    result = subprocess.run(fallback, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(
            "Download failed. Video may be DRM-protected, private, or live — not supported in v1.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run_download(
    workdir: Path,
    url: str | None,
    *,
    force: bool = False,
    skip_download: bool = False,
) -> None:
    """Run the download stage: source.mkv + audio.wav."""
    check_prerequisites()
    workdir.mkdir(parents=True, exist_ok=True)

    mkv_path = source_video_path(workdir)
    wav_path = audio_wav_path(workdir)

    if skip_download:
        if not file_nonempty(mkv_path):
            print(f"error: expected {mkv_path} (use --skip-download only when MKV exists)", file=sys.stderr)
            raise SystemExit(1)
        return

    mkv_exists = file_nonempty(mkv_path)
    wav_exists = file_nonempty(wav_path)

    if mkv_exists and wav_exists and not force:
        return

    if mkv_exists and not wav_exists:
        extract_audio(mkv_path, wav_path)
        return

    if wav_exists and not mkv_exists:
        if not url:
            print("error: YouTube URL required to re-download missing source.mkv", file=sys.stderr)
            raise SystemExit(1)
        download_video(url, mkv_path)
        extract_audio(mkv_path, wav_path)
        return

    if not url:
        print("error: YouTube URL required for download", file=sys.stderr)
        raise SystemExit(1)

    download_video(url, mkv_path)
    extract_audio(mkv_path, wav_path)
