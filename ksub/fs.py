"""Filesystem and subprocess helpers shared across stages."""

from __future__ import annotations

import subprocess
from pathlib import Path


def file_nonempty(path: Path) -> bool:
    """True when path exists and has non-zero size."""
    return path.is_file() and path.stat().st_size > 0


def extract_audio(mkv_path: Path, wav_path: Path) -> None:
    """Extract 16 kHz mono WAV from a video file."""
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(mkv_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
            "-y",
        ],
        check=True,
        capture_output=True,
    )