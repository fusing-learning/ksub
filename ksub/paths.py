"""Workdir path helpers — VLC-friendly matching names."""

from __future__ import annotations

from pathlib import Path

from ksub import config
from ksub.fs import file_nonempty

# Internal pipeline artifacts (not for playback)
SOURCE_VIDEO = "source.mkv"
AUDIO_WAV = "audio.wav"
WORDS_JSON = "words.json"

# Legacy filenames from earlier versions
LEGACY_SUBTITLED = "subtitled.mkv"
LEGACY_SUBTITLED_MP4 = "subtitled.mp4"


def media_basename(workdir: Path) -> str:
    """Base name shared by video + external SRTs (matches the workdir folder name)."""
    return workdir.name or "video"


def video_output_path(workdir: Path, *, mp4: bool = False) -> Path:
    """Final muxed video: <workdir>/<basename>.mkv (VLC pairs with <basename>.*.srt)."""
    ext = "mp4" if mp4 else "mkv"
    return workdir / f"{media_basename(workdir)}.{ext}"


def source_video_path(workdir: Path) -> Path:
    """Internal downloaded/remuxed source video path."""
    return workdir / SOURCE_VIDEO


def audio_wav_path(workdir: Path) -> Path:
    """Internal extracted audio path."""
    return workdir / AUDIO_WAV


def words_json_path(workdir: Path) -> Path:
    """Internal ASR word-timestamp JSON path."""
    return workdir / WORDS_JSON


def srt_path(workdir: Path, lang: str) -> Path:
    """External SRT path VLC auto-detects: <basename>.ko.srt, .zh.srt, .en.srt."""
    return workdir / f"{media_basename(workdir)}.{lang}.srt"


def legacy_srt_path(workdir: Path, lang: str) -> Path:
    """Legacy SRT path from the language registry: <lang>.srt."""
    return workdir / f"{config.get_lang(lang).cli}.srt"


def _first_existing(workdir: Path, candidates: list[Path]) -> Path | None:
    for path in candidates:
        if file_nonempty(path):
            return path
    return None


def resolve_srt_path(workdir: Path, lang: str) -> Path:
    """Resolve SRT for reading; prefers VLC name, falls back to legacy ko.srt etc."""
    found = _first_existing(workdir, [
        srt_path(workdir, lang),
        legacy_srt_path(workdir, lang),
    ])
    return found if found else srt_path(workdir, lang)


def resolve_video_input(workdir: Path) -> Path | None:
    """Video source for mux/transcribe: source.mkv, then canonical, then legacy outputs."""
    basename = media_basename(workdir)
    return _first_existing(workdir, [
        source_video_path(workdir),
        workdir / f"{basename}.mkv",
        workdir / f"{basename}.mp4",
        workdir / LEGACY_SUBTITLED,
        workdir / LEGACY_SUBTITLED_MP4,
    ])


def remove_legacy_srt(workdir: Path, lang: str) -> None:
    """Drop the legacy <lang>.srt once the canonical <basename>.<lang>.srt exists."""
    canonical = srt_path(workdir, lang)
    legacy = legacy_srt_path(workdir, lang)
    if file_nonempty(canonical) and legacy.is_file() and legacy != canonical:
        legacy.unlink()


def remove_legacy_video(workdir: Path, *, mp4: bool = False) -> None:
    """Drop the legacy subtitled.<ext> once the matching <basename>.<ext> exists."""
    canonical = video_output_path(workdir, mp4=mp4)
    legacy = workdir / (LEGACY_SUBTITLED_MP4 if mp4 else LEGACY_SUBTITLED)
    if file_nonempty(canonical) and legacy.is_file() and legacy != canonical:
        legacy.unlink()
