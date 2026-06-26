"""Workdir path helpers — VLC-friendly matching names."""

from __future__ import annotations

from pathlib import Path

# Internal pipeline artifacts (not for playback)
SOURCE_VIDEO = "source.mkv"
AUDIO_WAV = "audio.wav"
WORDS_JSON = "words.json"

# Legacy filenames from earlier versions
LEGACY_SUBTITLED = "subtitled.mkv"
LEGACY_SUBTITLED_MP4 = "subtitled.mp4"
LEGACY_SRT = {"ko": "ko.srt", "zh": "zh.srt", "en": "en.srt"}


def media_basename(workdir: Path) -> str:
    """Base name shared by video + external SRTs (matches the workdir folder name)."""
    return workdir.name or "video"


def video_output_path(workdir: Path, *, mp4: bool = False) -> Path:
    """Final muxed video: <workdir>/<basename>.mkv (VLC pairs with <basename>.*.srt)."""
    ext = "mp4" if mp4 else "mkv"
    return workdir / f"{media_basename(workdir)}.{ext}"


def srt_path(workdir: Path, lang: str) -> Path:
    """External SRT path VLC auto-detects: <basename>.ko.srt, .zh.srt, .en.srt."""
    return workdir / f"{media_basename(workdir)}.{lang}.srt"


def _first_existing(workdir: Path, candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_srt_path(workdir: Path, lang: str) -> Path:
    """Resolve SRT for reading; prefers VLC name, falls back to legacy ko.srt etc."""
    found = _first_existing(workdir, [
        srt_path(workdir, lang),
        workdir / LEGACY_SRT[lang],
    ])
    return found if found else srt_path(workdir, lang)


def resolve_video_input(workdir: Path) -> Path | None:
    """Video source for mux/transcribe: source.mkv, then final/legacy muxed output."""
    basename = media_basename(workdir)
    return _first_existing(workdir, [
        workdir / SOURCE_VIDEO,
        workdir / f"{basename}.mkv",
        workdir / LEGACY_SUBTITLED,
        workdir / f"{basename}.mp4",
        workdir / LEGACY_SUBTITLED_MP4,
    ])


def remove_legacy_playback_files(workdir: Path) -> None:
    """Drop old ko.srt / subtitled.mkv names after VLC-friendly files exist."""
    basename = media_basename(workdir)
    video = workdir / f"{basename}.mkv"
    if video.is_file():
        for legacy in (workdir / LEGACY_SUBTITLED, workdir / LEGACY_SUBTITLED_MP4):
            if legacy.is_file() and legacy != video:
                legacy.unlink()
    for lang, legacy_name in LEGACY_SRT.items():
        legacy = workdir / legacy_name
        canonical = srt_path(workdir, lang)
        if canonical.is_file() and legacy.is_file() and legacy != canonical:
            legacy.unlink()