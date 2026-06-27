"""Tests for playback artifact path cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from ksub.paths import (
    legacy_srt_path,
    remove_legacy_srt,
    remove_legacy_video,
    resolve_srt_path,
)


def _touch(path: Path) -> None:
    path.write_text("x", encoding="utf-8")


def test_legacy_srt_path_uses_language_registry(tmp_path: Path):
    workdir = tmp_path / "episode"

    assert legacy_srt_path(workdir, "zh") == workdir / "zh.srt"


def test_resolve_srt_path_rejects_unsupported_language_with_config_error(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported language: fr"):
        resolve_srt_path(tmp_path, "fr")


def test_remove_legacy_video_mp4_when_canonical_mp4_exists(tmp_path: Path):
    workdir = tmp_path / "episode"
    workdir.mkdir()

    _touch(workdir / "episode.mp4")
    _touch(workdir / "subtitled.mp4")

    remove_legacy_video(workdir, mp4=True)

    assert (workdir / "episode.mp4").is_file()
    assert not (workdir / "subtitled.mp4").exists()


def test_remove_legacy_video_mkv_when_canonical_mkv_exists(tmp_path: Path):
    workdir = tmp_path / "episode"
    workdir.mkdir()

    _touch(workdir / "episode.mkv")
    _touch(workdir / "subtitled.mkv")

    remove_legacy_video(workdir, mp4=False)

    assert (workdir / "episode.mkv").is_file()
    assert not (workdir / "subtitled.mkv").exists()


def test_remove_legacy_video_does_not_cross_extensions(tmp_path: Path):
    # A canonical .mp4 must not trigger deletion of the legacy .mkv (different format).
    workdir = tmp_path / "episode"
    workdir.mkdir()

    _touch(workdir / "episode.mp4")
    _touch(workdir / "subtitled.mkv")

    remove_legacy_video(workdir, mp4=True)

    assert (workdir / "subtitled.mkv").is_file()


def test_remove_legacy_srt_when_canonical_srt_exists(tmp_path: Path):
    workdir = tmp_path / "episode"
    workdir.mkdir()

    _touch(workdir / "episode.ko.srt")
    _touch(workdir / "ko.srt")

    remove_legacy_srt(workdir, "ko")

    assert (workdir / "episode.ko.srt").is_file()
    assert not (workdir / "ko.srt").exists()


def test_remove_legacy_srt_scoped_to_lang(tmp_path: Path):
    # Cleaning up "zh" must not touch a legacy "en.srt".
    workdir = tmp_path / "episode"
    workdir.mkdir()

    _touch(workdir / "episode.zh.srt")
    _touch(workdir / "zh.srt")
    _touch(workdir / "en.srt")

    remove_legacy_srt(workdir, "zh")

    assert not (workdir / "zh.srt").exists()
    assert (workdir / "en.srt").is_file()


def test_remove_legacy_keeps_files_when_canonical_is_empty(tmp_path: Path):
    workdir = tmp_path / "episode"
    workdir.mkdir()

    (workdir / "episode.mkv").write_text("", encoding="utf-8")
    (workdir / "episode.ko.srt").write_text("", encoding="utf-8")
    _touch(workdir / "subtitled.mkv")
    _touch(workdir / "ko.srt")

    remove_legacy_video(workdir, mp4=False)
    remove_legacy_srt(workdir, "ko")

    assert (workdir / "subtitled.mkv").is_file()
    assert (workdir / "ko.srt").is_file()
