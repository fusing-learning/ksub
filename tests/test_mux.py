"""Tests for mux command edge cases."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ksub.mux import run_mux


def _touch(path: Path, text: str = "x") -> None:
    path.write_text(text, encoding="utf-8")


def test_force_remux_existing_output_uses_temp_file(tmp_path: Path):
    workdir = tmp_path / "episode"
    workdir.mkdir()
    output_path = workdir / "episode.mkv"
    _touch(output_path, "old")
    _touch(workdir / "episode.ko.srt")

    seen_output: Path | None = None

    def fake_run(cmd, *, check, capture_output):
        nonlocal seen_output
        assert check is True
        assert capture_output is True
        seen_output = Path(cmd[-2])
        assert seen_output != output_path
        _touch(seen_output, "new")

    with patch("ksub.mux.check_prerequisites"):
        with patch("ksub.mux.subprocess.run", side_effect=fake_run):
            result = run_mux(workdir, [], force=True)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "new"
    assert seen_output is not None
    assert not seen_output.exists()


def test_force_remux_keeps_temp_file_when_replace_fails(tmp_path: Path, capsys):
    workdir = tmp_path / "episode"
    workdir.mkdir()
    output_path = workdir / "episode.mkv"
    _touch(output_path, "old")
    _touch(workdir / "episode.ko.srt")

    seen_output: Path | None = None

    def fake_run(cmd, *, check, capture_output):
        nonlocal seen_output
        assert check is True
        assert capture_output is True
        seen_output = Path(cmd[-2])
        _touch(seen_output, "new")

    with patch("ksub.mux.check_prerequisites"):
        with patch("ksub.mux.subprocess.run", side_effect=fake_run):
            with patch("pathlib.Path.replace", side_effect=OSError("replace failed")):
                with pytest.raises(OSError):
                    run_mux(workdir, [], force=True)

    captured = capsys.readouterr()
    assert output_path.read_text(encoding="utf-8") == "old"
    assert seen_output is not None
    assert seen_output.read_text(encoding="utf-8") == "new"
    assert "remuxed output kept at" in captured.err
    assert str(seen_output) in captured.err


def test_mux_prints_ffmpeg_stderr_on_failure(tmp_path: Path, capsys):
    workdir = tmp_path / "episode"
    workdir.mkdir()
    _touch(workdir / "source.mkv")
    _touch(workdir / "episode.ko.srt")

    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["ffmpeg"],
        stderr=b"ffmpeg mux failure\n",
    )

    with patch("ksub.mux.check_prerequisites"):
        with patch("ksub.mux.subprocess.run", side_effect=error):
            with pytest.raises(subprocess.CalledProcessError):
                run_mux(workdir, [])

    captured = capsys.readouterr()
    assert "ffmpeg mux failure" in captured.err


def test_mux_removes_legacy_srt_duplicates(tmp_path: Path):
    workdir = tmp_path / "episode"
    workdir.mkdir()
    _touch(workdir / "source.mkv")
    _touch(workdir / "episode.ko.srt")
    _touch(workdir / "episode.zh.srt")
    _touch(workdir / "episode.en.srt")
    _touch(workdir / "ko.srt")
    _touch(workdir / "zh.srt")
    _touch(workdir / "en.srt")

    def fake_run(cmd, *, check, capture_output):
        assert check is True
        assert capture_output is True
        _touch(Path(cmd[-2]), "muxed")

    with patch("ksub.mux.check_prerequisites"):
        with patch("ksub.mux.subprocess.run", side_effect=fake_run):
            result = run_mux(workdir, ["zh"], keep_source=True)

    assert result == workdir / "episode.mkv"
    assert (workdir / "episode.ko.srt").is_file()
    assert (workdir / "episode.zh.srt").is_file()
    assert (workdir / "episode.en.srt").is_file()
    assert not (workdir / "ko.srt").exists()
    assert not (workdir / "zh.srt").exists()
    assert not (workdir / "en.srt").exists()
