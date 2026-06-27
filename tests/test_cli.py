"""Tests for CLI stage orchestration and workdir resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ksub import cli


def test_print_summary_labels_actual_srt_filenames(tmp_path: Path, capsys):
    workdir = tmp_path / "Episode_Title"
    workdir.mkdir()
    (workdir / "Episode_Title.ko.srt").write_text("ko", encoding="utf-8")
    (workdir / "Episode_Title.en.srt").write_text("en", encoding="utf-8")

    cli._print_summary(workdir)

    out = capsys.readouterr().out
    assert "Episode_Title.ko.srt:" in out
    assert "Episode_Title.en.srt:" in out
    assert "\n  ko.srt:" not in out
    assert "\n  en.srt:" not in out


def test_translate_only_uses_workdir_without_url(tmp_path: Path):
    workdir = tmp_path / "episode"
    resolved_workdir = workdir.resolve()

    with patch("ksub.cli.download.fetch_title") as fetch_title:
        with patch("ksub.cli.translate.run_translate", return_value="cost line") as run_translate:
            with patch("ksub.cli._print_summary") as print_summary:
                cli.main([
                    "--translate-only",
                    "--workdir",
                    str(workdir),
                    "--langs",
                    "en",
                    "--force",
                    "--engine-translate",
                    "ollama",
                ])

    fetch_title.assert_not_called()
    run_translate.assert_called_once_with(
        resolved_workdir,
        ["en"],
        force=True,
        engine="ollama",
    )
    print_summary.assert_called_once_with(
        resolved_workdir,
        muxed_path=None,
        cost_line="cost line",
    )


def test_mux_only_requires_workdir(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--mux-only"])

    assert exc_info.value.code == 1
    assert "--mux-only requires --workdir" in capsys.readouterr().err


def test_skip_download_full_pipeline_uses_workdir_without_url(tmp_path: Path):
    workdir = tmp_path / "episode"
    resolved_workdir = workdir.resolve()

    with patch("ksub.cli.download.run_download") as run_download:
        with patch("ksub.cli.transcribe.run_transcribe") as run_transcribe:
            with patch("ksub.cli.segment.run_segment") as run_segment:
                with patch("ksub.cli.translate.run_translate") as run_translate:
                    with patch("ksub.cli.mux.run_mux") as run_mux:
                        with patch("ksub.cli._print_summary") as print_summary:
                            cli.main([
                                "--skip-download",
                                "--workdir",
                                str(workdir),
                                "--langs",
                                "",
                                "--no-mux",
                            ])

    run_download.assert_called_once_with(
        resolved_workdir,
        None,
        force=False,
        skip_download=True,
    )
    run_transcribe.assert_called_once_with(resolved_workdir, force=False)
    run_segment.assert_called_once_with(resolved_workdir, force=False)
    run_translate.assert_not_called()
    run_mux.assert_not_called()
    print_summary.assert_called_once_with(
        resolved_workdir,
        muxed_path=None,
        cost_line=None,
    )


def test_full_pipeline_resolves_url_title_and_runs_stages_in_order(tmp_path: Path):
    out_dir = tmp_path / "out"
    resolved_workdir = (out_dir / "Episode_Title").resolve()
    muxed_path = resolved_workdir / "Episode_Title.mkv"
    events: list[str] = []

    def record(name: str):
        def _inner(*args, **kwargs):
            events.append(name)
            if name == "translate":
                return "cost line"
            if name == "mux":
                return muxed_path
            return None

        return _inner

    with patch("ksub.cli.download.fetch_title", return_value="Episode Title") as fetch_title:
        with patch("ksub.cli.download.sanitize_title", return_value="Episode_Title") as sanitize_title:
            with patch("ksub.cli.download.run_download", side_effect=record("download")) as run_download:
                with patch("ksub.cli.transcribe.run_transcribe", side_effect=record("transcribe")) as run_transcribe:
                    with patch("ksub.cli.segment.run_segment", side_effect=record("segment")) as run_segment:
                        with patch("ksub.cli.translate.run_translate", side_effect=record("translate")) as run_translate:
                            with patch("ksub.cli.mux.run_mux", side_effect=record("mux")) as run_mux:
                                with patch("ksub.cli._print_summary") as print_summary:
                                    cli.main([
                                        "https://example.test/watch",
                                        "--out",
                                        str(out_dir),
                                        "--langs",
                                        "zh,en",
                                    ])

    fetch_title.assert_called_once_with("https://example.test/watch")
    sanitize_title.assert_called_once_with("Episode Title", "https://example.test/watch")
    run_download.assert_called_once_with(
        resolved_workdir,
        "https://example.test/watch",
        force=False,
        skip_download=False,
    )
    run_transcribe.assert_called_once_with(resolved_workdir, force=False)
    run_segment.assert_called_once_with(resolved_workdir, force=False)
    run_translate.assert_called_once_with(
        resolved_workdir,
        ["zh", "en"],
        force=False,
        engine=None,
    )
    run_mux.assert_called_once_with(
        resolved_workdir,
        ["zh", "en"],
        force=False,
        mp4=False,
        keep_source=False,
    )
    assert events == ["download", "transcribe", "segment", "translate", "mux"]
    print_summary.assert_called_once_with(
        resolved_workdir,
        muxed_path=muxed_path,
        cost_line="cost line",
    )
