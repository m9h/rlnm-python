import json
import os
import subprocess
import sys


def test_cli_help_and_version():
    for argument in ("--help", "--version"):
        result = subprocess.run(
            [sys.executable, "-m", "rlnm.cli", argument], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert result.stdout


def test_cli_failure_is_nonzero(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"analysis": "unknown"}))
    result = subprocess.run(
        [sys.executable, "-m", "rlnm.cli", "run", str(config), "--output", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 2
    assert "analysis must" in result.stderr


def test_research_demo_and_commands(tmp_path):
    from rlnm.cli import main

    directory = tmp_path / "research"
    assert main(["demo-research", "--output", str(directory)]) == 0
    assert (directory / "annotate.json").exists()
    assert (directory / "benchmark.json").exists()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"unknown": true}')
    for command in ("annotate", "benchmark-eigenbrain"):
        assert main([command, str(invalid), "--output", str(tmp_path / command)]) == 2
