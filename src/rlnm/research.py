"""Shared validation and provenance for optional research workflows."""

import csv
import importlib.metadata
import json
import platform
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .io import sha256


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def keys(config, allowed, required):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if set(config) - set(allowed):
        raise ValueError(f"Unknown config keys: {sorted(set(config) - set(allowed))}")
    if set(required) - set(config):
        raise ValueError(f"Missing config keys: {sorted(set(required) - set(config))}")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def text_setting(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def parcel_ids(values):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not len(values)
        or values.dtype.kind not in "iu"
        or np.any(values <= 0)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("parcel_ids must be unique positive integers in explicit array order")
    return values


def identifiers(values, name):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not len(values)
        or values.dtype.kind not in "US"
        or any(not str(x).strip() for x in values)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError(f"{name} must be unique nonempty strings")
    return values


def load_npz(path, required):
    with np.load(path, allow_pickle=False) as archive:
        if set(required) != set(archive.files):
            raise ValueError(f"{path} must contain exactly: {', '.join(required)}")
        return {key: archive[key] for key in required}


def write_tsv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def research_run(config_path, output, analysis, config):
    """Record failure as well as success; never overwrite an existing analysis."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("numpy", "scipy", "nibabel", "neuromaps", "brainsmash", "brainspace"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {
        "status": "running",
        "analysis": analysis,
        "version": __version__,
        "started_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "config": config,
        "versions": versions,
        "inputs": [],
        "space_is_user_declared": True,
    }
    paths = [Path(config_path).resolve()]
    started = time.monotonic()
    dump_json(output / "provenance.json", record)
    try:
        yield output, paths, record
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record["inputs"] = [
            {"path": str(p), "sha256": sha256(p)} for p in dict.fromkeys(paths) if p.is_file()
        ]
        record["outputs"] = [
            {"path": str(p.relative_to(output)), "sha256": sha256(p)}
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "provenance.json"
        ]
        dump_json(output / "provenance.json", record)
