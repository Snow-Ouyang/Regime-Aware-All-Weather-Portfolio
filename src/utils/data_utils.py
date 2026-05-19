"""Shared data path helpers for the cleaned research project."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """Return an absolute path under the project root."""

    return ROOT.joinpath(*parts)
