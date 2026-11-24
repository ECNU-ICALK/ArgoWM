"""Stable, human-readable names for maize-derived artifacts."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath


OUTPUT_NAMING_VERSION = "wgsr_maize_underscore_filename_v1"


def _validated_relative_path(filename: str) -> PurePosixPath:
    """Return a safe POSIX relative path used by the maize manifests."""

    if not filename or "\x00" in filename:
        raise ValueError(f"Invalid maize relative filename: {filename!r}")
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe maize relative filename: {filename!r}")
    if not path.name or path.name in {".", ".."}:
        raise ValueError(f"Invalid maize relative filename: {filename!r}")
    return path


def maize_output_stem(filename: str) -> str:
    """Convert ``001a/S1_sv1.png`` to the readable stem ``001a_S1_sv1``.

    The source path remains the canonical identity.  Only path separators and
    the final suffix are removed from the artifact name, so the generated
    files remain easy to inspect while preserving a deterministic one-to-one
    mapping.  Callers that handle a manifest must still check that the stems
    are unique within that manifest.
    """

    path = _validated_relative_path(filename)
    suffix = path.suffix
    leaf = path.name[: -len(suffix)] if suffix else path.name
    if not leaf:
        raise ValueError(f"Maize filename has an empty stem: {filename!r}")
    stem = "_".join((*path.parts[:-1], leaf))
    if not stem or stem in {".", ".."}:
        raise ValueError(f"Invalid maize output stem for {filename!r}")
    return stem


def maize_legacy_output_stem(filename: str) -> str:
    """Return the pre-v1 SHA-256 stem for compatibility with old artifacts."""

    _validated_relative_path(filename)
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()


__all__ = [
    "OUTPUT_NAMING_VERSION",
    "maize_legacy_output_stem",
    "maize_output_stem",
]
