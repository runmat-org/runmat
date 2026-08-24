"""Shared fail-closed primitives for versioned meshing verification catalogs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CatalogError(ValueError):
    """Raised when governed meshing catalog evidence is invalid."""


def read_document(path: Path, label: str) -> dict[str, Any]:
    try:
        return object_value(json.loads(path.read_text()), label)
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"cannot read {label}: {error}") from error


def object_value(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{context} must be an object")
    return value


def string_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context} must be a non-empty string")
    return value


def string_array(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CatalogError(f"{context} must be a non-empty array")
    strings = [string_value(item, f"{context}[]") for item in value]
    if len(strings) != len(set(strings)):
        raise CatalogError(f"{context} must be unique")
    return strings


def relative_path(value: Any, context: str) -> Path:
    path = Path(string_value(value, context))
    if path.is_absolute() or ".." in path.parts:
        raise CatalogError(f"{context} must stay below its declared root")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_test_anchor(repo_root: Path, test: Any, context: str) -> None:
    test_object = object_value(test, context)
    source = repo_root / relative_path(test_object.get("source"), f"{context}.source")
    name = string_value(test_object.get("name"), f"{context}.name")
    if not source.is_file():
        raise CatalogError(f"{context}: test source does not exist")
    if f"fn {name}(" not in source.read_text():
        raise CatalogError(f"{context}: test anchor {name} does not exist")
