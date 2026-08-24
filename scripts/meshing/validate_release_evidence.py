#!/usr/bin/env python3
"""Fail closed if meshing release evidence can be bypassed or omitted."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW = REPO_ROOT / ".github/workflows/release.yml"


class ReleaseEvidenceError(ValueError):
    """Raised when the release workflow does not enforce meshing evidence."""


def _job_block(workflow: str, job: str) -> str:
    match = re.search(rf"(?m)^  {re.escape(job)}:\s*$", workflow)
    if match is None:
        raise ReleaseEvidenceError(f"release workflow is missing the {job} job")
    following = re.search(r"(?m)^  [A-Za-z0-9_-]+:\s*$", workflow[match.end() :])
    end = len(workflow) if following is None else match.end() + following.start()
    return workflow[match.start() : end]


def validate_workflow(path: Path = DEFAULT_WORKFLOW) -> None:
    try:
        workflow = path.read_text()
    except OSError as error:
        raise ReleaseEvidenceError(f"cannot read release workflow: {error}") from error

    for permission in ("contents: write", "id-token: write", "attestations: write"):
        if permission not in workflow:
            raise ReleaseEvidenceError(f"release workflow is missing permission {permission}")

    evidence = _job_block(workflow, "meshing-release-evidence")
    required = (
        "needs: [validate-release-state]",
        "Validate governed meshing evidence catalogs",
        "Run meshing contract and algorithm conformance",
        "Run serial, local, durable, remote, and differential conformance",
        "--test occt_exact_meshing",
        "--test meshing_process_conformance",
        "--profile stable",
        "--samples 15",
        "actions/attest@v4",
        "subject-path: target/runmat-meshing-release-evidence/*",
        "Upload mandatory meshing release evidence",
        "if-no-files-found: error",
    )
    for anchor in required:
        if anchor not in evidence:
            raise ReleaseEvidenceError(f"meshing release evidence is missing: {anchor}")

    for anchor in ("continue-on-error:", "if: false", "if-no-files-found: warn"):
        if anchor in evidence:
            raise ReleaseEvidenceError(f"meshing release evidence contains bypass: {anchor}")

    promotion = _job_block(workflow, "create-release")
    dependency = "needs: [build-and-test, meshing-release-evidence, build-release]"
    if dependency not in promotion:
        raise ReleaseEvidenceError("release promotion does not require meshing release evidence")


def main() -> int:
    try:
        validate_workflow()
    except ReleaseEvidenceError as error:
        print(f"meshing release-evidence validation failed: {error}", file=sys.stderr)
        return 1
    print("validated blocking, mandatory, attested meshing release evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
