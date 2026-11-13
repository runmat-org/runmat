#!/usr/bin/env python3
"""Verify committed scientific evidence without re-running numerical workloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "qualification/scientific/policy.json"
RECEIPTS = ROOT / "qualification/scientific/receipts"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not relative:
        raise ValueError("unsafe evidence path")
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("evidence path escapes its repository")
    return path


def policy(root: Path = ROOT) -> dict:
    value = read_json(root / POLICY.relative_to(ROOT))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported scientific policy")
    ids = [cohort["id"] for cohort in value["cohorts"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("scientific cohorts must be nonempty and unique")
    for cohort in value["cohorts"]:
        if not cohort["required"] and not cohort.get("exception"):
            raise ValueError("optional scientific cohort needs an explicit exception")
    return value


def input_identity(root: Path, revision: str = "HEAD", *, require_clean: bool = True) -> dict:
    roots = policy(root)["input_roots"]
    if require_clean:
        dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *roots], cwd=root)
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *roots], cwd=root)
        if dirty or untracked:
            raise ValueError("scientific inputs are not a committed, frozen source snapshot")
    # Git's canonical mode/blob/path inventory is independent of checkout CRLF
    # conversion and commit ancestry. New, removed and executable-mode inputs
    # change the digest; version/license history does not get silently relabelled.
    return inventory_identity(source_entries(root, revision))


def source_entries(root: Path, revision: str = "HEAD") -> dict:
    roots = policy(root)["input_roots"]
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision, "--", *roots], cwd=root)
    if not raw:
        raise ValueError("empty scientific source inventory")
    entries = {}
    for entry in raw.rstrip(b"\0").split(b"\0"):
        metadata, path = entry.split(b"\t", 1)
        entries[path.decode()] = metadata.decode().split(" ")
    return entries


def inventory_identity(entries: dict) -> dict:
    canonical = [[path, *metadata] for path, metadata in sorted(entries.items())]
    return {"sha256": digest(json.dumps(canonical, separators=(",", ":")).encode()),
            "files": len(entries), "encoding": "git-mode-type-blob-path-tuples-v1"}


def verify_historical_measurement(root: Path, cohort: dict, receipt: dict) -> None:
    reference = receipt.get("historical_measurement")
    if not isinstance(reference, dict) or reference not in cohort.get("approved_historical_measurements", []):
        raise ValueError("historical measurement is not explicitly approved for this cohort")
    path = safe_file(root, reference["path"])
    if digest(path.read_bytes()) != reference["sha256"]:
        raise ValueError("approved historical measurement is missing or changed")
    record = read_json(path)
    if record.get("schema_version") != 1 or record.get("cohort") != cohort["id"]:
        raise ValueError("approved historical measurement has the wrong cohort")
    if json.dumps(record.get("measured"), sort_keys=True) != json.dumps(receipt["measured"], sort_keys=True):
        raise ValueError("receipt changes the approved historical measurement")


def verify_receipt(root: Path, cohort: dict, receipt: dict, inputs: dict, policy_hash: str) -> None:
    if receipt.get("schema_version") != 1 or receipt.get("cohort") != cohort["id"]:
        raise ValueError("wrong scientific receipt identity")
    if receipt.get("policy_sha256") != policy_hash:
        raise ValueError("scientific policy differs from the qualified receipt")
    measured = receipt["measured"]
    for key in ("source_revision", "target", "toolchain", "command", "log_sha256", "log_location"):
        if not measured.get(key):
            raise ValueError(f"scientific receipt lacks {key}")
    if not isinstance(measured["command"], list) or not all(isinstance(item, str) and item for item in measured["command"]):
        raise ValueError("scientific receipt must preserve its actual argument vector")
    if "historical_measurement" in receipt:
        verify_historical_measurement(root, cohort, receipt)
    elif "command" in cohort and measured["command"] != cohort["command"]:
        raise ValueError("scientific command differs from its declared cohort")
    revision = measured["source_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("scientific receipt lacks its original source revision")
    for field in ("passed", "failed", "ignored", "filtered_out", "exit_code"):
        value = measured.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"scientific receipt lacks a nonnegative integer {field}")
    if measured["exit_code"] != 0 or measured["failed"] != 0 or measured["passed"] <= 0:
        raise ValueError("scientific workload did not pass")
    if measured.get("ignored") != 0:
        raise ValueError("required scientific workload has omitted cases")
    if cohort.get("expected_passed") and measured["passed"] != cohort["expected_passed"]:
        raise ValueError("scientific case inventory is incomplete")
    names = measured.get("case_names", [])
    if len(names) != measured["passed"] or len(set(names)) != len(names):
        raise ValueError("scientific receipt lacks its complete named case inventory")
    patterns = cohort.get("ordinary_skip", [])
    if patterns and (any(not any(name.startswith(pattern) for name in names) for pattern in patterns)
                     or any(not any(name.startswith(pattern) for pattern in patterns) for name in names)):
        raise ValueError("scientific case names differ from the routed cohort")
    if measured.get("selected_filter") != cohort.get("selected_filter") or (
        measured["filtered_out"] and not cohort.get("selected_filter")
    ):
        raise ValueError("scientific filtering differs from its declared cohort")
    for field in ("log_sha256",):
        value = measured[field]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid evidence digest")
    if not measured.get("artifacts") or any(
        len(artifact.get("sha256", "")) != 64
        or any(c not in "0123456789abcdef" for c in artifact["sha256"])
        or not artifact.get("location")
        for artifact in measured["artifacts"]
    ):
        raise ValueError("scientific receipt lacks executed artifact identities")
    if cohort.get("required_target") and measured["target"] != cohort["required_target"]:
        raise ValueError("scientific receipt has the wrong target")
    if measured.get("inputs") == inputs:
        return
    conservation = receipt.get("conservation")
    if not conservation or conservation.get("measured_inputs") != measured.get("inputs") or conservation.get("accepted_inputs") != inputs:
        raise ValueError("scientific source inputs changed without reviewed conservation")
    if not conservation.get("reason") or not conservation.get("review"):
        raise ValueError("source conservation requires its explicit review")
    review = conservation["review"]
    path = safe_file(root, review["path"])
    if not review["path"].startswith("qualification/scientific/") or digest(path.read_bytes()) != review["sha256"]:
        raise ValueError("scientific conservation review is missing or changed")
    record = read_json(path)
    if record.get("measured_inputs") != measured["inputs"] or record.get("accepted_inputs") != inputs:
        raise ValueError("conservation review does not bind both source snapshots")
    if cohort["id"] not in record.get("cohorts", []) or not record.get("changed_inputs") or not record.get("qualification_evidence"):
        raise ValueError("conservation review lacks scope, changes or affected qualification")
    # Invert the exact declared edits and reconstruct the measured inventory.
    # This proves there are no undeclared changes without requiring historical
    # commits to remain reachable after a separately coordinated history rewrite.
    restored = dict(source_entries(root))
    seen = set()
    for change in record["changed_inputs"]:
        path = change["path"]
        if path in seen or not change.get("reason") or restored.get(path) != change.get("after"):
            raise ValueError("conservation edits differ from current source")
        seen.add(path)
        before = change.get("before")
        if before is None:
            restored.pop(path, None)
        else:
            if len(before) != 3 or len(before[2]) != 40 or any(c not in "0123456789abcdef" for c in before[2]):
                raise ValueError("invalid conserved Git input identity")
            restored[path] = before
    if inventory_identity(restored) != measured["inputs"]:
        raise ValueError("conservation edits do not reconstruct the measured source")
    for evidence in record["qualification_evidence"]:
        if digest(safe_file(root, evidence["path"]).read_bytes()) != evidence["sha256"]:
            raise ValueError("affected qualification evidence is missing or changed")


def verify(root: Path = ROOT, selection: str | None = None) -> list[str]:
    specification = policy(root)
    inputs = input_identity(root)
    policy_hash = digest((root / POLICY.relative_to(ROOT)).read_bytes())
    cohorts = [c for c in specification["cohorts"] if c["required"] and (not selection or c["id"] == selection)]
    if not cohorts:
        raise ValueError("no required scientific cohort selected")
    verified = []
    for cohort in cohorts:
        receipt = read_json(root / RECEIPTS.relative_to(ROOT) / (cohort["id"] + ".json"))
        verify_receipt(root, cohort, receipt, inputs, policy_hash)
        verified.append(cohort["id"])
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fingerprint", "verify", "command"))
    parser.add_argument("--cohort")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    try:
        if args.action == "fingerprint":
            print(json.dumps(input_identity(ROOT, args.revision), indent=2))
        elif args.action == "verify":
            for name in verify(selection=args.cohort):
                print(f"verified committed scientific receipt: {name}")
        else:
            cohort = next(c for c in policy()["cohorts"] if c["id"] == args.cohort)
            print(json.dumps(cohort["command"]))
    except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        parser.exit(1, f"scientific evidence rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
