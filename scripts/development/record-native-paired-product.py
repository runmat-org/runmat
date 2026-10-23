#!/usr/bin/env python3
"""Record external physical facts for an already qualified native pair.

This tool grants no pack, link, or execution authority. It requires the
create-new receipt emitted by the actual paired witness, rehashes both real
images, and records the operating system loader's dependency observations.
The matching Rust witness validates canonical receipts; this observer does
not implement a second artifact validator or grant installation permission.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

MAX_OBSERVATION_BYTES = 64 * 1024 * 1024


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_bounded(path, maximum):
    with path.open("rb") as source:
        data = source.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"{path} exceeds its evidence byte bound")
    return data


def digest_file(path, expected_bytes):
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path} differs from its completed extent")
    value = hashlib.sha256()
    seen = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            seen += len(chunk)
            if seen > expected_bytes:
                raise ValueError(f"{path} grew beyond its completed extent")
            value.update(chunk)
    if seen != expected_bytes:
        raise ValueError(f"{path} changed during physical observation")
    return "sha256:" + value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def image_path(root, role):
    name = f"paired-{role}" + (".exe" if os.name == "nt" else "")
    path = root / name
    physical = path.resolve(strict=True)
    if path.is_symlink() or not physical.is_file() or not physical.is_relative_to(root):
        raise ValueError(f"{role} image is not a physical file within the pair output")
    return physical


def loader_dependencies(path):
    if sys.platform != "darwin":
        raise ValueError("this external loader observer currently supports macOS only")
    process = subprocess.Popen(
        ["/usr/bin/otool", "-L", str(path)],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError(f"loader inspection timed out for {path}") from error
    if process.returncode:
        raise RuntimeError(
            f"loader inspection failed for {path}: {stderr.decode(errors='replace')}"
        )
    observed = []
    for line in stdout.decode(errors="strict").splitlines()[1:]:
        if not line.startswith("\t"):
            continue
        install_name = line.strip().split(" (", 1)[0]
        kind = (
            "host_os_install_name"
            if install_name.startswith(("/usr/lib/", "/System/Library/"))
            else "image_relative_install_name"
        )
        observed.append({"kind": kind, "install_name": install_name})
    return sorted(observed, key=lambda item: (item["kind"], item["install_name"]))


def observe(role, path, completed):
    expected_bytes = completed["output_bytes"]
    actual = {"digest": digest_file(path, expected_bytes), "bytes": expected_bytes}
    expected = {
        "digest": completed["output_digest"],
        "bytes": completed["output_bytes"],
    }
    if actual != expected:
        raise ValueError(f"{role} image differs from its completed link receipt")
    return {
        "role": role,
        **actual,
        "loader_dependencies": loader_dependencies(path),
    }


def build_subject(product):
    """Retain verified semantic/build inputs independently of image output.

    These are observational copies of Rust-validated records, not newly
    issued native identities. Full manifest equality includes source/program,
    selected product, policies, test entries and compiled object identities.
    """
    return {
        "paired_manifest": product["manifest"],
        "roles": {
            role: {
                field: product[role][field]
                for field in ("canonical_plan_identity", "selected_build", "inputs")
            }
            for role in ("application", "tests")
        },
    }


def compare_observations(previous, current, previous_bytes):
    if previous.get("schema_version") != 2 or previous.get("scope") != current["scope"]:
        raise ValueError("comparison input is not a compatible physical observation")
    if previous.get("build_subject") != current["build_subject"]:
        raise ValueError("observations describe different semantic or link-build inputs")
    old_images = previous.get("images")
    if not isinstance(old_images, list) or len(old_images) != len(current["images"]):
        raise ValueError("comparison input has no complete paired image inventory")
    image_bytes_equal = all(
        {field: old[field] for field in ("role", "digest", "bytes")}
        == {field: new[field] for field in ("role", "digest", "bytes")}
        for old, new in zip(old_images, current["images"])
    )
    loader_dependencies_equal = all(
        old["loader_dependencies"] == new["loader_dependencies"]
        for old, new in zip(old_images, current["images"])
    )
    return {
        "prior_observation": digest(previous_bytes),
        "same_semantic_and_build_inputs": True,
        "image_bytes_reproduced": image_bytes_equal,
        "loader_dependencies_reproduced": loader_dependencies_equal,
        "physical_products_reproduced": image_bytes_equal and loader_dependencies_equal,
    }


def validate_receipt(path, receipt_bytes, validator):
    process = subprocess.run(
        [str(validator.resolve(strict=True)), "--validate-receipt", str(path.resolve(strict=True))],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    if process.stdout.decode("ascii").strip() != digest(receipt_bytes):
        raise ValueError("receipt changed or validator did not validate the exact observed bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-output", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--receipt-validator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    root = args.pair_output.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("pair output is not a directory")
    receipt_bytes = read_bounded(args.execution_receipt, MAX_OBSERVATION_BYTES)
    validate_receipt(args.execution_receipt, receipt_bytes, args.receipt_validator)
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported paired execution receipt")
    product = receipt["product"]
    if product.get("schema_version") != 1:
        raise ValueError("unsupported persisted native pair")
    images = [
        observe("application", image_path(root, "application"), product["application"]),
        observe("tests", image_path(root, "tests"), product["tests"]),
    ]
    record = {
        "schema_version": 2,
        "scope": "external_physical_observation_of_completed_native_pair",
        "execution_receipt": digest(receipt_bytes),
        "build_subject": build_subject(product),
        "images": images,
    }
    if args.compare is not None:
        previous_bytes = read_bounded(args.compare, MAX_OBSERVATION_BYTES)
        previous = json.loads(previous_bytes)
        record["comparison"] = compare_observations(previous, record, previous_bytes)
    output_bytes = canonical(record)
    if len(output_bytes) > MAX_OBSERVATION_BYTES:
        raise ValueError("physical observation exceeds its wire bound")
    with args.out.open("xb") as output:
        output.write(output_bytes)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f"native pair observation failed: {error}", file=sys.stderr)
        sys.exit(1)
