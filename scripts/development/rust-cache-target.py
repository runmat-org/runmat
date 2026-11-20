#!/usr/bin/env python3
"""Keep the Windows Rust target cache complete and bounded.

`seal` runs after the tests and before rust-cache saves the target. It removes
compiler outputs in `debug/deps` that no Cargo invocation in this job used,
drops the target entirely when it is still larger than the restore budget, and
writes a manifest of the remaining compiler outputs with their sizes.

`verify` runs after rust-cache restores and before any Cargo command. A restore
that failed partway (for example, `tar` exiting with "No space left on device")
leaves missing or truncated files behind while Cargo still trusts the restored
fingerprints. When any manifest entry is missing or has the wrong size, or the
target was restored without a manifest, `verify` clears the target and the
Cargo registry so the job builds cold and correct.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import stat
import sys

MANIFEST_NAME = "target-manifest.tsv"
MANIFEST_HEADER = "# runmat rust-cache target manifest v1"
# Compiler outputs carry Cargo's 16-hex metadata hash: libfoo-<hash>.rlib,
# foo-<hash>.d, foo-<hash>.dll, and build/foo-<hash>/ directories.
HASHED_NAME = re.compile(r"^.+-([0-9a-f]{16})(?:\.[^/\\]*)?$")
# rust-cache keeps these profile directories; it removes everything else
# before saving, so the manifest covers only these.
SEALED_DIRECTORIES = ("deps", "build", ".fingerprint")


def _hash_of(name: str) -> str | None:
    match = HASHED_NAME.match(name)
    return match[1] if match else None


def _remove(path: Path) -> None:
    def make_writable(function, failed_path, _error):
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=make_writable)
    else:
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()


def _clear_contents(directory: Path) -> None:
    """Empty a directory but keep it, so its inherited NTFS compression stays set."""
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        _remove(child)


def _profile_directories(target: Path) -> list[Path]:
    return [profile for profile in (target / "debug", target / "release") if profile.is_dir()]


def _sealed_files(target: Path) -> list[tuple[str, int]]:
    """List the compiler outputs rust-cache keeps, as (relative path, size)."""
    entries: list[tuple[str, int]] = []
    for profile in _profile_directories(target):
        for name in SEALED_DIRECTORIES:
            directory = profile / name
            if not directory.is_dir():
                continue
            for child in directory.iterdir():
                if _hash_of(child.name) is None:
                    continue
                files = [child] if child.is_file() else [p for p in child.rglob("*") if p.is_file()]
                for file in files:
                    if file.is_symlink():
                        continue
                    entries.append((file.relative_to(target).as_posix(), file.stat().st_size))
    return sorted(entries)


def _logical_size(target: Path) -> int:
    return sum(file.stat().st_size for file in target.rglob("*") if file.is_file() and not file.is_symlink())


def used_hashes(ledger: Path) -> set[str]:
    """Collect the Cargo metadata hashes of every artifact this job reported."""
    hashes: set[str] = set()
    if not ledger.is_file():
        return hashes
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        for part in re.split(r"[/\\]", line.strip()):
            found = _hash_of(part)
            if found:
                hashes.add(found)
    return hashes


def prune_unused(target: Path, hashes: set[str]) -> tuple[int, int]:
    """Remove `deps` outputs whose metadata hash no Cargo invocation reported.

    Cargo treats a missing output as stale and rebuilds that unit, so a pruned
    variant costs a rebuild, never a wrong build. Files without a hash (such as
    shader compiler DLLs copied beside test executables) stay.
    """
    removed = 0
    freed = 0
    for profile in _profile_directories(target):
        deps = profile / "deps"
        if not deps.is_dir():
            continue
        for child in deps.iterdir():
            found = _hash_of(child.name)
            if found is None or found in hashes or not child.is_file():
                continue
            freed += child.stat().st_size
            _remove(child)
            removed += 1
    return removed, freed


def seal(target: Path, manifest_directory: Path, ledger: Path, cap_bytes: int) -> int:
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest = manifest_directory / MANIFEST_NAME
    hashes = used_hashes(ledger)
    if not target.is_dir():
        print(f"No Rust target at {target}; nothing to seal")
    elif not hashes:
        print(f"Cargo artifact ledger {ledger} is empty or missing; keeping every deps output")
    else:
        removed, freed = prune_unused(target, hashes)
        print(f"Pruned {removed} unused deps outputs ({freed / 1024 ** 3:.1f} GiB); "
              f"{len(hashes)} artifact hashes used by this job")
    if target.is_dir():
        size = _logical_size(target)
        print(f"Rust target to cache: {size / 1024 ** 3:.1f} GiB logical; restore budget {cap_bytes / 1024 ** 3:.1f} GiB")
        if size > cap_bytes:
            # A target this large cannot be restored beside the next build.
            # Cache the registry only; the next job builds the target cold.
            print(f"::warning::Rust target {size / 1024 ** 3:.1f} GiB exceeds the "
                  f"{cap_bytes / 1024 ** 3:.1f} GiB restore budget; caching without the target")
            _clear_contents(target)
    entries = _sealed_files(target) if target.is_dir() else []
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(MANIFEST_HEADER + "\n")
        for relative, size in entries:
            stream.write(f"{relative}\t{size}\n")
    print(f"Sealed {len(entries)} compiler outputs in {manifest}")
    return 0


def _read_manifest(manifest: Path) -> list[tuple[str, int]] | None:
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or lines[0] != MANIFEST_HEADER:
        return None
    entries = []
    for line in lines[1:]:
        relative, separator, size = line.rpartition("\t")
        if not separator or not size.isdigit():
            return None
        entries.append((relative, int(size)))
    return entries


def incomplete_entries(target: Path, entries: list[tuple[str, int]]) -> list[str]:
    problems = []
    for relative, size in entries:
        path = target / relative
        try:
            actual = path.stat().st_size
        except OSError:
            problems.append(f"missing {relative}")
            continue
        if actual != size:
            problems.append(f"size {actual} != {size} for {relative}")
    return problems


def verify(target: Path, manifest_directory: Path, cargo_home: Path | None) -> int:
    manifest = manifest_directory / MANIFEST_NAME
    restored_target = target.is_dir() and any(target.iterdir())
    if not restored_target and not manifest.exists():
        print("No Rust target cache was restored; building cold")
        return 0
    entries = _read_manifest(manifest)
    if entries is None:
        problems = [f"no readable completion manifest at {manifest}"]
    else:
        problems = incomplete_entries(target, entries)
    if not problems:
        print(f"Restored Rust target is complete: {len(entries)} sealed compiler outputs present")
        manifest.unlink()
        return 0
    print(f"::warning::Restored Rust cache is incomplete ({len(problems)} problems); "
          "clearing the target and Cargo registry so this job builds cold")
    for problem in problems[:20]:
        print(f"  {problem}")
    _clear_contents(target)
    if cargo_home is not None:
        # The same interrupted extraction can truncate registry archives.
        for name in ("registry", "git"):
            directory = cargo_home / name
            if directory.exists():
                _remove(directory)
    if manifest.exists():
        manifest.unlink()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--target", type=Path, required=True)
        command.add_argument("--manifest-dir", type=Path, required=True)
        if name == "seal":
            command.add_argument("--ledger", type=Path, required=True)
            command.add_argument("--cap-gib", type=float, required=True)
        else:
            command.add_argument("--cargo-home", type=Path)
    args = parser.parse_args(argv)
    if args.command == "seal":
        return seal(args.target, args.manifest_dir, args.ledger, int(args.cap_gib * 1024 ** 3))
    return verify(args.target, args.manifest_dir, args.cargo_home)


if __name__ == "__main__":
    sys.exit(main())
