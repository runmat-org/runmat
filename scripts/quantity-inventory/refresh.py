#!/usr/bin/env python3
"""Refresh discovery atomically; never overwrite classifications or claim closure."""
import json
import os
from pathlib import Path
import re
import subprocess
from review_ledger import apply, retain_existing

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "quantity-migration"
CONFIG = json.loads((DEST / "scope.json").read_text())


def replace_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def refresh():
    ledger_path = DEST / "reviews.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"schema_version": 1, "rust": {}, "references": {}}
    for name, reference in (("inventory", False), ("references", True)):
        old_path = DEST / (name + ".json")
        if old_path.exists():
            retain_existing(json.loads(old_path.read_text())["records"], ledger["references" if reference else "rust"], reference)
    destination = DEST / "inventory.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w") as output:
        subprocess.run([
            "cargo", "run", "--offline", "--manifest-path",
            str(ROOT / "scripts/quantity-inventory/Cargo.toml"), "--bin", "runmat-quantity-inventory", "--",
            *CONFIG["rust_ast_roots"],
        ], cwd=ROOT, stdout=output, check=True)
    # Parse before publishing a new inventory; retain the last successful file.
    inventory = json.loads(temporary.read_text())
    apply(inventory["records"], ledger["rust"])
    replace_json(destination, inventory)
    temporary.unlink(missing_ok=True)

    names = CONFIG["representation_names"]
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b|\b(?:geometryUnits|geometry_units|source_units|source_unit|coordinateUnits|meters_per_source_unit)\b")
    records = []
    scanned = 0
    for source_root in CONFIG["reference_roots"]:
        for directory, subdirs, files in os.walk(ROOT / source_root, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if d not in {"target", "node_modules", ".git", ".next", "dist", "public"})
            for name in sorted(files):
                source = Path(directory) / name
                if source.suffix not in {".rs", ".ts", ".tsx", ".json", ".yaml"} or source.is_symlink():
                    continue
                scanned += 1
                relative = os.path.relpath(source, ROOT)
                for line, text in enumerate(source.read_text().splitlines(), 1):
                    matches = sorted(set(match.group() for match in pattern.finditer(text)))
                    if matches:
                        records.append({"path": relative, "line": line, "representations": matches, "source": text.strip(), "disposition": "unclassified"})
    apply(records, ledger["references"], True)
    replace_json(ledger_path, ledger)
    replace_json(DEST / "references.json", {
        "schema_version": 1, "scope": CONFIG["reference_roots"], "files_scanned": scanned,
        "limits": ["Lexical references complement the Rust AST inventory; comments and tests are included.", "This is not inferred type resolution or a complete inventory of physical arithmetic."],
        "records": records,
    })
    replace_json(DEST / "summary.json", {
        "status": "discovery_requires_classification",
        "rust_files_parsed": inventory["files_parsed"], "rust_candidate_counts": inventory["counts"],
        "reference_files_scanned": scanned, "reference_sites": len(records),
        "unclassified_rust_candidates": sum(r["disposition"] == "unclassified" for r in inventory["records"]),
        "unclassified_reference_sites": sum(r["disposition"] == "unclassified" for r in records),
        "migration_complete": False,
    })
    print(json.dumps(json.loads((DEST / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    refresh()
