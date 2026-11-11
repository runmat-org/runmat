"""Run native CI for code inputs; verify prose and scientific evidence separately."""

import json
import os
from pathlib import PurePosixPath
import re
import subprocess


def non_native_changes_only(paths):
    paths = list(paths)
    return bool(paths) and all(
        path in {"README.md", "CHANGELOG.md"}
        or (path.startswith("docs/") and PurePosixPath(path).suffix.lower()
            in {".md", ".mdx", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"})
        # These JSON records bind measured results to source inputs. The
        # scientific-receipts job verifies them on every push. Policy, routing
        # scripts, quality exceptions and executable tests retain native CI.
        or (PurePosixPath(path).parent.as_posix() in {
                "qualification/scientific/receipts",
                "qualification/scientific/reviews",
                "qualification/scientific/evidence",
            } and PurePosixPath(path).suffix == ".json")
        for path in paths
    )


def native_required(event, event_name):
    if event_name not in {"push", "pull_request"}:
        return True
    if event_name == "pull_request":
        base = event.get("pull_request", {}).get("base", {}).get("sha", "")
        head = event.get("pull_request", {}).get("head", {}).get("sha", "")
    else:
        base, head = event.get("before", ""), event.get("after", "")
    if any(not re.fullmatch(r"[0-9a-f]{40}", sha) or sha == "0" * 40
           for sha in (base, head)):
        return True
    try:
        # Compare both trees, covering all commits in a push and deleted paths.
        for sha in (base, head):
            if subprocess.run(["git", "cat-file", "-e", sha], capture_output=True).returncode:
                subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", sha], check=True)
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", base, head]
        ).decode("utf-8").split("\0")
        return not non_native_changes_only(path for path in changed if path)
    except (subprocess.CalledProcessError, UnicodeError):
        # Missing history or unrecognized paths must retain full coverage.
        return True


if __name__ == "__main__":
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    required = native_required(event, os.environ["GITHUB_EVENT_NAME"])
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"native={str(required).lower()}\n")
    print(f"Native compilation required: {required}")
