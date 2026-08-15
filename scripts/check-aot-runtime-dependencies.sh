#!/usr/bin/env bash
set -euo pipefail

dependency_tree="$(cargo tree -p runmat-aot-runtime --edges normal --prefix none)"
for forbidden in runmat-jit runmat-vm; do
  if grep -Eq "^${forbidden} v" <<<"${dependency_tree}"; then
    echo "runmat-aot-runtime must not depend on ${forbidden}" >&2
    exit 1
  fi
done

echo "runmat-aot-runtime dependency boundary is clean"
