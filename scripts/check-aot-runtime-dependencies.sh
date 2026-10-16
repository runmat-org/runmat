#!/usr/bin/env bash
set -euo pipefail

dependency_tree="$(cargo tree -p runmat-aot-runtime --target all --edges normal --prefix none)"
# This is the default, pure linked-image closure. The optional
# `source-evaluation` component has a separate, explicit frontend grant; it
# must not silently become a dependency of every native executable.
for forbidden in runmat-core runmat-hir runmat-parser runmat-lexer runmat-jit runmat-vm; do
  if grep -Eq "^${forbidden} v" <<<"${dependency_tree}"; then
    echo "runmat-aot-runtime must not depend on ${forbidden}" >&2
    exit 1
  fi
done

feature_tree="$(cargo tree -p runmat-aot-runtime --target all --edges normal,features --prefix none)"
for forbidden in cranelift 'runmat-native-codegen feature "compiler"'; do
  if grep -Fq "${forbidden}" <<<"${feature_tree}"; then
    echo "runmat-aot-runtime must not enable ${forbidden}" >&2
    exit 1
  fi
done

echo "runmat-aot-runtime pure dependency boundary is frontend-free"
