#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <executable> <link-plan.json>" >&2
  exit 2
fi

binary_path="$1"
plan_path="$2"
if [[ ! -f "$binary_path" || ! -f "$plan_path" ]]; then
  echo "closed-world binary and link plan must both exist" >&2
  exit 2
fi
if ! command -v nm >/dev/null || ! command -v jq >/dev/null; then
  echo "closed-world verification requires nm and jq" >&2
  exit 2
fi
if ! jq -e '.schema_version == 7 and (.runtime_composition.selected.bindings | type == "array")' "$plan_path" >/dev/null; then
  echo "link plan does not select exact runtime composition" >&2
  exit 1
fi

symbol_file="$(mktemp "${TMPDIR:-/tmp}/runmat-closed-world-symbols.XXXXXX")"
expected_file="$(mktemp "${TMPDIR:-/tmp}/runmat-closed-world-expected.XXXXXX")"
trap 'rm -f "$symbol_file" "$expected_file"' EXIT

case "$(uname -s)" in
  Darwin) nm -gU "$binary_path" >"$symbol_file" ;;
  *) nm -g --defined-only "$binary_path" >"$symbol_file" ;;
esac
jq -r '.runtime_composition.selected.bindings[].native_symbol' "$plan_path" | sort -u >"$expected_file"

# Inspect every binding ABI version: an obsolete symbol must appear in the
# exact-plan comparison rather than being silently excluded by the scanner.
actual_bindings="$(sed -n 's/^.*[_ ]\(runmat_builtin_binding_v[[:digit:]][[:digit:]]*_[[:alnum:]_]*\)$/\1/p' "$symbol_file" | sort -u)"
expected_bindings="$(cat "$expected_file")"
if [[ "$actual_bindings" != "$expected_bindings" ]]; then
  echo "linked builtin symbols do not match the exact link plan" >&2
  diff -u "$expected_file" <(printf '%s\n' "$actual_bindings") >&2 || true
  exit 1
fi

forbidden_pattern='runmat_(vm|jit|core|parser)|runmat_hir[0-9]+(lowering|inference|validation)|runmat_mir[0-9]+(lowering|analysis)|cranelift'
if rg -q "$forbidden_pattern" "$symbol_file"; then
  echo "closed-world executable retained compiler, VM, or JIT symbols" >&2
  rg "$forbidden_pattern" "$symbol_file" >&2
  exit 1
fi

echo "closed-world binary matches its exact builtin plan and omits compiler, VM, and JIT symbols"
