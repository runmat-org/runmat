#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="${repo_root}/docs/builtins/meta.json"

jq -r '
  def has_capabilities:
    ((.integer_capabilities // []) | length) > 0;
  def is_screen_candidate:
    [.signatures[].inputs[].ty]
    | any(
        . == "Any"
        or . == "NumericArray"
        or . == "NumericScalar"
        or . == "IntegerScalar"
        or . == "SizeArg"
        or . == "LikePrototype"
      );
  .builtins as $builtins
  | [
      ["metric", "count", "interpretation"],
      [
        "public_descriptor_records",
        ($builtins | length),
        "All checked-in public builtin descriptor records."
      ],
      [
        "integer_capability_builtin_names",
        ($builtins | map(select(has_capabilities)) | length),
        "Builtin names with one or more settled per-form integer capability records."
      ],
      [
        "integer_capability_forms",
        (
          $builtins
          | map(.integer_capabilities // [])
          | flatten
          | length
        ),
        "Settled call-form records across capability-bearing builtin names."
      ],
      [
        "signature_screen_candidates",
        ($builtins | map(select(is_screen_candidate)) | length),
        "Conservative triage population selected by numeric, Any, size, or prototype input descriptors."
      ],
      [
        "signature_screen_settled",
        (
          $builtins
          | map(select(is_screen_candidate and has_capabilities))
          | length
        ),
        "Screen-positive builtin names already carrying capability records."
      ],
      [
        "signature_screen_untriaged",
        (
          $builtins
          | map(select(is_screen_candidate and (has_capabilities | not)))
          | length
        ),
        "Screen-positive audit queue; this is an upper bound, not a count of missing integer overloads or defects."
      ],
      [
        "signature_screen_excluded",
        ($builtins | map(select(is_screen_candidate | not)) | length),
        "Descriptors with no screened numeric, Any, size, or prototype input."
      ]
    ]
  | .[]
  | @tsv
' "${catalog}"
