#!/usr/bin/env bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
cd "${repo_root}"

if ! command -v rg >/dev/null 2>&1; then
  echo "integer-storage-census: rg is required" >&2
  exit 1
fi

count_metric() {
  local metric=$1
  local scope=$2
  local pattern=$3
  shift 3

  local files
  local lines
  local matches

  files=$( (rg -l "${pattern}" "$@" --glob '*.rs' || true) | wc -l | tr -d ' ')
  lines=$( (rg -n "${pattern}" "$@" --glob '*.rs' || true) | wc -l | tr -d ' ')
  matches=$( (rg -o "${pattern}" "$@" --glob '*.rs' || true) | wc -l | tr -d ' ')

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${metric}" "${scope}" "${files}" "${lines}" "${matches}"
}

printf 'metric\tscope\tfiles\tmatching_lines\tmatches\n'

count_metric \
  dense_tensor_constructors \
  crates \
  '(^|[^[:alnum:]_])Tensor::(new|new_with_dtype|from_f32|from_f32_slice|new_integer)\(' \
  crates

count_metric \
  strong_named_direct_data \
  crates \
  '\b[a-zA-Z_][a-zA-Z0-9_]*tensor\.data\b|\bt\.data\b' \
  crates

count_metric \
  runtime_value_tensor \
  crates/runmat-runtime/src \
  'Value::Tensor' \
  crates/runmat-runtime/src

count_metric \
  runtime_integer_storage_calls \
  crates/runmat-runtime/src \
  'integer_storage\(' \
  crates/runmat-runtime/src

count_metric \
  runtime_floating_materialization \
  crates/runmat-runtime/src \
  'materialize_f64|tensor_values_f64|to_f64_vec|to_f64\(' \
  crates/runmat-runtime/src

count_metric \
  runtime_direct_materialize_f64 \
  crates/runmat-runtime/src \
  'materialize_f64' \
  crates/runmat-runtime/src

count_metric \
  runtime_tensor_values_f64 \
  crates/runmat-runtime/src \
  'tensor_values_f64' \
  crates/runmat-runtime/src

count_metric \
  runtime_to_f64_vec \
  crates/runmat-runtime/src \
  'to_f64_vec' \
  crates/runmat-runtime/src

count_metric \
  runtime_scalar_to_f64 \
  crates/runmat-runtime/src \
  'to_f64\(' \
  crates/runmat-runtime/src

count_metric \
  floating_provider_view \
  crates \
  'HostTensorView' \
  crates

count_metric \
  integer_provider_view \
  crates \
  'HostIntegerTensorView' \
  crates

count_metric \
  unified_numeric_provider_view \
  crates \
  'HostNumericTensorView' \
  crates

count_metric \
  unified_numeric_provider_owned \
  crates \
  'HostNumericTensorOwned' \
  crates
