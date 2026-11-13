#!/usr/bin/env bash
# Run once against the frozen candidate; ordinary CI verifies its committed receipt.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${RUNMAT_SCIENTIFIC_OUTPUT:?Set a new evidence output directory}"
: "${RUNMAT_QUALIFY_NATIVE_TARGET:?Set the qualified Rust target triple}"
test ! -e "$RUNMAT_SCIENTIFIC_OUTPUT"
mkdir -p "$RUNMAT_SCIENTIFIC_OUTPUT"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 RUST_TEST_THREADS=1
export CARGO_BUILD_TARGET="$RUNMAT_QUALIFY_NATIVE_TARGET"
phase() {
  local name="$1"; shift
  set +e
  "$@" > "$RUNMAT_SCIENTIFIC_OUTPUT/$name.log" 2>&1
  local code=$?
  set -e
  printf '%s\t%s\n' "$name" "$code" >> "$RUNMAT_SCIENTIFIC_OUTPUT/results.tsv"
  test "$code" = 0
}
phase algorithm-conformance cargo test --locked --offline --release --all-targets \
  -p runmat-meshing-core -p runmat-meshing-size -p runmat-meshing-curve \
  -p runmat-meshing-surface -p runmat-meshing-tetrahedron -p runmat-meshing-execution
# The required driver-recovery-physics cohort runs all three driver-replacement
# cases. Reuse that receipt rather than repeat those long solves here.
for test in \
  random_response_reaches_an_admitted_result_over_encrypted_quic \
  response_spectrum_fields_reach_an_admitted_result_over_encrypted_quic; do
  phase "$test" cargo test --locked --offline --release -p runmat-execution-runner-native --all-features --lib "$test"
done
for test in random_response_qualification response_spectrum::qualification pseudo_acceleration_missing_mass; do
  phase "${test//:/_}" cargo test --locked --offline --release -p runmat-analysis-fea --all-features --lib "$test"
done
for test in \
  structural_static_step_documentation_example_runs_and_publishes_fields \
  response_spectrum_documentation_example_runs_its_two_step_study; do
  phase "$test" cargo test --locked --offline --release -p runmat --all-features --test check_cli "$test"
done
phase exact-process-conformance cargo test --locked --offline --release -p runmat-execution-runner-native \
  --all-features --test occt_exact_meshing --test meshing_process_conformance
# Use the same feature graph/cache as the release CLI rather than constructing
# a second no-default-features runtime solely for the two scientific probes.
phase build-probes cargo build --locked --offline --release -p runmat \
  --features "${RUNMAT_QUALIFICATION_FEATURES:-blas-lapack,vendored-openssl,occt-native}" \
  --example meshing-verification --example meshing-corpus
examples="${CARGO_TARGET_DIR:-target}/$RUNMAT_QUALIFY_NATIVE_TARGET/release/examples"
phase complete-corpus env RUNMAT_EXECUTION_STATE_DIR="$RUNMAT_SCIENTIFIC_OUTPUT/corpus-state" \
  "$examples/meshing-corpus" --tier complete --report "$RUNMAT_SCIENTIFIC_OUTPUT/integration-corpus-linux-x86_64.json"
phase stable-performance env RUNMAT_EXECUTION_STATE_DIR="$RUNMAT_SCIENTIFIC_OUTPUT/performance-state" \
  python3 scripts/meshing/performance.py --binary "$examples/meshing-verification" \
  --profile stable --samples 15 --report "$RUNMAT_SCIENTIFIC_OUTPUT/performance-linux-x86_64.json"
