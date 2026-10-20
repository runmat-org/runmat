# runmat-test-runner-native

Native host adapters for `runmat-test-runner`.

This crate owns operating-system process creation and cleanup, bounded pipe
transport, filesystem report/artifact storage, native clocks and cancellation,
and telemetry forwarding. Test selection, lifecycle, retries, result semantics,
report formatting, and exit policy remain in portable or product-layer crates.

The `native_paired` example is an explicit host integration witness:

```sh
cargo run -p runmat-test-runner-native --example native_paired -- \
  /absolute/pack.json /absolute/components /absolute/host-admission.json \
  /absolute/existing-empty-output-directory
```

Supply a matching target-built runtime pack, its complete selected components,
and independently issued host admission including qualified toolchain/sysroot
trees and dependency locators. The example does not generate qualification or
copy permissions from the pack. Its default object optimization is Speed; pack
optimization must match. Use the same source/features when producing the runtime
archive and running the witness. A missing or stale input is an error, not a
skipped test. Existing image outputs are never overwritten.

The witness discovers three real local procedures from one frozen source,
compiles both native roles, runs the application image, then uses the framed
target-test receiver to check passing arithmetic, typed assertion failure and
language failure. Both process paths rehash the actual linker bytes and execute
private copies with bounded console capture, deadlines and owned process-tree
retirement. Receiver results report only installed managed-ledger peaks, not RSS
or a full-call finite-memory certificate. No application memory bound is claimed.

This is a hosted local witness, not deterministic-service qualification. The
receiver currently admits only the linked SourceEvaluation service (or a pure
selection); Clock and other selected services remain unsupported and are refused
before invocation. Generic portable TestAttemptV1 workers remain a different
protocol/product. Linux/Windows execution and deployed finite-policy coverage
require their own matching qualification.
