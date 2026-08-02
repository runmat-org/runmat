# runmat-test-runner-native

Native host adapters for `runmat-test-runner`.

This crate owns operating-system process creation and cleanup, bounded pipe
transport, filesystem report/artifact storage, native clocks and cancellation,
and telemetry forwarding. Test selection, lifecycle, retries, result semantics,
report formatting, and exit policy remain in portable or product-layer crates.
