# runmat-jit

`runmat-jit` owns native compilation policy, publication, execution hosts, runtime feedback, entry cells, invalidation, and executable-memory lifecycle. It consumes verified products from `runmat-native-codegen` and the runtime-owned native ABI. It does not lower MIR, define runtime value storage, own MATLAB semantics, or reconstruct bytecode control flow.

The generic host is split by responsibility: compilation retains executable memory, each invocation owns an opaque generation-checked value arena and ABI frame, callbacks contain panics at the C boundary, and semantic-site execution consumes the immutable Native IR payload through shared runtime operations. Rust `Value` layouts never enter generated code or the native ABI. Unsupported semantic cohorts fail explicitly; they are never replayed through the bytecode interpreter after native entry.

Operators, calls, aggregates, conditions, and MATLAB column iteration stay runtime-owned. The JIT adapters only materialize Native IR operands, preserve requested-output contracts, and select compiled edges. Loop snapshots are invocation-owned: an iterable is evaluated once, retained across compiled backedges, and retired on exhaustion or any edge leaving the natural loop body. Cancellation is checked at every generic semantic site, with exact frame roots, resume points, source spans, and side-effect epochs refreshed around host work.

Host-native executable memory is unavailable in a web/WASM process. WASM consumers retain and verify the same portable executable and generic Native IR products; browser execution uses its platform executor rather than embedding a native Cranelift JIT.
