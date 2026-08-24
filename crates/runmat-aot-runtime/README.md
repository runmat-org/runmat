# runmat-aot-runtime

`runmat-aot-runtime` is the process-image host linked into a standalone RunMat program. It decodes and verifies bounded Native IR, function-registry, and resume-point payloads emitted from the canonical executable product, binds the linked `runmat_aot_entry`, and executes it through `runmat-native-executor` and `runmat-runtime` semantics.

The crate is built as both an `rlib` for tests and a `staticlib` for distribution. The static archive is an internal build product embedded into the matching `runmat` executable; it is not a public SDK or a separately installed runtime. Target, ABI, catalog, archive, and linker compatibility are validated by the AOT orchestration layer before linking.

Native-specialized executables use the archive's ordinary runtime discovery. Closed-world executables instead reference the exact stable symbols derived from reachable catalog binding identities and install those bindings as an invocation-scoped authority, so nested builtin calls cannot fall back to the process-global registry. The linker extracts only referenced archive members and dead-strips unused sections. Runtime Native IR decoding retains its portable HIR/MIR operation schema, but parser/lowering/static-analysis, Core, object-emission, VM, and JIT code are not part of the closed-world process image.

This crate does not compile source, lower MIR, emit objects, select reachability, discover a linker, or own a second builtin/value/runtime implementation. Its exported C boundary copies and validates every bounded linked payload before decoding and catches panics before they can unwind into the generated launcher.
