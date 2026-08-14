# runmat-aot-runtime

`runmat-aot-runtime` is the process-image host linked into a standalone RunMat program. It decodes and verifies bounded Native IR, function-registry, and resume-point payloads emitted from the canonical executable product, binds the linked `runmat_aot_entry`, and executes it through `runmat-jit`'s ordinary invocation host and `runmat-runtime` semantics.

The crate is built as both an `rlib` for tests and a `staticlib` for distribution. The static archive is an internal build product embedded into the matching `runmat` executable; it is not a public SDK or a separately installed runtime. Target, ABI, catalog, archive, and linker compatibility are validated by the AOT orchestration layer before linking.

This crate does not compile source, lower MIR, emit objects, select reachability, discover a linker, or own a second builtin/value/runtime implementation. Its exported C boundary copies and validates every bounded linked payload before decoding and catches panics before they can unwind into the generated launcher.
