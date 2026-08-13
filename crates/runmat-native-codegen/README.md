# runmat-native-codegen

`runmat-native-codegen` owns RunMat's deterministic generic Native IR, its verifier, MIR lowering, ABI compatibility binding, liveness, and deterministic textual form. It consumes canonical MIR, `AnalysisStore`, executable manifests, shared facts/contracts, and the runtime-owned native ABI. It never performs private type or builtin inference.

Decoded IR must pass all three relevant checks: `NativeAssembly::verify` for structural invariants, `verify_against_manifest` for executable identity and requirements, and `verify_against_mir` for independent construct completeness. Target-specific cache keys bind the complete executable key to the Native IR schema and runtime ABI/layout. Region boundaries bind live values and guards to exact SSA/frame state, while conditionally embedded short-circuit constructs remain explicitly inventoried.

Cranelift lowering, object emission, and optimization modules are added in their owning RM-1064 slices. Hotness, tiering, entry cells, provider placement, executable-memory lifecycle, and runtime language semantics remain outside this crate.
