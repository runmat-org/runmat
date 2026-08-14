# runmat-native-codegen

`runmat-native-codegen` owns RunMat's deterministic generic Native IR, its verifier, MIR lowering, ABI compatibility binding, liveness, and deterministic textual form. It consumes canonical MIR, `AnalysisStore`, executable manifests, shared facts/contracts, and the runtime-owned native ABI. It never performs private type or builtin inference.

Decoded IR must pass all three relevant checks: `NativeAssembly::verify` for structural invariants, `verify_against_manifest` for executable identity and requirements, and `verify_against_mir` for independent construct, local, binding, and canonical-name completeness. Target-specific cache keys bind the complete executable key to the Native IR schema and runtime ABI/layout. Region boundaries bind live values and guards to exact SSA/frame state, while conditionally embedded short-circuit constructs remain explicitly inventoried.

Cranelift lowering, object emission, and optimization modules are added in their owning RM-1064 slices. Hotness, tiering, entry cells, provider placement, executable-memory lifecycle, and runtime language semantics remain outside this crate.

On native targets, generic Cranelift lowering compiles the verified Native IR block graph directly. Generated entrypoints use the runtime-owned ABI and identify semantic work with a typed, path-independent site request; a host adapter maps that identity back to immutable Native IR and shared runtime operations. Generated code does not embed MIR payload layouts, runtime `Value` storage, or Turbine's legacy value ABI. Web/WASM keeps the same verified Native IR product and does not compile host-native Cranelift modules.

Native IR function metadata retains every ordered MIR local together with its local kind and optional semantic binding identity/name. Bound locals must have a canonical HIR-derived name at lowering time. The verifier rejects incomplete metadata, so workspace, global, persistent, and workspace-first name resolution never depends on executor-specific VM slots. Native IR schema version 2 introduced this serialized local catalog.
