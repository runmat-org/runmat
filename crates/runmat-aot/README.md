# RunMat AOT orchestration

This host-only crate composes verified native user objects with the exact embedded RunMat runtime archive. Archive identity, target/ABI/schema/catalog compatibility, payload compression, linker discovery, response-file construction, temporary inputs, diagnostics, and atomic output publication live in separate bounded modules.

The runtime archive is an internal two-phase build product. A matching compressed payload and manifest may be supplied through `RUNMAT_AOT_RUNTIME_ARCHIVE` and `RUNMAT_AOT_RUNTIME_MANIFEST`; the crate copies them into Cargo's output directory and embeds them into the consuming RunMat binary. Without both inputs, normal developer builds remain valid and the compile workflow reports that this RunMat build has no embedded native runtime. The archive is never discovered from an installation directory and is not a public SDK.

Native system linking is host-only. Browser/WASM consumers continue to use portable execution artifacts and receive explicit native-capability absence; this crate does not enter the portable dependency graph.

Selected paired products use `compile_operator_selected_pair`: the existing
bounded `selected_admission` loader consumes a caller-issued host admission,
selected-pack manifest and component root. The CLI selected compiler uses this
same loader. The pack cannot grant its own tool qualification, source-loading
capability or storage allowance. Both linked roles have completed-image digests;
the pair is not an atomic two-file publication transaction.

The `runmat-aot-pack --selected-tree` and `--selected-pack` producer modes build
exact tool/sysroot tree metadata and runtime components from explicit build-owner
declarations. They do not qualify those products or issue host admission. In
particular, `scripts/build-runmat-with-aot-runtime.sh --profile dev` builds the
ordinary embedded archive; that alone does not supply a selected pack or an
independent toolchain qualification.
