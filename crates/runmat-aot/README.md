# RunMat AOT orchestration

This host-only crate composes verified native user objects with the exact embedded RunMat runtime archive. Archive identity, target/ABI/schema/catalog compatibility, payload compression, linker discovery, response-file construction, temporary inputs, diagnostics, and atomic output publication live in separate bounded modules.

The runtime archive is an internal two-phase build product. A matching compressed payload and manifest may be supplied through `RUNMAT_AOT_RUNTIME_ARCHIVE` and `RUNMAT_AOT_RUNTIME_MANIFEST`; the crate copies them into Cargo's output directory and embeds them into the consuming RunMat binary. Without both inputs, normal developer builds remain valid and the compile workflow reports that this RunMat build has no embedded native runtime. The archive is never discovered from an installation directory and is not a public SDK.

Native system linking is host-only. Browser/WASM consumers continue to use portable execution artifacts and receive explicit native-capability absence; this crate does not enter the portable dependency graph.
