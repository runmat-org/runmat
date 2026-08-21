# runmat-value

`runmat-value` owns RunMat's sole live runtime value and its intrinsic portable storage model. It is intentionally independent of builtin registration, static analysis, session state, executable dispatch, placement policy, provider implementations, and host facilities.

The crate is organized by value domain:

- `numeric/` owns exact scalar, dtype, integer, storage, and typed-view contracts;
- `array/` owns dense, sparse, complex, logical, character, string, and symbolic arrays;
- `aggregate/`, `object/`, and `callable/` own recursive live payloads;
- `symbolic/` owns intrinsic symbolic expression representation;
- `value.rs`, `exception.rs`, and `trace.rs` compose the live sum type, error payload, and GC traversal.

Public APIs expose typed storage and explicit materialization boundaries; private representation is not an ABI. The crate may depend on lightweight handle/API crates, but never on builtin catalogs, HIR/MIR/VM, runtime services, provider implementations, or host facilities. Display formatting is the one explicitly temporary R03 accommodation: Rust orphan rules require the existing `Display` implementation to remain beside `Value`; R09 replaces its session-format policy with the planned presentation boundary.
