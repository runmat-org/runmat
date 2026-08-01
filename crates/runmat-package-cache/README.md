# runmat-package-cache

`runmat-package-cache` is the portable cache policy authority for RunMat packages. It owns canonical cache metadata, optimistic transaction semantics, object verification, materialization transitions, leases, pins, quota/corruption records, recovery, and garbage-collection planning.

```text
runmat-package ──► runmat-package-cache ◄── native/browser composition roots
                              │
                   transactional backend port
                         ╱                 ╲
        runmat-package-cache-native     IndexedDB adapter
```

The crate compiles for native and `wasm32-unknown-unknown`. It contains no native paths, SQLite, filesystem locks, Git transport, IndexedDB, JavaScript globals, authentication, CLI rendering, or Desktop state. Backends atomically compare-and-swap a complete metadata snapshot together with object byte writes/deletes. Conflicts are ordinary retryable outcomes; quota and corruption are explicit domain results.

`lib.rs` is a small re-export façade. Object models, state, backend protocol, lease operations, materialization, archive policy, and GC stay in bounded modules rather than accumulating in a cache godfile.
