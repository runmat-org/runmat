# runmat-package-cache

`runmat-package-cache` is the portable cache policy authority for RunMat packages. It owns canonical cache metadata, optimistic transaction semantics, object verification, materialization transitions, transactional lease acquisition/renewal/release, pins, quota/corruption records, recovery, and garbage-collection planning.

```text
runmat-package ──► runmat-package-cache ◄── native/browser composition roots
                              │
                   transactional backend port
                         ╱                 ╲
        runmat-package-cache-native     IndexedDB adapter
```

The crate compiles for native and `wasm32-unknown-unknown`. It contains no native paths, SQLite, filesystem locks, Git transport, IndexedDB, JavaScript globals, authentication, CLI rendering, or Desktop state. Backends atomically compare-and-swap a complete metadata snapshot together with object byte writes/deletes. Conflicts are ordinary retryable outcomes; quota and corruption are explicit domain results.

`lib.rs` is a small re-export façade. Object models, state, backend protocol, lease operations, materialization, archive policy, and GC stay in bounded modules rather than accumulating in a cache godfile.

Recovery is deterministic: an atomic backend transaction is the only publication point; staging state and payloads that do not form a complete verified closure are discarded; a missing or evicted payload is a typed cache miss; digest failure is corruption rather than a fallback to unverified bytes; active pins and unexpired leases are GC roots; and stale revision writers retry from a fresh snapshot. The same status and GC policy functions drive native CLI and browser surfaces.

Git archives and RunMat Server project snapshots enter through source-specific inventory adapters, but both are converted to the same validated blobs and canonical `TreeManifest` before publication. A Server source binds normalized service origin, project ID, exact snapshot ID, and tree digest. Authorization and deletion are acquisition concerns: no failed transfer is published, while a previously authorized exact snapshot can be replayed from local plaintext cache after later access loss or remote deletion.
