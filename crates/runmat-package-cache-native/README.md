# runmat-package-cache-native

Native adapters for the portable `runmat-package-cache` policy.

```text
portable transactions ──► SQLite state + payloads
portable mount plan  ──► private staging ──► immutable tree
native coordination ──► narrow process lock
```

SQLite owns atomic cache publication and compare-and-swap serialization. Filesystem
modules own platform paths, private staging, atomic promotion, and read-only mounts.
Git transport and archive policy are separate bounded modules; this crate does not
solve dependencies or own package graph policy.

The default root is the platform cache directory under `runmat/packages`; hermetic hosts and CI may set `RUNMAT_PACKAGE_CACHE_DIR`. Exact Git snapshots are published to SQLite with their verified blobs before materialization. A locked snapshot can therefore replay offline even if the normalized bare Git database is unavailable. Concurrent processes use repository- and tree-scoped locks, verify an existing materialization before reuse, and expose only read-only promoted trees.
