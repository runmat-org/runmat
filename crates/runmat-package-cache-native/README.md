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
