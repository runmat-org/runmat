# runmat-package

`runmat-package` is RunMat's portable package-domain authority. It owns stable identities, manifest interpretation, deterministic lockfiles, frozen resolved graphs, solver policy, source ports, verification policy, and vendor plans. It contains no CLI rendering, native filesystem/cache/database implementation, HTTP authentication, Core compilation, LSP state, Desktop state, or private Server types.

```text
runmat-config ──► runmat-package ──► consumers at composition roots
                        ▲
                        │
             runmat-package-cache
                        ▲
                        │
          runmat-package-cache-native

runmat-package never depends on cache, Core, LSP, CLI, Desktop, Server, or native host adapters.
```

All deterministic policy is native/WASM portable. Source, cache, network, and clock access crosses explicit async ports. Native paths, virtual mount paths, browser origins, IndexedDB keys, credentials, signed URLs, and authorization material are excluded from stable identities and lockfiles.

The crate is intentionally divided by responsibility. `lib.rs` is a re-export façade; identity, lock, graph, resolution, source, verification, and vendor modules remain bounded owners rather than accumulating in a resolver or package godfile.
