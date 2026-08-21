# Packages In CI, Offline Use, And Vendoring

CI should treat the manifest and lock as source, the shared package cache as an optional verified acceleration, and vendor output as an explicit hermetic input.

## Locked CI

Use an isolated cache root and reject lock drift:

```bash
export RUNMAT_PACKAGE_CACHE_DIR="$CI_PROJECT_DIR/.cache/runmat-packages"
runmat --locked package fetch
runmat --locked check src/main.m
runmat --locked run src/main.m
```

Cache the directory using a key that includes the operating-system family, RunMat cache schema/client compatibility, and a digest of `runmat.lock`. Restoring no cache is safe but slower. Restoring an older or partial cache is also safe: RunMat verifies metadata/payload agreement and exact content digests before reuse, treats missing/evicted content as a miss, and transactionally fills only allowed objects.

Do not use a cache restore as a lock substitute, cache credentials with package objects, or persist browser plaintext private mounts. A compromised cache can cause a verification failure or redownload but cannot change the locked accepted identity without also defeating digest/signature policy.

## Frozen And Offline Jobs

After `package fetch` succeeds, test the no-network closure:

```bash
runmat --frozen check src/main.m
runmat --frozen run src/main.m
```

`--offline` allows an absent lock to be reconciled only from locally available exact content where policy permits. `--frozen` requires a compatible existing lock, performs no network access, and never writes it. Run a frozen job in release pipelines to prove the cache or vendor input is complete.

## Vendoring

For air-gapped or reviewable source inputs:

```bash
runmat package vendor
git add vendor runmat-vendor.json runmat.lock
runmat --frozen run src/main.m
```

`runmat-vendor.json` binds project-relative copies to the exact graph and source identities. Frozen execution rejects a missing copy, stale graph, identity mismatch, changed manifest, or changed tree contents. Regenerate vendor output through the package command; do not hand-edit its manifest.

## Mirrors

Configure mirrors through named registries and explicit source replacement in the manifest or approved organization policy. Preserve the original package identity and require the mirror’s candidate metadata/artifact to satisfy the same locked release, tree, signature, provenance, license, and advisory policy. CI credentials belong in the host secret provider and are scoped to the exact mirror origin.

## Browser CI

Browser package tests need a real browser because IndexedDB transactions, WebCrypto non-extractable keys, storage persistence/quota, reload, and Web Worker lifecycle are not faithfully represented by a plain Node mock. The minimum conformance flow resolves a project, verifies and mounts a package, executes through the frozen graph, observes lock writeback, reloads the page, reopens the same IndexedDB cache/filesystem, resolves with locked offline policy, confirms zero fetches, and compares the graph/source revision.

Exercise quota denial and eviction as recoverable diagnostics, private-key unavailability as a fail-closed error, and logout/project-switch/worker-disposal as private-mount invalidation. Product tests should use the public TypeScript/WASM composition APIs rather than reproducing solver or cache policy.
