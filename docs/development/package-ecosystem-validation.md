---
title: "Package Ecosystem Validation"
category: "Development"
section: "12.7"
last_updated: "August 1, 2026"
---

# Package Ecosystem Validation

This record separates package-system conformance from broader MATLAB language compatibility. A successful package smoke does not imply that every source or test in an upstream toolbox is supported.

## COBRA Toolbox Audit

The Phase 9 audit used the public `opencobra/cobratoolbox` repository at commit `67c790dbac809d9d891fdbafc33e18c21fc9bddc`.

- Production source: 1,578 `.m` files under `src`.
- Production plus upstream tests: 1,985 `.m` files under `src` and `test`.
- Sparse audit checkout size: 129 MB, with 32 MB of production source and 72 MB of test data/source.
- A temporary manifest freezing `src` completed and produced a deterministic lock, graph digest, and source revision.
- A frozen `runmat check` of `removeGeneVersions.m`, with the actual MATLAB lookup directories represented as separate roots, resolved `buildRxnGeneMat`, `GPRparser`, `buildGrRules`, and `generateRules` to their exact upstream files. Analysis completed with one intentional runtime-dependent method-dispatch warning and no errors.
- The exact upstream `src/base/utilities/getDefaultValue.m` content was placed alone in a temporary package source root and exercised through a clean frozen graph. Static analysis was clean, source navigation identified the exact stable source ID/path, and both interpreter and JIT-enabled CLI execution passed using the supported logical-input branch.

The full upstream toolbox is not claimed as executable. The audit found older MATLAB function files that omit an outer function `end`, native/solver integrations, and language/builtin dependencies outside this package-graph ticket. Those remain explicit compatibility work; the package system must not hide them by weakening parsing or resolution.

## Deterministic Scale Gate

`runmat-core` owns a checked-in generated large-project conformance test. It creates 1,024 explicit pure-MATLAB functions, freezes the complete catalog, validates the source count and revision handoff, installs that handoff as the only execution authority, and analyzes, compiles, resolves, and executes the final function. The generated fixture avoids vendoring third-party code while keeping large-catalog behavior in ordinary CI.

The browser TypeScript suite publishes 512 distinct verified MATLAB source blobs in one IndexedDB transaction, exposes them through an immutable tree mount, navigates the complete source directory, verifies representative files by digest, closes the database, reopens it, and confirms the cache revision and payload survive reload. Real Chromium conformance additionally resolves a Git-gateway package, writes `runmat.lock`, executes through the frozen graph, reloads the page, reopens the IndexedDB-backed filesystem/cache, and completes locked offline resolution with zero further fetches.

## Source And Operational Parity

The package suites cover the source/operation matrix independently of upstream language support:

- Path projects freeze checkout-independent graphs and source identities, enforce version assertions, preserve MATLAB package/class/private visibility, and execute through verified vendor copies under frozen policy.
- Git dependencies normalize remotes, select exact commits/subdirectories, publish verified trees transactionally, deduplicate concurrent acquisition, and resolve/check/execute offline from the shared cache.
- RunMat Server project snapshots bind service/project/snapshot/tree identity and replay after remote access or snapshot loss when exact verified content is already cached.
- Registry releases validate candidate metadata, artifact/manifest/tree/release identity, signature/provenance policy, transactional cache publication, encrypted private artifacts, and locked offline replay.
- Solver conformance applies explicit source replacements/mirrors before acquisition while preserving canonical package identity and deterministic outcomes.
- Native cache reopen/crash tests, browser IndexedDB close/reopen tests, vendor tamper tests, and frozen CLI tests cover CI cache restoration and hermetic replay.

## Reproducing The External Audit

1. Shallow clone `https://github.com/opencobra/cobratoolbox.git` at the recorded commit with blob filtering.
2. Sparse-check out `src` and `test`.
3. Record `.m` counts before selecting probes.
4. Create a temporary `runmat.toml`; map every ordinary MATLAB lookup directory as its own `[sources].roots` entry rather than treating `genpath` as one recursive namespace root.
5. Set `RUNMAT_PACKAGE_CACHE_DIR` to a temporary isolated directory.
6. Run package resolution, locked/frozen static analysis, interpreter execution, and JIT-enabled execution.
7. Record exact clean probes and every incompatibility. Do not patch third-party source and report it as upstream compatibility.
