# Builtin example verification

Builtin documentation examples are verified through an inventory, plan, product-evidence, run, and reconciliation protocol. Each stage writes a closed, digest-protected JSON document. A worker executes the exact identities assigned by the plan; it does not rediscover examples or decide its own shard boundaries.

## 1. Create the inventory

```bash
node scripts/runtime/verify-builtin-examples.mjs inventory \
  --output artifacts/builtin-examples/inventory.json
```

The `runmat.builtin-examples.inventory.v3` document records the full Git revision, clean or dirty source state, verifier digest, normalized example definitions and fixture requirements, documentation-only admissions, and one execution identity for every required lane. Catalog examples require stable IDs and executable verification contracts. Legacy sidecar examples receive content-derived fallback IDs; presentation-only legacy examples remain visible as `documentation-only` records.

The complete inventory is the release closure authority. One or more `--builtin` arguments select an exact, canonical builtin set for an atomic migration bundle. `--filter` and `--limit` remain exploratory selectors and cannot be combined with that set. Any selector creates a development-scoped inventory that cannot be reconciled as whole-product closure evidence. `--export` accepts an existing documentation-export JSON file, which is useful when the exporter was produced separately.

## 2. Freeze the execution plan

```bash
node scripts/runtime/verify-builtin-examples.mjs plan \
  --inventory artifacts/builtin-examples/inventory.json \
  --topology ci/builtin-example-topology.json \
  --output artifacts/builtin-examples/plan.json
```

A topology uses `runmat.builtin-examples.topology.v1`:

```json
{
  "schema": "runmat.builtin-examples.topology.v1",
  "products": {
    "native-cli": { "artifactProfile": "distributed-aot" },
    "browser-wasm": { "artifactProfile": "web" },
    "desktop-native": { "artifactProfile": "desktop-host" }
  },
  "lanes": {
    "native-host": { "shards": 4 },
    "browser-host": { "shards": 4 },
    "browser-graphics": { "shards": 2, "requiredCapabilities": ["headless-chrome"] },
    "browser-wgpu": { "shards": 2, "requiredCapabilities": ["headless-chrome", "webgpu"] },
    "desktop-host": { "shards": 2 }
  }
}
```

Every lane required by the inventory must appear. Without `--topology`, each required lane gets one shard with default limits. Unknown lanes and fields are rejected.

Release and package workflows can add `--product native-cli`, `--product browser-wasm`, or `--product desktop-native` to produce a digest-protected plan for one shipped product. Public Runtime CI uses `--product public-products` to cover the native CLI and browser/WASM products without claiming evidence for the private Desktop product. Product-scoped plans retain every applicable lane and execution identity, but their reconciliation reports remain explicitly product-scoped and cannot be admitted as full cross-product closure. `scripts/runtime/run-builtin-example-product.mjs` executes each shard in such a plan against one manifest; the full CI workflow uses the single-shard `run` command through its Actions matrix instead.

Assignments use `sha256-execution-identity-u64be-mod-v1`: interpret the first 64 bits of the execution identity as an unsigned big-endian integer and take the remainder modulo the lane's shard count. Adding or reordering examples therefore does not renumber unrelated examples. Each shard records its exact sorted identity list and an assignment digest.

CI derives its Actions matrix directly from the frozen plan:

```bash
node scripts/runtime/verify-builtin-examples.mjs matrix \
  --plan artifacts/builtin-examples/plan.json \
  --output artifacts/builtin-examples/execution-matrix.json
```

The closed matrix repeats the plan digest and the canonical `(lane, shardIndex, assignmentDigest, product)` tuple for every planned shard. Workers receive those tuples as data; changing, omitting, or duplicating one produces evidence that reconciliation rejects.

| Lane | Product | Current adapter |
| --- | --- | --- |
| `native-host` | `native-cli` | available |
| `native-filesystem` | `native-cli` | available |
| `native-loopback-network` | `native-cli` | available |
| `native-foreign-runtime` | `native-cli` | available |
| `interactive-host` | `native-cli` | available |
| `browser-host` | `browser-wasm` | available |
| `browser-graphics` | `browser-wasm` | available |
| `browser-wgpu` | `browser-wasm` | available |
| `desktop-host` | `desktop-native` | available in RunMat Desktop |

Unavailable product capabilities produce explicit `unavailable` results. They are never omitted or reported as passes. `Portable` examples require both `native-host` and `browser-host` execution.

## 3. Run isolated shards

Stage the actual products, record their file identities, and run the required product probes before starting shards. A `runmat.builtin-examples.artifacts.v2` manifest binds typed entrypoints to paths, byte lengths and SHA-256 digests. The execution plan fixes the product profile, required roles and probes.

### Stage the native CLI and pack

Build the native pair with the [two-phase helper](/docs/runtime/development/build-system#native-standalone-runtime), then stage it:

```bash
node scripts/runtime/stage-builtin-example-product.mjs \
  --binary target/release/runmat \
  --aot-runtime-pack target/release/runmat-aot-runtime.zip \
  --destination artifacts/native/product
```

Adjust the paths for the target, profile and executable name. On Windows, include the product's DLL directories with `--dependency-dir` and publisher-notice directories with `--dependency-notice-dir`. The staged tree must contain the complete native product.

The `distributed-aot` profile requires `runmat-binary` and `compile-runtime-pack`. Its identity covers the staged CLI, `runmat-aot-runtime.zip` and adjacent native dependencies. Release packaging excludes the pack from the normal CLI archive and publishes it as a separate asset.

### Record product identities

Keep the native product root and every browser file within the directory containing its manifest. Record the native pair:

```bash
node scripts/runtime/verify-builtin-examples.mjs artifacts \
  --product native-cli --profile distributed-aot \
  --source-revision "$(git rev-parse HEAD)" \
  --root artifacts/native/product \
  --entrypoint runmat-binary=runmat \
  --entrypoint compile-runtime-pack=runmat-aot-runtime.zip \
  --output artifacts/native/artifacts.json
```

The browser `web` profile binds the exact generated loader and Wasm module:

```bash
node scripts/runtime/verify-builtin-examples.mjs artifacts \
  --product browser-wasm --profile web \
  --source-revision "$(git rev-parse HEAD)" \
  --file wasm-js=artifacts/browser/runmat_wasm_web.js \
  --file wasm-binary=artifacts/browser/runmat_wasm_web_bg.wasm \
  --output artifacts/browser/artifacts.json
```

The `desktop-host` profile covers the complete Desktop tree and records both Runtime and Desktop revisions:

```bash
node scripts/runtime/verify-builtin-examples.mjs artifacts \
  --product desktop-native --profile desktop-host \
  --source-revision "$RUNMAT_REVISION" \
  --producer-revision "$DESKTOP_REVISION" \
  --root artifacts/desktop/product \
  --entrypoint runmat-desktop-binary=runmat-desktop \
  --output artifacts/desktop/artifacts.json
```

Existing `embedded-aot` records retain their meaning: the archive is inside the CLI, and `runmat-binary` is their only required entrypoint. New distributed-AOT plans require the separate-pack profile and its installation probe.

### Probe native compilation

Run the probe against the manifested pair:

```bash
node scripts/runtime/verify-builtin-examples.mjs probe \
  --kind native-distributed-aot-install-compile-and-execute-v1 \
  --artifact-manifest artifacts/native/artifacts.json \
  --output artifacts/native/distributed-aot.product-probe.json
```

The probe verifies the complete tree, resolves the two typed entrypoints, installs the pack with `aot install --offline --from`, compiles a fixed source vector offline and runs the standalone executable. It verifies the tree again after execution. Installation failure stops compilation.

The evidence binds the source revision, product profile, artifact-manifest digest, fixed-vector digest, process result and stdout digest. Use the [native pack acceptance check](/docs/runtime/development/build-system#verify-the-built-pair) for isolated cache, concurrent-installation and corrupt-pack checks against that same pair.

### Probe the Desktop host

Desktop uses the same product-evidence rules with a product-specific probe. The verifier starts the manifested Desktop binary in its private builtin-example host mode, sends one closed request through files named by environment variables, exercises a scripted host interaction, and validates the closed result. Each Desktop example runs in a fresh process and isolated workspace. The verifier mode is selected before telemetry, Tauri, windows, or other ordinary application services initialize; it installs deterministic input, dialog, filesystem, and figure boundaries instead.

```bash
node scripts/runtime/verify-builtin-examples.mjs probe \
  --kind desktop-hidden-host-protocol-v1 \
  --artifact-manifest artifacts/desktop/artifacts.json \
  --output artifacts/desktop/desktop-host.product-probe.json
```

```bash
node scripts/runtime/verify-builtin-examples.mjs run \
  --inventory artifacts/builtin-examples/inventory.json \
  --plan artifacts/builtin-examples/plan.json \
  --lane native-host \
  --shard-index 0 \
  --assignment-digest SHA256_FROM_PLAN \
  --artifact-manifest artifacts/native/artifacts.json \
  --work-dir artifacts/builtin-examples/work/native-host-0 \
  --result-out artifacts/builtin-examples/results/native-host-0.shard-result.json
```

Workers verify artifact bytes before execution and write inside their own work directory. The plan fixes per-case timeout, shard wall timeout, concurrency, process-output, structured-result, and figure-size limits. A `runmat.builtin-examples.shard-result.v1` manifest repeats the source, inventory, plan, runner, assignment, product, artifact, environment, adapter, and limit identities. Its result set must exactly equal the planned shard set.

Use a different work and result path for every lane and shard. Empty planned shards are valid and still emit a manifest. Do not reuse a browser artifact manifest for a native lane or mix product manifests between shards.

## 4. Reconcile all evidence

```bash
node scripts/runtime/verify-builtin-examples.mjs reconcile \
  --inventory artifacts/builtin-examples/inventory.json \
  --plan artifacts/builtin-examples/plan.json \
  --results-dir artifacts/builtin-examples/results \
  --artifact-manifest artifacts/native/artifacts.json \
  --artifact-manifest artifacts/browser/artifacts.json \
  --artifact-manifest artifacts/desktop/artifacts.json \
  --product-probe artifacts/native/distributed-aot.product-probe.json \
  --product-probe artifacts/desktop/desktop-host.product-probe.json \
  --closure \
  --report-dir artifacts/builtin-examples/report \
  --output artifacts/builtin-examples/reconciliation.json
```

Reconciliation rejects missing, duplicate, unexpected, wrongly assigned, stale, malformed, or definition-mismatched results. It also rejects shards that disagree about the bytes used for a product. Portable success results must agree across native and browser lanes; expected-error examples must agree on their stable error identifier.

Closure requires an all-product plan, a complete clean-source inventory, a locally verified artifact manifest matching every planned product, every product probe required by the plan, and no `unavailable`, `infra-error`, or `timed-out` execution unit. Probe evidence must name the same manifest digest used by the shards and must record a passing result. Reconciliation rejects missing, duplicate, stale, failed, or artifact-mismatched probes. Ordinary example failures are complete evidence and produce a closed report with status `failed`; infrastructure gaps reject closure instead of being confused with a test failure. Reconciliation writes JSON plus compact Markdown and HTML summaries.

`scripts/runtime/combine-builtin-example-reports.mjs` accepts the same `--inventory` and `--plan` reconciliation arguments for CI compatibility. Its earlier positional v2 report-combining interface remains available during migration.

## Legacy single-process interface

Existing local commands still use the compatibility runner:

```bash
node scripts/runtime/verify-builtin-examples.mjs --all
node scripts/runtime/verify-builtin-examples.mjs --check-inventory
```

The legacy `RUNMAT_EXAMPLE_SHARD_INDEX` and `RUNMAT_EXAMPLE_SHARD_COUNT` variables retain their previous contiguous-range behavior. Release and distributed verification should use inventory/plan/run/reconcile because it freezes identities, products, assignments, and closure requirements before execution.

## CI and release integration

Branch CI checks the current builtin-example inventory and public-product plan. It runs ordinary regressions and bounded acceptance against the native CLI and browser artifacts built by their workflows. Exhaustive example execution belongs to hardening runs; dev and main pushes do not invoke the example-closure workflow.

Run `.github/workflows/builtin-example-closure.yml` explicitly, or call it from a hardening workflow, to execute the full public-product plan. It freezes the inventory and plan, derives the shard matrix, and builds the native CLI with its separate AOT runtime pack and the browser module once. Its native product excludes OCCT; release products retain their full feature set. Product trees travel in tar containers to preserve executable modes and manifested bytes. Each shard uploads a result artifact named for the workflow attempt. Reconciliation rejects missing, duplicate, unexpected or stale evidence.

Full cross-product closure also requires Desktop evidence. A private workflow must stage the Desktop binary, record both repository revisions, run its protocol probe and every `desktop-host` shard, combine those results with the native and browser evidence from the same public revision, and reconcile with `--closure`. Public-product results alone cannot establish Desktop closure.

The native release workflow signs and notarizes macOS binaries before staging them. It copies Windows dependency DLLs before creating the product manifest. Bounded acceptance checks the manifested CLI and separate runtime pack, provisions the pack, compiles offline and executes the result. Packaging retains the manifests and acceptance receipts; the normal CLI archive excludes the separately distributed pack.

The npm workflow builds the publishable bindings once after choosing the package version. It records the generated browser entrypoints, runs SDK regressions and bounded Chrome-worker acceptance against those bytes, and retains the evidence. It packs and publishes with lifecycle scripts disabled so publication preserves the tested WebAssembly bytes. Exhaustive browser example execution remains part of the explicit closure run.
