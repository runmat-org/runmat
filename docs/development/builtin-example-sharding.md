# Builtin example verification

Builtin documentation examples are verified through an inventory, plan, product-evidence, run, and reconciliation protocol. Each stage writes a closed, digest-protected JSON document. A worker executes the exact identities assigned by the plan; it does not rediscover examples or decide its own shard boundaries.

## 1. Create the inventory

```bash
node scripts/runtime/verify-builtin-examples.mjs inventory \
  --output artifacts/builtin-examples/inventory.json
```

The `runmat.builtin-examples.inventory.v2` document records the full Git revision, clean or dirty source state, verifier digest, normalized example definitions and fixture requirements, documentation-only admissions, and one execution identity for every required lane. Catalog examples require stable IDs and executable verification contracts. Legacy sidecar examples receive content-derived fallback IDs; presentation-only legacy examples remain visible as `documentation-only` records.

The complete inventory is the release closure authority. `--builtin`, `--filter`, and `--limit` create a development-scoped inventory that cannot be reconciled as closure evidence. `--export` accepts an existing documentation-export JSON file, which is useful when the exporter was produced separately.

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
    "native-cli": { "artifactProfile": "embedded-aot" },
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
| `native-loopback-network` | `native-cli` | unavailable |
| `native-foreign-runtime` | `native-cli` | unavailable |
| `interactive-host` | `native-cli` | unavailable |
| `browser-host` | `browser-wasm` | available |
| `browser-graphics` | `browser-wasm` | available |
| `browser-wgpu` | `browser-wasm` | available |
| `desktop-host` | `desktop-native` | available in RunMat Desktop |

Unavailable adapters are planned and produce explicit `unavailable` results. They are never omitted or reported as passes. `Portable` examples require both `native-host` and `browser-host` execution.

## 3. Run isolated shards

Each tested product has a `runmat.builtin-examples.artifacts.v2` manifest beside its staged product root. The manifest identifies typed entrypoints and binds files to their relative paths, byte lengths, and SHA-256 digests. The plan freezes the artifact profile, entrypoint roles, and required product probes before workers run. The `embedded-aot` native profile requires `runmat-binary`; its manifest recursively closes the complete staged product tree, including adjacent runtime libraries installed beside the executable. The native compilation runtime is embedded in the CLI bytes by `scripts/build-runmat-with-aot-runtime.sh` or its PowerShell counterpart. RunMat does not currently ship or load a sidecar AOT runtime, so the verifier does not expose a synthetic sidecar profile. The browser `web` profile remains an exact-file contract containing the generated `wasm-js` loader and `wasm-binary` module. The `desktop-host` profile requires `runmat-desktop-binary`, recursively closes the staged Desktop product, and binds both the public Runtime source revision and the private Desktop producer revision.

Create manifests from the real staged products. The native root and every browser `--file` path must remain within the directory that contains the manifest.

```bash
node scripts/runtime/verify-builtin-examples.mjs artifacts \
  --product native-cli --profile embedded-aot \
  --source-revision "$(git rev-parse HEAD)" \
  --root artifacts/native/product \
  --entrypoint runmat-binary=runmat \
  --output artifacts/native/artifacts.json

node scripts/runtime/verify-builtin-examples.mjs artifacts \
  --product browser-wasm --profile web \
  --source-revision "$(git rev-parse HEAD)" \
  --file wasm-js=artifacts/browser/runmat_wasm_web.js \
  --file wasm-binary=artifacts/browser/runmat_wasm_web_bg.wasm \
  --output artifacts/browser/artifacts.json

node scripts/runtime/verify-builtin-examples.mjs artifacts \
  --product desktop-native --profile desktop-host \
  --source-revision "$RUNMAT_REVISION" \
  --producer-revision "$DESKTOP_REVISION" \
  --root artifacts/desktop/product \
  --entrypoint runmat-desktop-binary=runmat-desktop \
  --output artifacts/desktop/artifacts.json
```

Before native shards run, probe the staged CLI itself. The probe resolves the typed `runmat-binary` entrypoint from the manifest, verifies the complete tree, compiles a fixed source vector with `runmat compile`, executes the resulting standalone program, and verifies the tree again. Its closed evidence document binds the source revision, product profile, artifact-manifest digest, fixed vector digest, process result, and stdout digest.

```bash
node scripts/runtime/verify-builtin-examples.mjs probe \
  --kind native-embedded-aot-compile-and-execute-v1 \
  --artifact-manifest artifacts/native/artifacts.json \
  --output artifacts/native/embedded-aot.product-probe.json
```

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
  --product-probe artifacts/native/embedded-aot.product-probe.json \
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

`.github/workflows/builtin-example-closure.yml` is the reusable public-product gate. Separate jobs freeze the inventory and public-product plan, derive a closed execution matrix from its lane, shard index, and assignment digest tuples, build the embedded-AOT native CLI and browser module once, and transfer the staged product trees in tar containers so executable modes and manifested bytes survive Actions artifact transport. Each matrix job downloads those products, executes exactly one planned shard in its own work directory, and uploads one uniquely named result artifact. Reconciliation downloads the protocol, products, probes, and complete result-artifact set independently; exact plan reconciliation rejects omitted, duplicated, unexpected, or stale shard evidence. Artifact names include the workflow run attempt so a rerun cannot collide with evidence from an earlier attempt. Protected `dev` and `main` pushes call this workflow from branch CI; it can also be dispatched directly.

A private Desktop integration workflow must own full cross-product closure. Before closure can be claimed, that workflow must stage the Desktop binary, record both repository revisions, run the Desktop protocol probe and every `desktop-host` shard, combine those results with native CLI and browser/WASM product evidence from the same public revision, and invoke reconciliation with `--closure`. This split keeps private product construction out of the public repository while preserving one authoritative all-product closure rule.

The integration product intentionally excludes OCCT. It keeps the runtime features required by the builtin example surface without rebuilding the heavy CAD dependency for this independent gate. The release workflow continues to build its full feature set. On release, signing and notarization finish before the platform product is staged; the staged bytes are then manifested, compile-and-execute probed, exercised by every native product lane, reconciled, and archived. Windows dependency DLLs are copied into the staged tree before its recursive manifest is created. Each platform artifact retains its manifest, probe, shard results, and reconciliation report alongside the archive.

The npm workflow builds the publishable bindings once after choosing the package version. It creates an exact-file browser manifest, runs and reconciles every browser product lane against those bytes, retains the evidence, packs the already-built package with lifecycle scripts disabled, and publishes that exact tarball with lifecycle scripts still disabled. This prevents `prepublishOnly` from replacing the tested WebAssembly bytes during publication.
