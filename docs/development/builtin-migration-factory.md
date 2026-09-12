# Builtin migration factory

The builtin migration factory coordinates the reviewed C00-C07 migration without becoming runtime or catalog authority. Generated queues, workspaces, audits, verification results, and seals are content-addressed development evidence. They must remain outside production source.

## Authority model

The factory consumes two complementary inventories:

- `runmat-compiled-builtin-migration-inventory` v1 is the semantic and registration authority. The runtime exporter reports the compiled catalog, legacy functions and documentation, constants, runtime bindings, declaration provenance, GPU specifications, fusion specifications, build configuration, and its own snapshot digest.
- The JavaScript source scanner records paths, documentation files, tests, examples, and bounded lexical observations. Every such observation is labeled `discovery_only`; it cannot establish runtime registration or semantic completion.

Export the compiled snapshot for the exact build configuration being migrated, then pass it to every inventory-derived command:

```sh
cargo run -p runmat-runtime --bin export_builtin_migration_inventory -- \
  --output /tmp/runmat-compiled-inventory.json
node scripts/development/builtin-migration-factory.mjs inventory \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --output /tmp/runmat-builtin-inventory.json
```

The exporter writes to stdout when no argument is supplied, which is the interface used by reviewed gate plans. `--output` creates a new artifact and refuses to replace an existing path. `--help` prints usage without constructing the inventory; every other argument is rejected.

The factory rejects a missing snapshot, a future schema version, unknown fields at any nesting level, malformed enum or record payloads, noncanonical ordering, inconsistent binding/provenance relationships, structurally invalid producer validation, or a snapshot whose SHA-256 digest does not match its contents. Compiler-declared callable spellings participate in the public spelling inventory. Constant spellings do so only when the identity has no callable form, which lets a constant-only identity enter reviewed control without allowing case variants such as `Inf` and `inf` to override the callable's public spelling. Migration-readiness findings are different from structural errors: the factory preserves every typed finding in inventory evidence and requires the control manifest to give it an exact reviewed disposition. A legacy GPU or fusion registry group remains a reviewed raw key rather than being guessed into a callable identity. Inventory v2 pins the source revision, the complete ordered source-root and file snapshot, a separate digest over every scanner-consumed path, the compiled snapshot digest, the migration-finding digest, and the reviewed identity-disposition digest. Its dirty-state observation is scoped to those frozen roots, so an unrelated untracked file cannot invalidate the baseline while any tracked, untracked, modified, or deleted source inside the migration surface remains visible.

## Reviewed topology, control, and scheduling

Migration topology and execution policy are separate reviewed authorities. The frozen topology owns atomic bundle membership, cohorts, target domain and family, typed `canonical`, `alias`, or `internal` disposition, bundle composition, and the topology review evidence. The v2 `runmat-builtin-migration-control-manifest` names that topology by digest and contains only the execution policy needed to carry it out:

- baseline source, disposition, migration-finding, and compiled-target context not already owned by topology;
- C00-C07 with fixed order and semantic labels;
- bundle prerequisites, additional authored scopes, integration-produced files, gate plans, owner role, and reviewed complexity;
- exact public spelling, runtime owner when one exists, shared dependencies, reviewed complexity, maturity applicability, exact callable and constant authorities, expected removals, baseline evidence, and owner for every identity;
- exact reviewed dispositions for every compiled migration-readiness finding;
- reviewed exceptions, execution targets, and storage policy.

The serialized control cannot repeat bundle membership, cohort, atomic reason, disposition, domain, family, composition, or identity-target evidence. Its parser reconstructs the operational view by joining the exact control keys to a deterministically reconstructed frozen topology. The join retains every topology-owned authored scope and adds only separately reviewed, non-overlapping execution scopes. Missing or extra keys, topology drift, overlapping scopes, or attempts to restate topology-owned fields are rejected.

Successful validation returns immutable topology and control views that are recognized only by the modules that performed the complete deterministic checks. Queue, lease, preparation, audit, gate, inventory-delta, disposition, and seal entry points reject copied or lookalike objects even when their fields resemble a reviewed manifest. Invocation-specific state, such as the bundle selected for an inventory-delta proof, is passed separately and never attached to the reviewed control object.

Identifiers beginning with `__` are supported for real internal bindings. Distinct identity keys or public spellings that collide case-insensitively are rejected. Canonical callable identities require a runtime owner. A canonical constant-only identity may use `null` when its reviewed authority names at least one exact runtime constant and no callable runtime binding; aliases and internal identities may also explicitly use `null`. Catalog entry counts, catalog constant counts, runtime binding records, and case-sensitive runtime constant spellings are independent reviewed facts. Bundle prerequisites must exist and form a DAG. Bundle/identity membership comes only from topology and remains reciprocal. A bundle is an atomic scheduling and write-set boundary, not a package-taxonomy authority: identities in one bundle may have different reviewed target domains and families when they share a legacy owner that must move atomically. They must still belong to one cohort, and the queue derives their exact target-family set from the materialized topology view. Authored scopes cannot overlap integration outputs, and cross-bundle authored collisions are reported by the queue. Multiple bundles may require the same integration-owned generated product without acquiring authorship or a scheduling collision; every declaration for a shared product ID and path must be identical. Repository-relative paths must be canonical POSIX paths; alternate spellings such as embedded `.` segments cannot bypass scope comparisons.

Generating the initial review surface does not confer authority. `draft-control` emits a deterministic, content-addressed v1 scaffold for every exact inventory identity and migration finding. It pins the baseline and the compiled-authority, lexical-observation, and complete inventory-row digests, but leaves public spelling, disposition, alias target, cohort, bundle, domain, family, runtime owner, dependencies, complexity, maturity, expected authorities, removals, baseline evidence, owner, gate plans, exceptions, and storage policy unresolved. Its bundle list is empty, every review status is `unreviewed`, and the parser rejects attempts to insert inferred facts or review claims into the draft.

The identity-disposition review has a compact authoring form for large inventories. A v1 `runmat-builtin-disposition-review` groups exact, explicitly enumerated identities that share one reviewed canonical, alias, or internal decision. It accepts no wildcard, pattern, path-derived selector, or default disposition. Groups must cover the baseline inventory exactly once, and alias maps must cover their group exactly and target a reviewed canonical identity. Domain, family, bundle, and cohort are separate target-layout decisions made in the reviewed topology; the disposition review does not infer them from current paths. Existing catalog entries are baseline evidence rather than a target-state veto: review may identify one as a duplicated alias or an accidentally exposed internal binding, provided its obsolete authority is removed and verified during the atomic cutover. The `compile-dispositions` command expands that reviewer-authored input into the complete row-per-identity disposition file consumed by inventory generation. Both artifacts remain development review evidence; neither is a runtime or catalog authority.

The queue preserves unresolved and conflicting path-derived domain or family observations in its evidence, but it does not treat those fields as blockers after the reviewed topology supplies the target layout. The reviewed disposition is handled the same way. Any unresolved inventory field outside that closed topology-owned set still blocks scheduling. This distinction lets the inventory report the repository as it exists without allowing historical paths to override the reviewed migration topology.

```sh
node scripts/development/builtin-migration-factory.mjs compile-dispositions \
  --review /tmp/rm1064-disposition-review.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  --output /tmp/reviewed-dispositions.json
```

```sh
node scripts/development/builtin-migration-factory.mjs draft-control \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  --output /tmp/rm1064-control-draft.json
```

`component-graph` derives indivisible authority components from the same inventory. After reviewers complete the three exact cohort inputs, reconciliation, and stability corrections, `compose-topology` produces the deterministic candidate. `freeze-topology` accepts only an attestation bound to that candidate and every transitive input. `validate-topology` repeats the composition and requires byte-for-byte equality with the frozen result. The command's `--help` output lists the complete argument set for each stage.

The topology is reviewed and frozen first. It transitively binds the draft, inventory, component graph, three cohort reviews, reconciliation, and stability corrections. `scaffold-control` then emits the deterministic post-topology review surface. Scaffold schema v2 binds every bundle and identity row to both the inventory and topology and leaves every execution-policy decision explicitly unresolved. Its source observations preserve distinct typed records for catalog and runtime ownership, compiled provenance, catalog documentation, legacy sidecars, runtime documentation shadows, documentation/example sources, tests, runtime registration, native-link inputs, legacy and catalog resolvers, providers, fusion, and generated registry paths. Each record carries the exact frozen source digest or an explicit absent-snapshot state. Canonical identities also receive a non-authoritative runtime-binding proposal by combining the reviewed topology target path with the compiler-observed function and variant; aliases, internals, and constant-only identities do not receive invented callable authority.

Control review is split into one global review and one review file per topology bundle. Bundle reviews own the exact bundle policy and identity controls; their gate plans refer to reusable program-profile IDs. The global review owns those exact executable profiles, migration-finding dispositions, exceptions, execution targets, storage policy, and global review evidence. The baseline compiled target records the platform used to produce the frozen inventory; it does not limit later execution to that platform. The reviewed execution-target set admits subject inventories independently, and reviewed storage profiles must cover exactly that set. A content-addressed review-set manifest binds the exact bytes of all review files, requires one review for every bundle, rejects symlink escapes, and requires the defined executable profiles to equal the referenced set.

`compose-control` validates the complete review set and expands program profiles into a deterministic, unreviewed candidate. An independent control attestation binds that candidate and every transitive input digest. `freeze-control` reconstructs the topology, scaffold, review set, and candidate; validates the attestation; and emits the resulting control manifest. No command can promote a reviewed-looking JSON object directly. Every control-consuming command replays the same chain and requires byte-for-byte equality with the final control.

The repeated topology arguments are the complete evidence chain used by every control-consuming command:

```bash
topology_args=(
  --topology /tmp/rm1064-reviewed-topology.json
  --candidate /tmp/rm1064-topology-candidate.json
  --attestation /tmp/rm1064-topology-attestation.json
  --component-graph /tmp/rm1064-component-graph.json
  --draft /tmp/rm1064-control-draft.json
  --c01-c03-review /tmp/rm1064-c01-c03-review.json
  --c04-c05-review /tmp/rm1064-c04-c05-review.json
  --c06-c07-review /tmp/rm1064-c06-c07-review.json
  --reconciliation /tmp/rm1064-topology-reconciliation.json
  --stability-corrections /tmp/rm1064-topology-stability-corrections.json
)
```

```bash
node scripts/development/builtin-migration-factory.mjs scaffold-control \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  --output /tmp/rm1064-control-scaffold.json

node scripts/development/builtin-migration-factory.mjs init-control-reviews \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  --control-scaffold /tmp/rm1064-control-scaffold.json \
  --review-directory /tmp/rm1064-control-review-drafts
```

Reviewers complete `global.json` and every file under `bundles/` in that directory. The index command validates those reviewer-authored payloads, computes their content digests, and writes a new sealed review set without modifying the authoring directory:

```bash
node scripts/development/builtin-migration-factory.mjs index-control-reviews \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  --control-scaffold /tmp/rm1064-control-scaffold.json \
  --review-directory /tmp/rm1064-control-review-drafts \
  --review-set-directory /tmp/rm1064-control-reviews

node scripts/development/builtin-migration-factory.mjs compose-control \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  --control-scaffold /tmp/rm1064-control-scaffold.json \
  --control-review-set /tmp/rm1064-control-reviews/review-set.json \
  --output /tmp/rm1064-control-candidate.json
```

The independent reviewer starts from a deterministic attestation template, changes its review status and evidence, and seals that reviewed input. The sealing command validates the exact candidate and computes the envelope digest; it cannot supply the review decision.

```bash
node scripts/development/builtin-migration-factory.mjs scaffold-control-attestation \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  --control-scaffold /tmp/rm1064-control-scaffold.json \
  --control-review-set /tmp/rm1064-control-reviews/review-set.json \
  --control-candidate /tmp/rm1064-control-candidate.json \
  --output /tmp/rm1064-control-attestation-review.json

node scripts/development/builtin-migration-factory.mjs seal-control-attestation \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  --control-scaffold /tmp/rm1064-control-scaffold.json \
  --control-review-set /tmp/rm1064-control-reviews/review-set.json \
  --control-candidate /tmp/rm1064-control-candidate.json \
  --attestation-review /tmp/rm1064-control-attestation-review.json \
  --output /tmp/rm1064-control-attestation.json
```

The same four control-review inputs accompany freezing and every control-consuming command:

```bash
control_args=(
  --control-scaffold /tmp/rm1064-control-scaffold.json
  --control-review-set /tmp/rm1064-control-reviews/review-set.json
  --control-candidate /tmp/rm1064-control-candidate.json
  --control-attestation /tmp/rm1064-control-attestation.json
)

node scripts/development/builtin-migration-factory.mjs freeze-control \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --output /tmp/rm1064-control.json
```

```bash
node scripts/development/builtin-migration-factory.mjs validate-control \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}"
node scripts/development/builtin-migration-factory.mjs queue \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --dispositions /tmp/reviewed-dispositions.json \
  --output /tmp/runmat-builtin-queue.json
```

Queue v2 is derived at bundle granularity. It exposes prerequisite and migration-finding blockers, reviewed complexity, applicable maturity gates, authored scopes, integration outputs, and source observations. Queue state may record workflow progress, but cannot override blockers or create authority.

## Storage admission

The control manifest defines reviewed host profiles. Each profile is selected by exact operating system, architecture, and execution-host identity and gives storage two named roles:

- `source-worktree`, for source and worktree safety;
- `target-temp`, for disjoint build targets and temporary products.

Each role has an absolute evidence path, a POSIX device or Windows volume identity, minimum-free and pause-below watermarks, and a maximum observation age. Selectors must be unique, and every admitted machine must match exactly one profile. This lets macOS, Linux, Windows, and multiple hosts on one platform use their real paths and volume identities without weakening one another's policy. Within every profile, source and target roles must identify different filesystems; build targets remain disjoint; and OCCT remains disabled unless the affected surface requires it. Every machine gate records the selected profile and execution-host identity, timestamp, paths, filesystem identities, available bytes, reviewed thresholds, and derived `admitted` or `paused` status. A product gate cannot pass with a stale observation, the wrong build target or host profile, a different filesystem, or either role below its pause watermark.

## Leases and preparation

An authored lease v1 binds one bundle to the control digest, owner, base revision, exact authored write set, and forbidden integration outputs. Audit derives changed paths from Git at the lease base. Changes outside the reviewed scope or direct edits to generated integration products fail.

Lease assignment is also a two-stage workflow. A closed v1 request names only the reviewed control digest, bundle, lease ID, owner, UTC interval, and review evidence. `issue-lease` rejects unreviewed or stale requests and derives the base revision, authored scope, and integration-output exclusions from the frozen control; a request cannot supply or widen those fields. The resulting lease embeds the reviewed request and carries its own content digest.

```bash
node scripts/development/builtin-migration-factory.mjs issue-lease \
  --request /tmp/array-shape-lease-request.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --output /tmp/array-shape-lease.json
```

`prepare` requires the compiled inventory, control, and lease. Its workspace must be outside the repository. Canonical path checks reject symlink escapes into source. Prepare v2 copies legacy documentation byte-for-byte, writes comment-only templates, records inventory evidence, and emits a source-field checklist. It never edits source and refuses to overwrite a modified review file.

```bash
node scripts/development/builtin-migration-factory.mjs prepare accumarray \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --lease /tmp/array-lease.json \
  --workspace /tmp/runmat-builtin-review
```

The source-field disposition schema is v2. It inventories every JSON leaf by JSON Pointer and value digest. Closure requires every prepared leaf to be marked `preserved`, `normalized`, `corrected`, or `removed`. Preserved fields name a typed catalog destination. Changes and removals require a reason and evidence. Missing, duplicate, added, or mutated baseline leaves fail reconciliation.

## Audit and machine gates

Audit v4 accepts a complete bundle, not a convenient subset. An audit evidence manifest points to prepare results, completed field dispositions, and machine gate results. The CLI derives the changed-path set; callers cannot assert it themselves.

Machine gate results use a closed v5 schema and a code-owned producer identity. Each reviewed bundle owns closed gate plans whose typed program, fixed arguments, approved direct-tool digests for every execution target, and expected artifact roles are part of the control digest. Repository programs identify Node and any direct auxiliary tool, such as Git. Rust programs select one of the closed Cargo operations `check`, `clippy`, `fmt`, or `test`, or a named workspace binary; they identify Cargo, rustc, and the exact subcommand tools used by that operation. All gate programs must provide toolchains for the complete reviewed execution-target set. A producer request can select only the reviewed bundle and gate. It cannot replace the Cargo manifest, target directory, or configuration through extra arguments.

The adapter resolves the reviewed tools for the subject inventory's execution target, verifies their bytes plus the script or manifest bytes against the reviewed plan and frozen baseline inventory, and admits storage before launching the process. It removes inherited Rust compiler, wrapper, formatter, documentation, and flag overrides, selects the reviewed rustc, rustdoc, and rustfmt paths explicitly when applicable, and places the reviewed Cargo tool directory first for Cargo subcommands. The evidence records the full direct-tool set and this enforced selection. This contract covers tools selected directly by the gate; it does not claim that arbitrary linkers, native build scripts, or generated test executables are themselves reviewed producer identities.

The checked repository must resolve to the reviewed source-worktree filesystem. Cargo target, process temporary directories, and every declared artifact output must resolve to the reviewed target-temp filesystem. The adapter supplies those exact Cargo and temporary paths through a closed environment record and captures the path/filesystem bindings in producer evidence. Cargo targets are stable per control and bundle for cache reuse; temporary directories are isolated per evidence artifact. The minimum-free watermark is a hard rejection floor, the higher pause watermark stops new work conservatively, and only an admitted observation can launch a gate.

After admission, the adapter executes from the bound repository, captures the exact exit status, signal, stdout and stderr digests, validates its parser-specific machine output, derives checks and aggregate status, and writes required evidence artifacts outside the repository without overwriting existing files. Each gate result binds the exact execution target used to select its executable and storage profile. Producer-time validation requires that target to equal the subject build; audit and sealing require it to remain within the reviewed control target set. Later validation therefore never substitutes the frozen inventory's platform for the platform that produced the result. Each artifact reference records its reviewed role, canonical path, byte length, and content digest; parsing the gate result verifies the referenced bytes and their target-volume binding. A passing gate must emit exactly its reviewed artifact roles, while a failed producer may emit none and cannot introduce an unreviewed role. The adapter records the frozen revision that reviewed the producer entrypoint or Cargo manifest, the direct tool bytes, and the clean committed subject revision actually tested. The subject source digest binds the complete repository state used by the producer; the direct-tool contract does not separately classify every imported module or compiled dependency as a producer executable. Requests cannot supply an executable, arguments, working directory, environment, identities, checks, result, raw output digest, timestamp, execution target, or storage facts. Catalog-contract and runtime-binding adapters consume the recursively validated compiled inventory exporter; the architecture and other reviewed exit-status plans derive identity checks only from their frozen bundle scope.

Baseline and subject provenance are deliberately separate. `prepare` remains tied to the frozen baseline so its source-field checklist still describes the material being migrated. `produce-gate` requires a fresh compiled inventory and derives a subject inventory using the dispositions frozen in control. Gate, audit, verification, and seal evidence carry both the baseline inventory digest and the subject inventory digest, plus the subject source revision and digest. Product gates refuse a dirty subject, and audit cannot pass unless the subject is a clean commit. This allows a baseline prepare artifact to remain valid after migration without labeling the migrated code as if it were still the baseline.

The documentation-cutover adapter executes the reviewed catalog documentation exporter and reconciles that output with the completed source-field dispositions. It uses the content-pinned Git tool declared by that gate to read each legacy JSON source from the frozen revision, verifies those bytes against the baseline inventory, and derives the complete leaf set. Its closed v1 evidence artifact records every source path, source digest, escaped JSON Pointer, original value digest, reviewed disposition, and any reason or review evidence. Retained leaves also record a typed catalog destination, its reviewed expected digest, and the digest observed at that exact pointer in the canonical catalog export. Removed leaves require an explicit reviewed reason and have no destination. The artifact binds the source revision and digest, compiled inventory, control manifest, bundle, source dispositions, and complete catalog export. Missing or duplicate leaves, nearby matching values, stale exports, changed source values, and destination mismatches cannot pass.

Documentation evidence must be written outside the repository to a new path. The input file contains only reviewed source dispositions and that output path:

```json
{
  "source_dispositions": [
    {
      "baseline_digest": "sha256:...",
      "value": { "schema_version": 2, "kind": "runmat-builtin-source-field-disposition" }
    }
  ],
  "artifact_output": "/tmp/array-shape-documentation-evidence.json"
}
```

The bundle's reviewed `documentation-cutover` gate plan must select the `documentation_cutover` parser and the `runmat-builtins` catalog documentation exporter with its transition argument. Run it with `--inputs`:

```bash
node scripts/development/builtin-migration-factory.mjs produce-gate \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --bundle array-shape --gate documentation-cutover \
  --artifact array-shape-documentation \
  --inputs /tmp/array-shape-documentation-input.json \
  --output /tmp/array-shape-documentation-gate.json
```

The deterministic-product producer requires an explicit reviewed selection: `--product wasm-registry` for the current shared registry, or `--no-products` when a bundle has no integration output. It rejects an omitted, unknown, duplicate, or noncanonical selection. For each selected product it runs the integration generator twice into disposable outputs, compares both content digests, and compares the result with the checked-in product. Its parser requires exact coverage of the bundle's reviewed product IDs and paths and verifies the generator bytes against the frozen source inventory. The inventory-delta producer rebuilds the compiler-backed and lexical inventory from the current worktree, admits changes only within the bundle's authored and integration scopes, requires compiled authority outside the bundle to remain unchanged, reconciles every migration finding disposition, and checks each identity against its reviewed final authority and removals. Catalog-contract, runtime-binding, deterministic-product, and inventory-delta requests each supply an `artifact_output` path through their inputs file, just as documentation cutover supplies its dedicated evidence path. Exit-status gates have no artifact output unless their reviewed parser is replaced by a typed product parser.

The example gate is driven by a reviewed `example-gate-cli.mjs --manifest PATH` plan. Its manifest names the exact bundle identity set and the inventory, plan, reconciliation, shard, product-manifest, and optional product-probe evidence produced by the standalone verifier. Inventory v3 represents the bundle as a canonical multi-builtin selector rather than a text filter or result limit. The gate independently validates every closed schema, verifies product bytes against their manifests, recomputes reconciliation from the exact shard set, requires an all-product plan over every lane declared by those examples, and checks that each planned product digest is present in its shard evidence. Required example maturity needs at least one catalog-owned executable example; a reviewed not-applicable identity may have no example. The resulting `example-reconciliation` artifact retains per-identity example and execution identities plus digests of every input evidence file.

Nearby files, matching tokens, successful exit codes, file-presence counts, or authored delta JSON cannot substitute for these machine contracts. Expected removals are complete files tied to their frozen baseline bytes. In-file authority transitions are proven by compiled catalog/binding provenance and the final lexical ownership inventory; the factory does not maintain a second parser for Rust items or match arms.

Run a reviewed producer with:

```bash
node scripts/development/builtin-migration-factory.mjs produce-gate \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --bundle array-shape --gate architecture --artifact array-shape-architecture \
  --output /tmp/array-shape-architecture.json
```

Each result pins its own artifact ID, subject source revision and digest, baseline and subject inventory digests, control digest, bundle, complete identity set, checks, result, referenced product artifacts, and storage admission. Gate identities cover catalog contracts, runtime bindings, documentation cutover, native linking, WASM registration, architecture boundaries, focused tests, strict Clippy, formatting/diff checks, native/browser examples, provider/host/foreign tests, deterministic generated products, and inventory delta.

Audit maps each reviewed maturity requirement to structural gate evidence. A passing token search, test filename, example string, or file-presence count is not closure. Canonical identities additionally require exact catalog cardinality and removal of legacy sidecars, runtime documentation shadows, and legacy resolvers. Aliases and internal identities use disposition-specific rules. Expected file removals require matching baseline path/digest evidence in control and actual absence. In-file registration and ownership transitions are admitted only when the compiler-backed inventory delta and current lexical inventory both agree with the reviewed destination.

For every preserved, normalized, or corrected legacy documentation leaf, the documentation producer reports the exact source path and JSON Pointer, typed catalog destination identity and pointer, reviewed expected value digest, and observed destination value digest. This destination reconciliation prevents a completed checklist from passing when the destination field is absent or contains unrelated content.

## Verification and sealing

Verification manifest v3 is a reviewed, closed request containing content-addressed references to one passing audit v4 and the required gate artifacts. Every reference includes path, artifact ID, and digest. Verification rejects missing, duplicate, mutated, unavailable, failed, stale, or wrong-bundle evidence and emits verification result v3.

Seal manifest v2 combines an exact passing verification with integration-owned evidence. A seal requires at least `deterministic-products` and `inventory-delta`, complete bundle identity coverage, matching baseline and subject provenance, and reviewed prerequisite seal references. A passing seal is still integration evidence; it does not mutate catalog or runtime authority.

```bash
node scripts/development/builtin-migration-factory.mjs audit \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --lease /tmp/array-lease.json \
  --batch /tmp/array-batch.json --evidence /tmp/array-audit-evidence.json \
  --output /tmp/array-audit.json
node scripts/development/builtin-migration-factory.mjs verify \
  --manifest /tmp/array-verification.json --output /tmp/array-verification-result.json
node scripts/development/builtin-migration-factory.mjs seal \
  --manifest /tmp/array-seal.json --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --output /tmp/array-seal-result.json
```

All schemas reject unknown fields and unsupported versions. Existing schema versions retain their prior meaning; producers and consumers must advance together rather than interpreting new fields opportunistically.

Run the focused factory suite with:

```sh
env TMPDIR=/private/tmp node --test \
  scripts/development/builtin-migration/tests/*.test.mjs
```
