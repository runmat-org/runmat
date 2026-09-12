# Builtin migration factory

The builtin migration factory coordinates the reviewed C00-C07 migration without becoming runtime or catalog authority. Generated queues, workspaces, audits, verification results, and seals are content-addressed development evidence. They must remain outside production source.

## Authority model

The factory consumes two complementary inventories:

- `runmat-compiled-builtin-migration-inventory` v3 is the semantic and registration authority. The runtime exporter reports canonical catalog entries, typed alias edges, static constant declarations, legacy functions and documentation, runtime constant and callable bindings, declaration provenance, GPU specifications, fusion specifications, each registration's canonical builtin path, typed migration-finding subjects, path-derived identity membership for grouped provider owners, build configuration, and its own snapshot digest. Exact provider owners must resolve to implementation provenance in the same Rust module scope as the recorded registration path. A provider declaration may live in a descendant module, such as an implementation owner's `specification` module; unrelated modules and textual-prefix collisions do not share authority. A missing or mismatched declaration path is a structural validation error, not reviewable migration work.
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

The factory rejects a missing snapshot, a future schema version, unknown fields at any nesting level, malformed enum or record payloads, noncanonical ordering, inconsistent binding/provenance relationships, structurally invalid producer validation, or a snapshot whose SHA-256 digest does not match its contents. Compiler-declared callable spellings participate in the public spelling inventory. Constant spellings do so only when the identity has no callable form, which lets a constant-only identity enter reviewed control without allowing case variants such as `Inf` and `inf` to override the callable's public spelling. Migration-readiness findings are different from structural errors: the factory preserves every typed finding in inventory evidence and requires the control manifest to give it an exact reviewed disposition. A legacy GPU or fusion registry group remains a reviewed raw key rather than being guessed into a callable identity. Inventory v3 pins the source revision, the complete ordered source-root and file snapshot, a separate digest over every scanner-consumed path, the compiled snapshot digest, the migration-finding digest, and the reviewed identity-disposition digest. Its dirty-state observation is scoped to those frozen roots, so an unrelated untracked file cannot invalidate the baseline while any tracked, untracked, modified, or deleted source inside the migration surface remains visible.

## Reviewed topology, control, and scheduling

Migration topology and execution policy are separate reviewed authorities. The frozen topology owns atomic bundle membership, cohorts, target domain and family, typed `canonical`, `alias`, or `internal` disposition, bundle composition, and the topology review evidence. The v4 `runmat-builtin-migration-control-manifest` names that topology by digest and contains only the execution policy needed to carry it out:

- baseline source, disposition, migration-finding, and compiled-target context not already owned by topology;
- C00-C07 with fixed order and semantic labels;
- bundle prerequisites, additional authored scopes, references to globally owned integration products, gate plans, owner role, and reviewed complexity;
- typed primary, alias, or internal public identity; exact callable and constant spelling sets; separate callable and constant implementation authorities with their exact bindings; shared dependencies; reviewed complexity; maturity applicability; distinct expected catalog-entry, alias, constant, documentation, native-link, and WASM authorities; and owner for every identity;
- exact reviewed dispositions for every compiled migration-readiness finding;
- reviewed exceptions, migration execution targets, terminal R31 qualification lanes, and storage policy.

The serialized control cannot repeat bundle membership, cohort, atomic reason, disposition, domain, family, composition, or identity-target evidence. Its parser reconstructs the operational view by joining the exact control keys to a deterministically reconstructed frozen topology. The join retains every topology-owned authored scope and adds only separately reviewed, non-overlapping execution scopes. Missing or extra keys, topology drift, overlapping scopes, or attempts to restate topology-owned fields are rejected.

Successful validation returns immutable topology and control views that are recognized only by the modules that performed the complete deterministic checks. Queue, lease, preparation, audit, gate, inventory-delta, disposition, and seal entry points reject copied or lookalike objects even when their fields resemble a reviewed manifest. Invocation-specific state, such as the bundle selected for an inventory-delta proof, is passed separately and never attached to the reviewed control object.

Identifiers beginning with `__` are supported for real internal bindings. Distinct identity keys or public spellings that collide case-insensitively are rejected. Primary identities own their spelling. Alias rows own one typed edge and declaration package but no copied contract, documentation, or executable binding; when the canonical identity is in another bundle, the alias bundle has an explicit semantic prerequisite on it. Executable internal rows retain a typed `HiddenInternal` catalog entry and their exact implementation binding without entering public completion or documentation. Callable and constant forms each have an independent implementation authority, so mixed-form identities may have distinct owners without exceptions. Every present form requires exactly one owner and exact typed bindings; an absent form records a closed reason. Catalog entry and constant counts remain separate expected authorities. Bundle prerequisites must exist and form a DAG. Bundle/identity membership comes only from topology and remains reciprocal. A bundle is an atomic scheduling and write-set boundary, not a package-taxonomy authority: identities in one bundle may have different reviewed target domains and families when they share a legacy owner that must move atomically. They must still belong to one cohort, and the queue derives their exact target-family set from the materialized topology view. Authored scopes cannot overlap integration outputs, and cross-bundle authored collisions are reported by the queue. Multiple bundles may require the same integration-owned generated product without acquiring authorship or a scheduling collision; every declaration for a shared product ID and path must be identical. Repository-relative paths must be canonical POSIX paths; alternate spellings such as embedded `.` segments cannot bypass scope comparisons.

Shared integration outputs have one authority in the global control review. Each registry entry binds a product ID, exact path, producer profile, generator path, and the frozen source digests for both the checked-in product and its generator. Bundle reviews contain only product IDs; composition resolves those references and rejects unknown products, duplicate paths, uncovered registry entries, or attempts to place a shared output in a worker-owned write set. Expected removals remain bundle-owned because a removal is part of one atomic migration. Their evidence is lossless and typed: every record identifies the evidence kind, exact source path, frozen snapshot state, content digest when present, and complete affected-identity set. A missing baseline path is represented explicitly rather than being collapsed into an empty digest or inferred later from current source.

Rust parent composition uses an explicit 99-product registry: catalog and runtime roots, the catalog alias root, paired catalog/runtime domain parents, and paired nested parents for every family shared by more than one frozen bundle. The registry is fixed source data rather than a filesystem census. Global review and composed-control validation require the reviewed composition products and baseline projection to cover that registry exactly, including every product ID, path, crate role, module path, and aggregation role. Catalog aggregation follows the owning domain: the root, `constants`, `array`, and `array/creation` products aggregate constants, while other catalog products aggregate callable entries only. Most products have the `bundle-referenced` lifecycle and must be referenced by at least one exact bundle transition. An existing parent with no frozen bundle transition may instead be reviewed as `reviewed-baseline-only`; that lifecycle is limited to Rust module-composition products, cannot be referenced by a bundle, and remains part of deterministic checked-in product verification. The baseline reserves absent products with empty child sets, but materialization does not create an unused parent until a reviewed transition makes it reachable.

Effective authored tree scopes derive exact file exclusions from the complete global product registry, including baseline-only products. The derivation happens after topology and additional authored scopes are reviewed; review inputs cannot supply exclusions themselves. The same path-scope implementation governs collision detection, lease admission, authored/integration diffs, authority validation, and composition transitions. A bundle may therefore edit sibling leaves inside its tree while every integration-owned parent file remains outside its authority.

Generating the initial review surface does not confer authority. `draft-control` emits a deterministic, content-addressed v1 scaffold for every exact inventory identity and migration finding. It pins the baseline and the compiled-authority, lexical-observation, and complete inventory-row digests, but leaves typed public identity, forms, per-form implementation, topology disposition, cohort, bundle, domain, family, dependencies, complexity, maturity, expected authorities, removals, baseline evidence, owner, gate plans, exceptions, and storage policy unresolved. Its bundle list is empty, every review status is `unreviewed`, and the parser rejects attempts to insert inferred facts or review claims into the draft.

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

The topology is reviewed and frozen first. It transitively binds the draft, inventory, component graph, three cohort reviews, reconciliation, and stability corrections. `scaffold-control` then emits the deterministic post-topology review surface. Scaffold schema v4 binds every bundle and identity row to both the inventory and topology and leaves every execution-policy decision explicitly unresolved. Its source observations preserve distinct typed records for catalog and runtime ownership, compiled provenance, catalog documentation, legacy sidecars, runtime documentation shadows, documentation/example sources, tests, runtime registration, native-link inputs, legacy and catalog resolvers, providers, fusion, and generated registry paths. Each record carries the exact frozen source digest or an explicit absent-snapshot state. A separately self-digested, non-authoritative proposal section distinguishes primary public spellings, aliases, and internal identities from callable, constant, and implementation ownership facts. Callable and constant forms have separate typed implementation authorities, so an identity such as `inf` can retain its callable owner and its distinct constant-registration owner without an exception or duplicated ownership fact. Canonical callable identities receive target-path proposals from the reviewed topology; internal callables and constant forms derive proposals only from exact compiled provenance. Migration-finding routing uses the compiled typed identity or owner membership and never infers affected identities from a display key.

Control review is split into one global review and one review file per topology bundle. Bundle reviews own the exact bundle policy and identity controls; their gate plans refer to reusable program-profile IDs. The global review owns those exact executable profiles, the baseline projection for integration-owned Rust parent modules, migration-finding dispositions, exceptions, target policy, storage policy, and global review evidence. A bundle that changes a shared parent owns one closed typed composition transition and references each affected parent product exactly. The parent itself remains outside every authored lease. The baseline compiled target records the platform used to produce the frozen inventory; it does not limit later execution to that platform. Target policy keeps the platforms that actually run C00–C07 bundle gates separate from R31 terminal qualification. R31 uses a closed product-domain taxonomy for public native/CLI/AOT, browser/WASM, Desktop, Server/remote/cluster, packaging/release, foreign interop, parallel/distributed execution, providers/GPU, deterministic execution, security/recovery, performance/resources, and workspace quality. This taxonomy fixes the semantic obligations without prescribing the later reviewed commands or product manifests, and the parser rejects a missing, substituted, or renamed lane. Each lane explicitly classifies Linux AArch64, Linux x86_64, macOS AArch64, macOS x86_64, and Windows x86_64 as required or reviewed not-applicable; every lane retains at least one required target, required cells form the immutable terminal matrix, and Windows executes last. Reviewed storage profiles cover exactly the migration-time set, not deferred R31 machines. A content-addressed review-set manifest binds the exact bytes of all review files, requires one review for every bundle, rejects symlink escapes, and requires the defined executable profiles to equal the referenced set.

Browser examples and browser-runtime support remain separate reviewed maturity decisions. When either applies, one product-backed browser gate builds the affected browser/WASM product, initializes it in a real browser, executes the catalog examples for the bundle's exact identity set, and binds the resulting artifact, product probe, execution plan, shard results, and reconciliation. Sharing this proof avoids a second, weaker smoke-test path while still requiring reviewers to classify example coverage and browser-runtime applicability independently. A wasm32 type check alone cannot satisfy either obligation. Migration builds omit OCCT unless the bundle affects geometry/CAD; the terminal feature matrix covers the final OCCT configuration.

`compose-control` validates the complete review set and expands program profiles into a deterministic, unreviewed candidate. An independent control attestation binds that candidate and every transitive input digest. `freeze-control` reconstructs the topology, scaffold, review set, and candidate; validates the attestation; and emits the schema-v5 control manifest, including the exact global composition baseline and per-bundle transitions. No command can promote a reviewed-looking JSON object directly. Every control-consuming command replays the same chain and requires byte-for-byte equality with the final control.

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

Queue v3 is derived at bundle granularity. It exposes prerequisite and earlier-cohort barriers, migration-finding work items, reviewed complexity, applicable maturity gates, authored scopes, integration outputs, and source observations. A finding assigned to the bundle is work the bundle must resolve and is verified by the inventory-delta gate; it does not block the bundle from starting. Bundles within one cohort may be authored together, but every bundle in an earlier cohort must be sealed before a later cohort becomes schedulable. Queue-state schema v3 may record `leased`, `submitted`, `integrated`, or `verified` workflow progress. Each state names its predecessor state and reviewed checkpoint and may append exactly one seal; it cannot remove or replace an accepted seal or regress recorded workflow state. A bundle becomes sealed only through a canonical content-addressed reference to a passing seal artifact for the exact bundle, control manifest, baseline inventory, lease, and identity set. The queue command resolves those references relative to the state file and rejects missing artifacts, path escapes, digest mismatches, stale or forged authority fields, and duplicate seal claims. Recorded progress cannot override blockers or create seal authority.

A reviewed queue-checkpoint v1 binds the queue-state digest, complete accepted-seal set, current integrated source revision and digest, and predecessor checkpoint. The hash chain proves that a supplied state is an append-only successor of its supplied predecessor; it does not prove that a local file is the newest state. Orchestration must publish the current checkpoint digest to a durable progress ledger or control-wave record and pass that exact trusted digest when issuing a lease. Replaying an older queue prefix, even with the repository rewound to its former `HEAD`, then fails against the published checkpoint digest.

## Storage admission

The control manifest defines reviewed host profiles. Each profile is selected by exact operating system, architecture, and execution-host identity and gives storage two named roles:

- `source-worktree`, for source and worktree safety;
- `target-temp`, for disjoint build targets and temporary products.

Each role has an absolute evidence path, a POSIX device or Windows volume identity, minimum-free and pause-below watermarks, and a maximum observation age. Selectors must be unique, and every admitted machine must match exactly one profile. This lets macOS, Linux, Windows, and multiple hosts on one platform use their real paths and volume identities without weakening one another's policy. Within every profile, source and target roles must identify different filesystems; build targets remain disjoint; and OCCT remains disabled unless the affected surface requires it. Every machine gate records the selected profile and execution-host identity, timestamp, paths, filesystem identities, available bytes, reviewed thresholds, and derived `admitted` or `paused` status. A product gate cannot pass with a stale observation, the wrong build target or host profile, a different filesystem, or either role below its pause watermark.

## Leases and preparation

An authored lease v4 binds one bundle to the control digest, owner, reviewed integration-base revision, the exact compiled and lexical inventory at that revision, the trusted queue-checkpoint digest, the complete accepted seal set, the barrier subset that makes the bundle schedulable, its authored write set, and its forbidden integration outputs. Audit derives the authored and integration changed-path sets from the exact repository commits. Authored changes outside the reviewed scope and integration changes outside the reviewed generated products fail.

Lease assignment is also a two-stage workflow. A closed v4 request names the reviewed control digest, bundle, lease ID, owner, exact current integration-base revision, content bindings for the inventory at that revision, the trusted queue-checkpoint digest, the complete canonical accepted-seal set, its required barrier subset, both set digests, UTC interval, and review evidence. `issue-lease` requires that revision to be the canonical repository's clean `HEAD` and a descendant of the frozen control baseline. It verifies the inventory against the repository bytes, requires the checkpoint to match the externally published current digest, and requires both seal sets to match validated queue state. With no accepted seals, the lease base must equal the frozen control baseline and its exact inventory. Otherwise it must descend from every accepted seal and equal at least one accepted seal's integrated revision and source digest. Issuance also revalidates final authority and migration-finding state for every accepted bundle at the lease base. These checks detect an unsealed intermediate commit, later revert, corruption, or replayed queue prefix that an ancestry check alone would miss. The resulting lease embeds the reviewed request and derives the authored scope and integration-output exclusions from control.

The lease interval is active from `issued_at` inclusively until `expires_at` exclusively. Issuance and every live prepare, gate, audit, phase-capture, and seal boundary authorize that interval against one clock observation. Structural parsing remains time-independent so historical lease, verification, queue, and seal evidence can still be reproduced after a lease expires. Sealing therefore requires the exact authored lease artifact in addition to its digest recorded throughout the evidence chain.

Authored patches may proceed concurrently within a cohort, but finalization is serialized. After each passing seal is admitted, integration writes and reviews the next queue state and checkpoint and publishes that checkpoint digest. Any peer patch authored from the prior base must be replayed or rebased onto the new sealed checkpoint, then receive a refreshed lease before audit. A seal produced under the stale peer lease cannot be appended because its accepted-seal set omits the newly admitted seal.

For example, the reviewed request's `base_revision` is the exact `git:<40 lowercase hex characters>` value reported by `git rev-parse HEAD` at assignment time. Later authored and integration commits may advance the lane; audit retains that base as the start of its explicit two-phase history.

```bash
node scripts/development/builtin-migration-factory.mjs issue-lease \
  --request /tmp/array-shape-lease-request.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  --lease-base-inventory /tmp/runmat-lease-base-inventory.json \
  --state /tmp/runmat-queue-state.json \
  --queue-checkpoint /tmp/runmat-queue-checkpoint.json \
  --trusted-queue-checkpoint-digest sha256:... \
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
  --lease-base-inventory /tmp/runmat-lease-base-inventory.json \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --lease /tmp/array-lease.json \
  --workspace /tmp/runmat-builtin-review
```

The source-field disposition schema is v2. It inventories every JSON leaf by JSON Pointer and value digest. Closure requires every prepared leaf to be marked `preserved`, `normalized`, `corrected`, or `removed`. Preserved fields name a typed catalog destination. Changes and removals require a reason and evidence. Missing, duplicate, added, or mutated baseline leaves fail reconciliation.

## Audit and machine gates

Audit v7 accepts a complete bundle, not a convenient subset. An audit evidence manifest names the reviewed authored commit and points to prepare results, completed field dispositions, and machine gate results. The CLI derives the lease-base-to-authored and authored-to-integrated changed-path sets from Git; callers cannot assert either set themselves.

The audit treats bundle work and integration as separate commit phases. If `B` is the reviewed lease base, `A` is the clean authored revision, and `I` is the final integrated subject, Git must prove `B` is an ancestor of `A` and `A` is an ancestor of `I`. Every path changed by `B..A` must be inside the reviewed authored lease and must not be an integration output. Every path changed by `A..I` must be one of the bundle's exact reviewed integration-output paths. A bundle with no integration outputs therefore permits no `A..I` changes. The integration phase may leave a deterministic output unchanged; product gates still verify the complete reviewed product contract at `I`.

The audit result records all three revisions, both exact changed-path sets, the reviewed authored scopes and integration outputs, and digests of those policy sets. Verification v6 binds that phase record, the exact authored lease digest, and the barrier-seal set. Seal requests and results use schema v5, bind the same evidence, and reconstruct both ranges from the trusted repository before accepting verification or integration-gate evidence. All semantic checks and machine gates describe `I`, never the intermediate authored commit.

Machine gate results use a closed v7 schema and a code-owned producer identity. Each reviewed bundle owns closed gate plans whose typed program, fixed arguments, approved direct-tool digests for every execution target, and expected artifact roles are part of the control digest. Repository programs identify Node and any direct auxiliary tool, such as Git. Rust programs select one of the closed Cargo operations `check`, `clippy`, `fmt`, or `test`, or a named workspace binary; they identify Cargo, rustc, and the exact subcommand tools used by that operation. All gate programs must provide toolchains for the complete reviewed execution-target set. A producer request can select only the reviewed lease, bundle, and gate. It cannot replace the Cargo manifest, target directory, configuration, or any of the three bound inventory snapshots through extra arguments.

The adapter resolves the reviewed tools for the subject inventory's execution target, verifies their bytes plus the script or manifest bytes against the reviewed plan and frozen baseline inventory, and admits storage before launching the process. It removes inherited Rust compiler, wrapper, formatter, documentation, and flag overrides, selects the reviewed rustc, rustdoc, and rustfmt paths explicitly when applicable, and places the reviewed Cargo tool directory first for Cargo subcommands. The evidence records the full direct-tool set and this enforced selection. This contract covers tools selected directly by the gate; it does not claim that arbitrary linkers, native build scripts, or generated test executables are themselves reviewed producer identities.

The checked repository must resolve to the reviewed source-worktree filesystem. Cargo target, process temporary directories, and every declared artifact output must resolve to the reviewed target-temp filesystem. The adapter supplies those exact Cargo and temporary paths through a closed environment record and captures the path/filesystem bindings in producer evidence. Cargo targets are stable per control and bundle for cache reuse; temporary directories are isolated per evidence artifact. The minimum-free watermark is a hard rejection floor, the higher pause watermark stops new work conservatively, and only an admitted observation can launch a gate.

After admission, the adapter executes from the bound repository, captures the exact exit status, signal, stdout and stderr digests, validates its parser-specific machine output, derives checks and aggregate status, and writes required evidence artifacts outside the repository without overwriting existing files. Each gate result binds the exact execution target used to select its executable and storage profile. Producer-time validation requires that target to equal the subject build; audit and sealing require it to remain within the reviewed control target set. Later validation therefore never substitutes the frozen inventory's platform for the platform that produced the result. Each artifact reference records its reviewed role, canonical path, byte length, and content digest; parsing the gate result verifies the referenced bytes and their target-volume binding. A passing gate must emit exactly its reviewed artifact roles, while a failed producer may emit none and cannot introduce an unreviewed role. The adapter records the frozen revision that reviewed the producer entrypoint or Cargo manifest, the direct tool bytes, and the clean committed subject revision actually tested. The subject source digest binds the complete repository state used by the producer; the direct-tool contract does not separately classify every imported module or compiled dependency as a producer executable. Requests cannot supply an executable, arguments, working directory, environment, identities, checks, result, raw output digest, timestamp, execution target, or storage facts. Catalog-contract and runtime-binding adapters consume the recursively validated compiled inventory exporter; the architecture and other reviewed exit-status plans derive identity checks only from their frozen bundle scope.

Control baseline, lease base, and subject provenance are separate. The frozen C00 inventory remains the authority for reviewed producer bytes, topology, and original source evidence. The lease-base inventory describes the clean integrated state `B` from which the active bundle starts. Inventory delta compares `B` with the final subject `I`, so changes from previously sealed bundles are part of the admitted base rather than new out-of-scope drift. At issuance, those sealed identities are revalidated against `B`; at `I`, final-authority checks apply to the active bundle while the exact `B..I` delta proves sealed work was not modified. Prepare, gate, audit, verification, and seal evidence carry distinct control-baseline, lease-base, and subject inventory digests as applicable, plus the lease ID, exact authored lease digest, and subject source provenance.

The documentation-cutover adapter executes the reviewed catalog documentation exporter and reconciles that output with the completed source-field dispositions. It uses the content-pinned Git tool declared by that gate to read each legacy JSON source from the frozen revision, verifies those bytes against the baseline inventory, and derives the complete leaf set. Its closed v2 evidence artifact records every source path, source digest, escaped JSON Pointer, original value digest, reviewed disposition, and any reason or review evidence. Retained leaves also record a typed catalog destination, its reviewed expected digest, and the digest observed at that exact pointer in the canonical catalog export. Removed leaves require an explicit reviewed reason and have no destination. The artifact binds the source revision and digest, compiled inventory, control manifest, bundle, source dispositions, and complete catalog export. Missing or duplicate leaves, nearby matching values, stale exports, changed source values, and destination mismatches cannot pass.

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

The bundle's reviewed `documentation-cutover` gate plan selects the repository-owned `scripts/development/builtin-migration/documentation-export-cli.mjs` producer with an empty argument list. The producer runs the aggregate `runmat-builtins` transition exporter through the reviewed Cargo and rustc toolchain. Its complete direct-tool contract is Node, Cargo, rustc, and Git: Node owns the producer entrypoint, Cargo and rustc build and run the exporter, and the adapter uses the content-pinned Git executable to read legacy documentation from the frozen revision. A cargo-binary plan cannot truthfully represent that full contract and is rejected. Run the gate with `--inputs`:

```bash
node scripts/development/builtin-migration-factory.mjs produce-gate \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  --lease-base-inventory /tmp/runmat-lease-base-inventory.json \
  --lease /tmp/array-lease.json \
  --state /tmp/rm1064-queue-state.json \
  --queue-checkpoint /tmp/rm1064-queue-checkpoint.json \
  --trusted-queue-checkpoint-digest sha256:... \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --bundle array-shape --gate documentation-cutover \
  --artifact array-shape-documentation \
  --inputs /tmp/array-shape-documentation-input.json \
  --output /tmp/array-shape-documentation-gate.json
```

The global integration-product registry is the sole authority for product paths and exact generator paths and baseline digests. The adapter supplies the bundle's reviewed registry entries through a closed standard-input envelope. For generated Rust parents, it derives effective composition from the reviewed baseline, the seals accepted by the lease-bound queue checkpoint, and the active bundle's reviewed transition when that bundle changes composition. A bundle with no composition output retains the baseline plus accepted transitions without inventing a no-op transition. Callers cannot supply the derived projection in a gate inputs file. Composition projection v3 records declaration conditions separately from an ordered set of independently conditioned re-exports. Its closed condition vocabulary covers tests, Cargo features, the WebAssembly target, and their reviewed conjunctions; named re-exports use typed items and aliases, and documentation hiding is explicit. Catalog aggregation order is recorded independently from canonical child declaration order. Each contribution has one closed source kind: a typed slice, an entry-group slice, or delegation to the same typed function in a generated child parent. Runtime and support children cannot aggregate catalog data, grouped sources are limited to entries, and no field accepts raw Rust or raw `cfg` text. These rules reproduce current visibility, WebAssembly, feature, and catalog-order behavior while allowing domain and root parents to compose generated family parents recursively without copying identity lists between levels. Cross-bundle transitions use disjoint parent/child keys, so canonical seal ordering cannot overwrite another bundle's contribution. The deterministic-product producer requires an exact reviewed selection: one `--product` argument for each reviewed output, or `--no-products` when a bundle has no integration output. It rejects an omitted, unknown, duplicate, or noncanonical selection. For each selected product it independently revalidates the exact generator and its frozen source bytes, runs the generator twice into disposable outputs, compares both content digests, and compares the result with the checked-in product. Its parser requires exact coverage of the bundle's reviewed product IDs and paths. The inventory-delta producer rebuilds the compiler-backed and lexical inventory from the current worktree, admits changes only within the bundle's authored and integration scopes, requires compiled authority outside the bundle to remain unchanged, reconciles every migration finding disposition, and checks each identity against its reviewed final authority and removals. Catalog-contract, runtime-binding, deterministic-product, and inventory-delta requests each supply an `artifact_output` path through their inputs file, just as documentation cutover supplies its dedicated evidence path. Exit-status gates have no artifact output unless their reviewed parser is replaced by a typed product parser.

The native and browser example gates use the reviewed `example-gate-cli.mjs` producer with an empty argument list. Their `--inputs` document contains `artifact_output`, a canonical `evidence_root` on the admitted target-temp storage volume, and an inline, schema-checked `example_manifest`; callers cannot add arguments or select a manifest path. After storage admission, the adapter canonicalizes that manifest, writes it with create-new semantics to the gate artifact's fixed temporary directory, and passes a closed path/digest/length envelope over standard input. The producer accepts no command-line arguments, reopens the canonical regular file, and verifies its bytes against the envelope before it evaluates any referenced evidence. The resulting `example-reconciliation` proof retains the staged manifest's exact path, byte length, and digest.

```json
{
  "artifact_output": "/tmp/array-shape-example-reconciliation.json",
  "evidence_root": "/tmp/array-shape-example-evidence",
  "example_manifest": {
    "schema_version": 1,
    "kind": "runmat-builtin-example-gate-manifest",
    "source_revision": "git:...",
    "identities": ["reshape", "squeeze"],
    "inventory": "/tmp/example-inventory.json",
    "plan": "/tmp/example-plan.json",
    "reconciliation": "/tmp/example-reconciliation.json",
    "shard_results": ["/tmp/native-shard.json", "/tmp/browser-shard.json"],
    "artifact_manifests": ["/tmp/native-artifact.json", "/tmp/browser-artifact.json"],
    "product_probes": []
  }
}
```

The manifest names the exact bundle identity set and the inventory, plan, reconciliation, shard, product-manifest, and optional product-probe evidence produced by the standalone verifier. Inventory v3 represents the bundle as a canonical multi-builtin selector rather than a text filter or result limit. The gate independently validates every closed schema, verifies product bytes against their manifests, recomputes reconciliation from the exact shard set, requires an all-product plan over every lane declared by those examples, and checks that each planned product digest is present in its shard evidence. Required example maturity needs at least one catalog-owned executable example; a reviewed not-applicable identity may have no example. The resulting artifact retains per-identity example and execution identities plus digests of every input evidence file. Every referenced evidence path must be a canonical regular file beneath `evidence_root` on its admitted filesystem; symlinks, path escapes, and files on an unreviewed mounted filesystem fail before the producer runs.

Nearby files, matching tokens, successful exit codes, file-presence counts, or authored delta JSON cannot substitute for these machine contracts. Expected removals are complete files tied to their frozen baseline bytes. In-file authority transitions are proven by compiled catalog/binding provenance and the final lexical ownership inventory; the factory does not maintain a second parser for Rust items or match arms.

Run a reviewed producer with:

```bash
node scripts/development/builtin-migration-factory.mjs produce-gate \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  --lease-base-inventory /tmp/runmat-lease-base-inventory.json \
  --lease /tmp/array-lease.json \
  --state /tmp/rm1064-queue-state.json \
  --queue-checkpoint /tmp/rm1064-queue-checkpoint.json \
  --trusted-queue-checkpoint-digest sha256:... \
  "${topology_args[@]}" \
  "${control_args[@]}" \
  --bundle array-shape --gate architecture --artifact array-shape-architecture \
  --output /tmp/array-shape-architecture.json
```

Each result pins its own artifact ID, subject source revision and digest, distinct control-baseline, lease-base, and subject inventory digests, lease ID and exact lease digest, control ID, bundle, complete identity set, checks, result, referenced product artifacts, and storage admission. Gate identities cover catalog contracts, runtime bindings, documentation cutover, native linking, WASM registration, architecture boundaries, focused tests, strict Clippy, formatting/diff checks, native examples, product-backed browser execution and examples, provider, host, foreign, and parallel tests, deterministic generated products, and inventory delta. Foreign, parallel, host, native-example, browser-example, browser-runtime, provider, and WASM-registry applicability are independent reviewed maturity decisions. Architecture, focused tests, strict Clippy, and formatting remain mandatory audit gates; every bundle must also define deterministic-product and inventory-delta plans, including an explicit no-products selection when appropriate.

Integration products also declare a closed semantic verification contract. The generated WASM registry contract binds its stamped canonical registration digest and per-kind counts to the subject's native compiled registration manifest; byte-for-byte determinism alone cannot satisfy that contract.

Audit maps each reviewed maturity requirement to structural gate evidence. A passing token search, test filename, example string, or file-presence count is not closure. Canonical identities additionally require exact catalog cardinality and removal of legacy sidecars, runtime documentation shadows, and legacy resolvers. Aliases and internal identities use disposition-specific rules. Expected file removals require matching baseline path/digest evidence in control and actual absence. In-file registration and ownership transitions are admitted only when the compiler-backed inventory delta and current lexical inventory both agree with the reviewed destination.

For every preserved, normalized, or corrected legacy documentation leaf, the documentation producer reports the exact source path and JSON Pointer, typed catalog destination identity and pointer, reviewed expected value digest, and observed destination value digest. This destination reconciliation prevents a completed checklist from passing when the destination field is absent or contains unrelated content.

## Verification and sealing

Verification manifest v6 is a reviewed, closed request containing content-addressed references to one passing audit v7 and the required gate artifacts. Every reference includes path, artifact ID, and digest. Verification rejects missing, duplicate, mutated, unavailable, failed, stale, or wrong-bundle evidence and emits verification result v6.

Seal manifest v5 combines an exact passing verification with integration-owned evidence. A seal requires at least `deterministic-products` and `inventory-delta`, complete bundle identity coverage, matching control-baseline, lease-base, subject, and exact lease provenance, the complete accepted seal set, and the exact canonical barrier subset. A passing seal is still integration evidence; it does not mutate catalog or runtime authority.

```bash
node scripts/development/builtin-migration-factory.mjs audit \
  --compiled-inventory /tmp/runmat-compiled-inventory.json \
  --control /tmp/rm1064-control.json \
  --baseline-inventory /tmp/runmat-builtin-inventory.json \
  --lease-base-inventory /tmp/runmat-lease-base-inventory.json \
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
