# Documentation artifact production

The source repository produces one artifact containing narrative, requirements,
builtin documentation, complete example projects, saved views, and their renderer.
Website builds consume that artifact; they do not run FEA or compile WASM.

From the source repository root, prepare the shared public TypeScript contracts
(`npm --prefix bindings/ts run build:types`), export the requirement registry, and
build the native CLI. Build the graphics-only renderer using the same WASM crate:

```sh
cargo build --release -j 1 -p runmat --bin runmat
CARGO_BUILD_JOBS=1 wasm-pack build crates/runmat-wasm --target web --out-dir ../../target/docs/graphics-runtime --out-name runmat_wasm_web --release -- --features graphics-replay
```

The `graphics-replay` feature keeps the existing renderer and scene codec but
excludes session/execution exports. `graphics.mjs` rejects a full-session bundle.

Run native compilation with one compiler job. The ordinary builder captures
selected examples sequentially and publishes one shared inventory atomically.
Snapshot production defaults to the optimized `target/release/runmat`. Set
`RUNMAT_DOCS_NATIVE_EXECUTABLE` to select a separately built executable. Capture
preserves each study's mesh, numerical controls and execution limits.
When independent studies need to overlap, separate capture from publication:

```sh
node scripts/docs/build-example-snapshots.mjs --example heatsink-steady --capture-only target/docs/heat-captures
# After other publishers finish, merge this completed capture with the latest inventory.
node scripts/docs/build-example-snapshots.mjs --publish-captures target/docs/heat-captures
```

Capture output must be a new directory. A failed capture leaves the published
inventory untouched; publication rechecks every retained and captured source
against the canonical examples. Serialize publication transactions and bound
concurrent studies to the host's available memory. Native builds, meshing and
solving compete for resources; overlapping capture is an explicit build choice.
Capture retains a private executable copy so every worker uses the same native
build. That copy is removed before the example inventory is published and is
never included in a downloadable project.

Before publication, check each completed capture with the graphics-only module:

```sh
node scripts/docs/verify-example-replay.mjs --example heatsink-steady --captures target/docs/heat-captures
```

This verifies saved mesh presence, field colors, camera and legend metadata, and
cold scene import/release without initializing the execution runtime or solving.
It is an artifact/replay check, not browser GPU validation or a substitute for
the native numerical acceptance and Desktop saved-result replay checks. Inspect
the native posters separately for framing and visual interpretation.

Then run the source artifact producer with the exported requirements and builtin
metadata. Builtin reference pages come from the builtin documentation producer,
which merges catalog-owned documentation with the remaining legacy sidecars in
`docs/builtins/reference`; the export writes one `builtins/reference/<name>.json`
per documented builtin. Optional `--evidence-manifest` supplies actual assessment
records; without it, linked requirements are not described as passed.

Builtin references are packed from the complete canonical documentation export,
not the remaining `docs/builtins/reference` files. Each exported record is written
unchanged as JSON under `builtins/reference/<module_stem>.json`, preserving the
published filename's case, examples, assertions, links, evidence, and catalog
fields. The aggregate wrapper is not substituted for the existing per-builtin
artifact layout. Missing, duplicate, preview, or unsafe-filename products fail
before the staged artifact replaces the published output.

Optionally pass `--builtin-documentation PATH` to either docs producer to check
a staged aggregate export against the complete strict canonical export from
this source checkout. The shared adapter also accepts
`RUNMAT_BUILTIN_DOCUMENTATION_EXPORT` (file) or
`RUNMAT_BUILTIN_DOCUMENTATION_BINARY` (prebuilt exporter), exclusively. Otherwise
it runs the canonical exporter through Cargo in the source checkout. Supplied
products must match that source and include the full catalog, not a scoped
identity export. Public docs production always requires strict mode: no legacy
documents or missing catalog documentation are accepted. Transition exports
remain available to migration tooling but cannot be packed as public docs.
Duplicate ownership remains an exporter error, and preview products never
qualify for packing. Removing
legacy sidecars does not remove canonical builtin references from the artifact.

```sh
cargo run --locked -p runmat-builtins --bin export_builtin_documentation -- \
  --transition --output target/docs/builtin-documentation.json
node scripts/docs/build.mjs \
  --native-executable target/release/runmat \
  --graphics-runtime target/docs/graphics-runtime \
  --requirements target/docs/requirements.json \
  --builtin-descriptors docs/builtins/meta.json \
  --builtin-documentation target/docs/builtin-documentation.json \
  --out target/docs/source
```

Snapshot capture runs only catalog examples marked `snapshot: true`, through the
admitted native execution route with unchanged study settings. It preserves the
original publication and fields, captures three portable scenes and native
posters, and writes Desktop's existing content-addressed dataset/run-history
format. Production fails if execution, capture, input binding, or integrity fails.

The website consumes the directory using its existing shared preparation path:

```sh
RUNMAT_DOCS_SOURCE_ARTIFACT=/absolute/path/to/target/docs/source node scripts/prepare-docs.mjs
```

Prepared asset paths bind the original inputs and full saved-result inventory.
Desktop example links pin the project manifest and every file; imports install a
fresh project and do not execute source. A newly built Desktop app registers the
`runmat://example` handler. Website preview builds alone do not update an already
installed Desktop app's protocol registration.

## FEA docs checks

The FEA docs under `docs/simulation` describe the complete system. A documented feature that the
current release does not include is flagged where the docs describe it, with
`[not yet in the current release](pending:<id>)` in prose or a ```` ```fea pending=<id> ```` fence,
and has an entry in `docs/simulation/pending-features.json` (schema
`docs/schema/pending-features.schema.json`). `runmat check` reports the same id, and the docs
artifact exports the registry as `fea/pending-features.json` for the website badge.

```sh
node scripts/docs/docs-lint.mjs              # registry ids, landed flags, front matter, links
node scripts/docs/generate-pages.mjs         # validation/<family>.md, validation/status.md, the procedure table
node scripts/docs/generate-pages.mjs --check # CI: generated pages are current
# Status also counts automated tests per family from the source (test-counts.mjs); --check ignores
# those counts and the docs export refreshes them at publish time.
node scripts/docs/check-snippets.mjs --runmat target/release/runmat   # `runmat check` every ```fea block
```

When a feature lands, set its `landed_in`; the lint then lists every flag to delete. Snippets fenced
```` ```fea pending=<id> ```` must fail `runmat check` until the feature lands, and the snippet
checker names the flag once they pass. ```` ```fea fragment ```` marks a syntax illustration that is
not a whole file. Page front matter follows `docs/schema/doc-front-matter.schema.json`.
