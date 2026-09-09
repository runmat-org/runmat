# Builtin migration factory

`scripts/development/builtin-migration-factory.mjs` produces deterministic development evidence and review workspaces for the C00-C07 migration. It is not loaded by RunMat, is not an editable catalog, and its generated JSON and templates must not be checked in as authority.

The `inventory` command unions these repository surfaces:

- typed catalog source identities;
- runtime registrations, their implementation files, and typed lexical provenance (`literal-attribute` or `macro-invocation`);
- legacy documentation sidecars and runtime documentation shadows;
- runtime-local resolver declarations, catalog inference/link evidence, provider/fusion declarations, and generated WASM registry membership;
- tests and examples discoverable from those owned source files.

Source evidence cannot reliably distinguish a public canonical builtin from a callable alias or an internal binding. Catalog membership proves a canonical identity; all other intent stays explicitly unresolved until a reviewed disposition input says otherwise. The tool also preserves casing conflicts instead of silently selecting a spelling. Runtime totals report separate name-identity, binding-record, and provenance counts. The historical 1,412 figure remains labeled orientation evidence rather than a target the scanner manipulates its results to match.

```sh
node scripts/development/builtin-migration-factory.mjs inventory > /tmp/runmat-builtin-inventory.json
node scripts/development/builtin-migration-factory.mjs queue \
  --dispositions /tmp/reviewed-dispositions.json \
  --output /tmp/runmat-builtin-queue.json
node scripts/development/builtin-migration-factory.mjs prepare accumarray \
  --workspace /tmp/runmat-builtin-review
node scripts/development/builtin-migration-factory.mjs audit \
  --batch /tmp/array-batch.json \
  --dispositions /tmp/reviewed-dispositions.json
node --test scripts/development/builtin-migration/tests/*.test.mjs
```

## Reviewed disposition input

Generate a complete, deliberately unclassified seed outside the repository:

```sh
node scripts/development/builtin-migration-factory.mjs seed-dispositions \
  --output /tmp/reviewed-dispositions.json
```

Every seed row starts with `review.status: "unreviewed"` and null classification fields. An unreviewed row fails validation if classification is added. A reviewer changes the status, records nonempty evidence, and then supplies the disposition:

```json
{
  "schema_version": 1,
  "kind": "runmat-builtin-dispositions",
  "identities": {
    "oldfoo": {
      "review": { "status": "reviewed", "evidence": ["RM-1064 review link"] },
      "disposition": "alias",
      "canonical": "foo",
      "domain": "math",
      "family": "elementwise",
      "reason": null
    }
  }
}
```

Aliases require a canonical target. Internal bindings require a reviewed reason. Domain and family overrides are review inputs, not deductions.

## Queue interpretation

Every row includes ownership paths, registry/resolver dependencies, provider and host markers, documentation and test strength, conventional expected paths, maturity columns, migration state, and write-set collision keys. The queue sorts by a documented complexity score and then stable identity. Scores are scheduling estimates only: each nonzero factor contains the exact evidence that contributed its points.

The scanner intentionally uses bounded lexical recognition instead of compiling or importing production registries. Consequently, output diagnostics and unresolved fields are work items. They must not be converted into semantic conclusions without review. In particular, a missing generated-WASM match means only that the expected registration helper was not observed, not that the builtin is unsupported in browsers.

## Prepare and audit

`prepare` requires a workspace outside the repository and never edits RunMat source. It resolves canonical filesystem paths before writing and rejects both an output-root symlink into the repository and a pre-existing identity-workspace symlink that escapes the output root or enters the repository. It copies legacy JSON byte-for-byte, emits a field-by-field disposition checklist, records the selected inventory row, and creates comment-only catalog/runtime templates. Re-running it with unchanged source produces identical file content; it refuses to overwrite any review file whose content changed.

`audit` accepts repeated `--identity` selections or a batch document:

```json
{
  "schema_version": 1,
  "kind": "runmat-builtin-migration-batch",
  "identities": ["accumarray", "discretize"]
}
```

The audit is intentionally strict. Canonical identities require one catalog authority, runtime binding evidence, catalog documentation and typed examples, test evidence, native-link inputs, no old sidecar or runtime shadow, and no legacy resolver. Aliases must resolve to a canonical identity without copied documentation. Internal bindings must have runtime evidence and no public catalog/documentation surface. Missing, duplicated, contradictory, ambiguous, cyclic, or unresolved evidence fails the machine-readable report and exits nonzero.
