# Builtin example verification shards

The standalone builtin example verifier can split one filtered or complete inventory across independent workers. Sharding changes only execution assignment and report filenames; every worker derives the same sorted inventory and digest from the same documentation export.

Set both zero-based shard variables for each worker:

```bash
RUNMAT_EXAMPLE_SHARD_INDEX=0 RUNMAT_EXAMPLE_SHARD_COUNT=4 \
  node scripts/runtime/verify-builtin-examples.mjs --all
```

Each worker writes HTML, Markdown, and JSON reports with a `.shard-NNNN-of-NNNN` suffix. `RUNMAT_EXAMPLE_REPORT_JSON` can select an explicit JSON destination, `RUNMAT_EXAMPLE_SOURCE` can pin the exact source identity, and `RUNMAT_EXAMPLE_ARTIFACT` can record the tested product identity. Filters are applied before sharding, so every worker in a combined run must receive the same filter and source inputs.

Combine all shard JSON reports only after every worker exits:

```bash
node scripts/runtime/combine-builtin-example-reports.mjs \
  --output scripts/example-output-reports/example-output-report.json \
  scripts/example-output-reports/example-output-report.shard-*.json
```

The combiner rejects missing or duplicate shard indices, stale source identities, different inventory digests or key sets, inconsistent ranges, duplicate results, and invalid summaries. It exits unsuccessfully when any validated example failed. Generated reports are verification evidence, not editable source authorities.
