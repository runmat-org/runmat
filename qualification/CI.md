# CI and release qualification

Every dev candidate runs the ordinary Rust regressions on Linux, macOS and
Windows, including the fast FEA tests. The four package groups cover every
workspace package. Long scientific cohorts use the filters declared in
`scientific/policy.json`.

| Required lane | Coverage |
| --- | --- |
| Ordinary regressions | Language, runtime, geometry/analysis and tooling/distributed on three operating systems |
| Quality | Formatting, architecture, language ledger, metadata/governance validators, strict first-party Clippy and all-target compiler checks |
| Reviewed FEA debt | The approved architecture diagnostics and Clippy source-anchor/emission inventory, tracked by RM-1287 |
| Scientific receipts | Input identities and the committed evidence or approved disposition for each declared scientific cohort |
| Native products | One distributed CLI build per target; staged CLI execution, original errors and native-image compilation/execution against those bytes |
| Web product | One optimized SDK/WASM/CAD package; SDK tests, pinned publication checks, FEA authoring, exact-CAD adapter checks and bounded Chrome workers |

The Rust test driver links integration targets in batches of four, executes
each batch, and releases its completed test executables in the exclusive CI
target directory. Dependency libraries remain available to later batches and
are cached per platform/lane at a stable path for subsequent CI runs.
Each Linux group owns a mode-0700 runtime directory for its process tree;
the directory stays live through all batches and cold-reopen tests.
Standalone benchmark programs run when the driver receives `--benchmark-smokes`.
The quality lane retains all-target compile coverage.

The geometry catalog includes import-only witnesses. Their integration entries
exercise exact-CAD admission and expected solid-region counts; an empty
`mesh_scenarios` list establishes no meshing accuracy or performance evidence.
Existing governed mesh scenarios and their assertions remain in the manual lane.
The cheap catalog validator checks complete fixture coverage before compilation.

## Offline and manual qualification

Use the existing build machines and warm caches for expensive qualification.
Freeze the source commit and record commands, target/toolchain, artifact hashes,
pass/failure/skip totals and original logs. Renew affected receipts when their
relevant inputs change. `scripts/qualification/scientific.py verify` checks the
committed receipt dispositions on every dev candidate.

The full circuit large-board benchmark runs offline on the recorded reference
host in release mode under
`circuit-large-board-performance`. It checks a board with at least 50,000
primitives, edit deltas, fresh-compile equivalence and the 500 ms edit budget.
Ordinary geometry CI excludes this timing case and checks its required receipt;
the smaller circuit functional tests remain in the ordinary suite. Use
`python3 scripts/qualification/scientific.py command --cohort circuit-large-board-performance`
to obtain the exact command.

The two full-model FSI/CHT CLI witnesses use the manual
`coupled-full-model-witnesses` cohort. Its owner-approved exception records the
known wall-time and memory-ceiling failures. Keep their solve, publication and
assessment assertions intact, and report actual outcomes when running them.
Smaller coupling regressions remain in ordinary CI.

GitHub also provides explicit manual workflows:

- `offline-qualification.yml`: scientific conformance, full GPU and performance qualification, selected by exact source ref.
- `offline-meshing.yml`: governed meshing corpus and performance qualification, selected by exact source ref.
- `builtin-example-closure.yml`: exhaustive public-product example closure, tracked in the 1.0 hardening effort.
- `wasm-bindings.yml` with `headless_tests: true`: the extended Rust headless WASM suite. Set `publish: false` for qualification without publication.

Release validation reuses the ordinary CI gates. The release workflow builds,
signs, stages and accepts each final native product once. Its archive contains
the accepted executable, required redistributables, publisher notices and
RunMat license files.
