# Authored canonical acoustic execution envelope

This is a bounded execution envelope, not Stage 8 qualification.
The authored local CLI path admits a fresh, direct harmonic acoustic step over an authored
frequency list (one or more frequencies, solved and published in authored order through the
shared frequency publication contract in [frequency-publication.md](./frequency-publication.md)).
A one-frequency step also publishes its complete `AcousticComplex` output state. It uses the
canonical mesh Helmholtz worker, not the legacy graph acoustic solver. Multi-frequency semantic
states (a retained state per frequency), unsupported observations, restart inputs, coupled
sequences, authored remote submission and accelerators stay closed. Witnesses: the
`acoustic-cavity-harmonic.fea` example run in `crates/runmat-cli/tests/check_cli.rs` (three
frequencies) and the harmonic snapshot capture and verification in
`crates/runmat-cli/tests/fea_snapshot_cli.rs`.

The chain is checked source/materials → existing exact geometry and meshing → canonical study
planning → shared task pool → accepted numerical result/receipt → shared accuracy evaluation →
state publication → metered terminal publication → authenticated shared run artifacts.
Each worker phase retains its exact resource plan and scheduler response/accounting association.
Cold completion import reconstructs and validates that chain; it does not execute another solve
or manufacture a new scheduler attestation. A caller supplies the trusted root digest and access
authority. The published catalog must contain the entire retained closure.

Public pressure is nodal complex pressure; particle velocity is the accepted kernel's complex
volume-average per element. State and cold-query fields retain the same values. Runtime copies
these accepted native fields and source evidence without assembling or solving again. The
generic singleton completion preserves the actual acoustic operation, and the shared manifest
retains the authored definition, canonical mesh, numerical catalog and public run. Fresh local
publication identity includes the authenticated completion root, not only the scientific definition.

The generated unit-box CLI fixture covers Tet4/Tet10, producing process exit, cold mesh/run/field
and semantic-state reads, catalog payload authentication, and catalog-only whole-completion
replay with rejection of a foreign trusted root. Its small-wavenumber uniform-wall-pressure
witness is execution evidence, not an independent continuum accuracy reference. The independent
duct/cavity, phase, velocity and power evidence remains in [canonical-acoustic.md](canonical-acoustic.md).

Residual and equation-based power closure do not supply missing spatial/field/frequency convergence.
Mandatory unavailable accuracy evidence remains indeterminate; the result publishes, marked
indeterminate, only under a `markIndeterminate` unavailable-evidence policy and is otherwise blocked.
Reservations are conservative estimates, not measured peak-memory qualification. Process exit
and cold artifact reopening do not prove failed-process replacement, retry recovery or restart.
Wider boundaries, observations, spectra, remote/distributed execution, measured resources and
blocking release evidence remain open gates.

```sh
cargo test --locked -p runmat-analysis-execution --lib acoustic_study
cargo test --locked -p runmat-execution-runner-native --lib typed_fea::preparation::admission
cargo test --locked -p runmat --features occt-native --test typed_fea_geometry_cli authored_acoustic_orders
```
