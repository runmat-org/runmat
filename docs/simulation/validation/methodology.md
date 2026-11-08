## How a check is built

A reference check solves one problem in RunMat and compares a named quantity with an independent reference. The reference is one of:

- a closed-form solution, such as Euler's buckling load or the plane-wave transmission loss of a layer;
- a manufactured solution, where a chosen field sets the loads and the solver must recover that field;
- a patch test, where an affine or linear field must be reproduced exactly;
- a published benchmark.

Each check is a test that asserts the compared quantity against the reference, or a public study run whose values are kept as receipts. The test carries a witness tag naming the requirement obligation it supports and the check it belongs to. The capability ledger, `docs/simulation/ledger.json`, defines each check: the problem, the reference, the compared quantity and the tolerance. The family pages and [Status](./status.md) are generated from the ledger and the latest evidence run.

To add a check, add an entry to the ledger's `checks`, tag its tests with `; check: <id>`, run `evidence_report`, and run `node scripts/docs/generate-pages.mjs`.

## Tolerances

The tolerance is the bound the test asserts, and the family page quotes it as written in the test. Three kinds are used:

- Exactness bounds of 1e-8 to 1e-12 where the discretization reproduces the reference exactly: patch tests, uniform fields, energy and power balances, and closed-form arithmetic.
- Error bands, such as 1 % on a buckling load or 0.1 dB on a transmission loss, where the discretization error is finite at the mesh used.
- Convergence rates, where the error must fall by a stated factor per refinement and stay under a bound on the finest mesh.

A receipt-backed check has no pass band. Its table reports the value, the reference and the difference, and the check states how to read them.

## Mesh-convergence practice

A check whose error depends on mesh density, time step or load increments runs a sequence of at least two refinements, usually three. It asserts both the error ratio between successive refinements and a bound on the finest mesh. For a method of order $p$, halving the element size or time step divides the error by about $2^p$. Tests assert a ratio a little below that value, such as 3.8 for a second-order time integrator, so a correct implementation passes and a lost order of accuracy fails.

Quantities are compared where the reference holds. Stress is read away from idealized supports, because a fully fixed face is a singularity whose peak value does not converge. The cantilever check reads stress one section depth from the clamp and records the clamped-edge peak only to show that it does not settle.

## Where receipts live

A receipt records one value from a public study run: the quantity, the mesh, the CLI build, the commit, the date and the result id. Receipts are kept in `docs/simulation/examples/catalog.json` under each example, and a value on a docs page links to its receipt. The docs lint fails if the linked text differs from the receipt value at the precision shown.

Test results come from the evidence catalog that `evidence_report` writes to `docs/simulation/qualification/generated/`. It records each tagged test's result and run date, and the family pages show the latest run for each check.
