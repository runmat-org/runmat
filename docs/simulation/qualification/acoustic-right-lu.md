# Shared right LU and canonical Helmholtz — bounded native qualification

This is a numerical correction, not an acoustic qualification gate.
The original refined Tet10 authored duct failed under the authored 4,096-iteration
budget; the previously tried diagonal scaling was discarded. No mesh, tolerance,
iteration budget, physical source or material law is relaxed here.

## Shared numerical authority

`solve/krylov/right_ilu.rs` implements fixed-support nonsymmetric LU(0), distinct
from the existing positive-pivot SPD LDL-transpose preconditioner. Signed finite
invertible pivots are accepted. Missing/zero/nonrepresentable pivots and nonfinite
arithmetic reject; there is no shift, permutation, hidden fill or fallback.
One forward unit-L and one true upper-U solve apply the inverse. Setup owns CSR
offsets, columns, values and diagonal positions, with no auxiliary row maps.

The independent method reference is [Saad, Iterative Methods for Sparse Linear
Systems, second edition](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf),
sections 9.3.2 and 10.3.2. Right preconditioning preserves the unweighted residual
criterion; zero-fill factorization retains the selected sparse pattern. The
specific implementation, resource bound and qualification results below are
RunMat evidence, not claims made by that reference.

## Physics and execution boundary

The pressure operator is materialized exactly as
`[[K-omega^2 M, -omega B], [omega B, K-omega^2 M]]`, then transformed by the
existing shared affine reduction. The canonical workload explicitly records
`None` or `Ilu0`; policy, result validation, identity and publication provenance
must agree. Missing policy fields reject instead of silently adopting a default.

Sparse operator and factors are prepared once and reused during power refinement.
A retained physical warm start x0 forms the correction equation
`A M^-1 y = b-A x0`; recovery is `x=x0+M^-1 y`, never a reinterpretation of x0 as y.
Every continuation spends the same total authored iteration budget. Final
algebraic acceptance independently applies the original separate K/M/B operators;
power acceptance likewise uses the original physical boundary balance.

Revision 50 reserves full/reduced discovered assembly and CSR coexistence, factors,
the reduction clone and correction/application buffers in addition to existing
K/M/B and Krylov reservations. Full real support is bounded by twice the sum of
K/M/B supports. With no active periodic relation each recovery row has at most
one coefficient, bounding reduced support by twice that original support sum.
Active phase relations require two coefficients and at most four contributions
per full entry, bounding reduced support by eight times the original sum. Typed
reservation facts reject substitution/understatement; no authored cap increases.
These are analytical heap estimates, not measured peak RSS or GPU evidence.

## Evidence and remaining gate

Initial isolated shared-factor tests pass 6/6: independent signed nonsymmetric
literal L/U, missing-fill pattern, tiny/large negative pivots, invalid arithmetic,
dimensions/cancellation, and right-GMRES physical solution/residual checks.
Reduced solve/refinement tests pass 5/5: independent complex phased lifting,
sparse versus separate-operator action, nonzero physical warm starts, unit/tiny/
large loads, cancellation, and unchanged power/iteration-budget rejection.
The same frozen-graph FEA test binary also passes all 11 acoustic execution tests:
explicit policy identity and rejection, receipt/import validation, cold complex
field domains and paging, Bloch lineage, and independent frequency fan-in.
All 19 canonical acoustic tests pass, including matched traveling-wave, rigid
cavity convergence on both sides of resonance, phased constraints, and recovery.
The isolated native runner library compiles successfully with `--locked` (16m33s).
That native check precedes only the two test-only corrections described below.

The first resource run passes 6/7. Its new one-byte-under test incorrectly changes
the model serialization while comparing against the prior request; the test now
changes only the exact plan's typed ceiling. The corrected resource suite passes
7/7, including exact typed reservations and one-byte-under rejection. Strict FEA
library/test no-deps Clippy passes after correcting one needless test borrow.
The first 30 isolated source/test paths pass scoped formatting. The initial
eight-case authored duct run compiled in 25m03s and passed both
coarse Tet10 generated/explicit cases. It failed after 496.50s at fine generated
Tet10 target-pool admission before solving, not at Krylov convergence. Revision50
removes unnecessary general-phase reservation for nonperiodic recovery; the
default 1GiB pool and authored ceilings are unchanged. Revision50 focused resource
tests pass 7/7, including periodic and typed-reservation rejection. The unchanged
eight-case actual CAD run compiled in 13m31s and passes both coarse Tet10 drives;
the fine generated Tet10 case now passes (3,930 nodes, 2,441 elements). Its
pressure/velocity/intensity errors are respectively `2.28695658e-6`,
`1.71394379e-5`, `9.22480455e-6`, with original residual/power, packed full/partial
field and cold closure checks passing. Fine explicit Tet10 also passes with identical
mesh and errors, completing both Tet10 drives and their unchanged refinement gates.
Both coarse Tet4 drives pass (119 nodes, 347 elements); fine generated Tet4 passes
(574 nodes, 2,441 elements), with pressure/velocity/intensity errors respectively
`5.75423086e-4`, `4.46759829e-3`, `1.42830992e-4`. Fine explicit Tet4 passes with
identical mesh and errors. The original eight-case native matrix passes in
2,182.31s, including generated/explicit equivalence and both orders' original
refinement/finest gates, authenticated source, residual/power acceptance, packed
full/partial complex fields and cold reopening with foreign-root rejection.
The native executable and test executable hashes were unchanged through this run.

Full frozen-graph FEA regression is not green: 1,331 pass and 11 fail (1,313.32s).
Failures cover a strict floating-point load assertion, three overly restrictive
thermal activation admissions, a stale thermal backend reservation assertion,
electrical/modal proof ceilings, recovery-estimator refinement and three transient
continuum/restart gates. These are not waived by the acoustic checks.

Final read-only review found a zero-effective-RHS normalization inconsistency.
A correction now uses the original restricted source/lifting scale
when cancellation makes the effective RHS exactly zero, and the shared fixed
initial-residual convention for genuinely homogeneous warm starts. Target,
original acceptance and reporting use the same reference with no unit floor;
nonfinite norms and nonrepresentable targets reject. A literal independent
`K+iI` affine/homogeneous cold/warm test spans None/Ilu0 and load scales
`1e-12`, `1`, `1e12`. This correction/test and revision50 are imported into the
isolated duct graph. All six response/refinement tests pass, including the new
edge case; 14 canonical physics and 11 execution tests also pass. All 31 scoped
Rust paths pass formatting. Final strict no-deps FEA library/test lint passes
(13m50s including build-lock wait).
Preparation drops source/lifting/
shift temporaries before factoring; one extra restricted-vector temporary stays
within the existing simultaneous row/correction allowance.

This qualifies the bounded single-frequency matched-duct native envelope, not
the entire acoustic formulation. Acoustic radiation,
frequency sweeps, process restart, remote/GPU/distributed operation envelopes,
measured peak resources and release qualification remain separate open gates.
