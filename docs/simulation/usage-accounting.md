# Simulation usage accounting

This document defines how RunMat counts floating-point operations for simulation
usage limits under the RunMat license and subscriptions. These counts determine
consumption of a simulation allowance. They do not measure physical quantities,
CAD dimensions, simulation accuracy or execution speed.

RunMat's simulation allowances are based on numerical work rather than how long
a simulation takes. We considered core-hours, which charge for allocated CPU
capacity over time. We chose operation-based accounting so that faster execution
does not, by itself, change the usage charged. Our aim is to make simulations
faster while giving you an allowance you can relate to practical engineering work.

We calibrate the included allowances against representative projects: simple
brackets and everyday objects for free use, and substantial design-and-simulation
workflows for Scale. Published examples identify the model, physics, mesh, load
cases, design iterations and measured usage. Geometry alone does not determine
the amount of numerical work.

## Units

One GFLOP equals exactly 1,000,000,000 measured floating-point operations under
this specification. GFLOP measures an amount of work. GFLOP/s measures speed and
is not the unit of the simulation allowance.

Displays use decimal scales of the same operation count:

| Display unit | Operations |
| --- | ---: |
| GFLOP | 1,000,000,000 |
| TFLOP | 1,000,000,000,000 |
| PFLOP | 1,000,000,000,000,000 |

For example, 2,500 GFLOP and 2.5 TFLOP describe the same work. Changing the display
scale does not change the allowance consumed or round the stored count.

Measurement uses defined algorithmic operation counts, not physical instruction
counters or a conversion from elapsed time. Counters retain exact integer
operation totals; displayed decimal GFLOP are a presentation of those totals,
not rounded values used to settle usage.

## Basic counting convention

| Operation | Measured operations |
| --- | ---: |
| Real floating-point addition, subtraction, multiplication or division | 1 each |
| Fused multiply-add | 2 |
| Complex arithmetic | The constituent real operations in the specified algorithm |
| Vector, SIMD or GPU arithmetic | The defined operations for each active numerical element |

Single and double precision use the same convention. Parallel execution and
instruction fusion do not add charges. Different algorithms or different numbers
of iterations can perform different amounts of numerical work on the same model.

For a real dot product of positive length n, the direct product-and-sum algorithm
performs n multiplications and n - 1 additions: 2n - 1 operations. A backend's
instruction sequence does not change that reference count. An empty dot product
has zero arithmetic operations under this convention.

A direct complex dot product of length n > 0 counts six real operations per
complex product and two per subsequent complex addition: 8n − 2. Sparse
matrix-vector multiplication sums these row-dot counts over stored entries.
An empty row contributes zero; an explicitly stored zero still participates in
the row's reference arithmetic. Counts are the same for Single and Double.
The metered CSR operator quotes its immutable row pattern once and reserves the count
before each application. A refused reservation leaves the output unchanged.

Integer arithmetic, comparisons, memory movement, communication and waiting are
not floating-point operations and are unmetered. This deliberately leaves some
meshing work uncharged; no arbitrary conversion weight is assigned to it.

## Numerical-operation schedule

The released specification must identify exact reference algorithms or formulas
for each supported measurement boundary:

- Meshing: numerical geometry operations, including their special-function rules.
- Assembly: element formulation, integration work and matrix/vector accumulation.
- Direct solvers: actual symbolic structure/fill and the reference factorization
  and triangular-solve arithmetic; a generic dense n-cubed estimate is insufficient.
- Iterative solvers: executed matrix/vector operations, preconditioning, reductions
  and completed iterations.
- Nonlinear and transient analyses: constituent numerical work across iterations
  and time steps, without charging it again at enclosing study levels.
- External libraries: documented reference counts with the same precision and
  fusion conventions, rather than undocumented backend multipliers.

Square roots, transcendental functions, complex division and other nontrivial
operations require explicit rules. The basic table alone is not a complete
release measurement schedule. These rules and the supported-operation inventory
must be completed and reviewed before this draft is published as version 1.

### Sparse LLT reference arithmetic

For a column of an LLT factor with p stored entries below its diagonal, the
right-looking reference divides those p entries by the diagonal and updates the
lower triangle of their outer product. Its basic real arithmetic count is
p² + 2p. For Hermitian complex factors the count is 4p² + 2p: division by the
real diagonal costs 2p, diagonal updates cost 4p and off-diagonal updates cost
4p(p − 1). Each column also invokes one real square root; those calls remain
separate until the intrinsic-counting rule is finalized.

Forward and backward substitution together cost 4p + 2 real operations per
column and right-hand side, or 16p + 4 for a Hermitian complex factor. Reusing
a factor for k right-hand sides counts k substitutions without repeating its
factorization. Column sizes come from the stored symbolic fill, including the
lower-triangular blocks and trailing patterns of supernodes. Counts use checked
integers rather than the solver's floating-point performance estimate.

## Reuse, interrupted runs and retries

Loading, viewing or reusing an existing output does not charge its construction
again. New numerical work using that output is counted. Examples include solving
a new load case with an existing mesh, or applying an existing factorization to
another right-hand side.

Settlement charges completed numerical work. An abandoned run expires 24 hours
after its last usage report, charging only the completed work last durably
reported and releasing its unused grant. Reserved operations are not completed
work. Recovery must preserve completed-work identities to avoid charging the
same work twice; that ownership mechanism remains a release requirement.

## Estimates, budgets and usage receipts

A pre-run estimate is distinct from measured usage. Adaptive meshes, nonlinear
convergence and iterative solves can require additional work. Runs need an
fixed spending cap and admission for additional work before that cap is exceeded.
The default cap is the available allowance plus eligible packs at admission.
Users can choose a lower cap; a grant renewal cannot increase that run's cap.

A usage receipt identifies the accounting version, run/attempt identities,
completed numerical work by category, exact total operations and settlement state.
For example, its user-facing breakdown has categories for meshing, assembly,
factorization and solver iterations. Example totals must come from actual runs,
not invented figures.

Measurement versions remain available so users can interpret past receipts.
Resource limits for memory, CPU capacity and wall time remain separate from the
simulation usage meter. Cluster access and any cluster usage charges are also
separate entitlements.

## Allowances and representative projects

The included simulation allowance and renewal period are published at
https://runmat.com/pricing. The applicable Free Use grant or commercial agreement
identifies which allowance applies. Eligibility verification for academic use is
available at https://runmat.com/contact?type=academic.

Representative projects must include measured receipts and enough information to
reproduce their workloads. A full-house design workflow is a calibration target,
not a guarantee about every house model or a measured release claim.
