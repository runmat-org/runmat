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

The included allowances are intended for simple brackets and everyday objects
under free use, and larger design-and-simulation workflows under Professional. Usage
depends on the physics, mesh, load cases, iterations and numerical algorithms;
geometry alone does not determine the amount of measured work.

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

OCCT CAD preparation and tessellation are excluded from `rsua-1`. They consume
computing resources but do not consume the simulation allowance. Numerical
meshing, assembly and solver work outside that OCCT boundary follows the
counting rules below.

## Basic counting convention

| Operation | Measured operations |
| --- | ---: |
| Real floating-point addition, subtraction, multiplication or division | 1 each |
| Fused multiply-add | 2 |
| Real square root or transcendental-function evaluation | 1 per scalar evaluation |
| Complex arithmetic | The constituent real operations in the specified algorithm |
| Vector, SIMD or GPU arithmetic | The defined operations for each active numerical element |

Single and double precision use the same convention. Parallel execution and
instruction fusion do not add charges. Different algorithms or different numbers
of iterations can perform different amounts of numerical work on the same model.

The square-root and transcendental rule counts one evaluation in the reference
algorithm, not the arithmetic instructions inside a library implementation.
It applies to real square root, exponential, logarithmic, power, trigonometric,
inverse trigonometric, hyperbolic and inverse hyperbolic functions. For example,
each real `sqrt(x)`, `exp(x)`, `log(x)`, `pow(x,y)` or `atan2(y,x)` evaluation
counts once. A complex function follows its specified real decomposition;
it is not counted again on top of those constituent operations. Integer powers
implemented by a reference multiplication sequence count that sequence instead
of an additional power call. Real absolute values, sign changes and comparisons
contribute zero. Backend approximations, range reduction and instruction lowering
do not change these reference counts.

Real `hypot(x,y)` also counts one reference evaluation. Complex multiplication
uses four real multiplications and two additions (6). General complex division
uses a three-operation denominator, two three-operation numerators and two real
divisions (11). Complex reciprocal uses that denominator and two divisions (5).
A complex multiply followed by complex subtraction counts 8. A magnitude uses
one `hypot` evaluation when the imaginary component is nonzero; the real-only
absolute-value path contributes zero.

The reference three-dimensional orientation determinant uses nine coordinate
differences, six cross-product multiplications, three cross-product
subtractions, three final multiplications and two accumulation additions:
23 operations per evaluation. Adaptive-precision predicates implement this
logical determinant with additional precision when needed; their expansion
arithmetic is backend machinery, not additional reference work. Coordinate
normalization and other numerical work surrounding the determinant have their
own counts. Predicate admission precedes the evaluation, and failed geometry
paths retain the completed reference prefix.

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

RunMat applies reference counting at these numerical boundaries:

- Meshing: numerical geometry operations, including their special-function rules.
- Assembly: element formulation, integration work and matrix/vector accumulation.
- Direct solvers: actual symbolic structure/fill and the reference factorization
  and triangular-solve arithmetic; a generic dense n-cubed estimate is insufficient.
- Iterative solvers: executed matrix/vector operations, preconditioning, reductions
  and completed iterations.
- Nonlinear and transient analyses: constituent numerical work across iterations
  and time steps, without charging it again at enclosing study levels.
- External libraries: documented reference counts with the same precision and
  fusion conventions, or an explicit exclusion such as OCCT preparation and
  tessellation in `rsua-1`.

The primitive costs above apply within these stages. The following sections
describe the measured batches and explicit exclusions in `rsua-1`.

### Sparse LLT and LDLT reference arithmetic

For a column of an LLT factor with p stored entries below its diagonal, the
right-looking reference divides those p entries by the diagonal and updates the
lower triangle of their outer product. Its basic real arithmetic count is
p² + 2p. For Hermitian complex factors the count is 4p² + 2p: division by the
real diagonal costs 2p, diagonal updates cost 4p and off-diagonal updates cost
4p(p − 1). Each column also invokes one real square root, adding one operation
to either column total. The total factorization count is p² + 2p + 1 per real
column or 4p² + 2p + 1 per Hermitian complex column.

Forward and backward substitution together cost 4p + 2 real operations per
column and right-hand side, or 16p + 4 for a Hermitian complex factor. Reusing
a factor for k right-hand sides counts k substitutions without repeating its
factorization. Column sizes come from the stored symbolic fill, including the
lower-triangular blocks and trailing patterns of supernodes. Counts use checked
integers rather than the solver's floating-point performance estimate.

The LDLT reference retains the pre-division column when applying each trailing
update. Its factorization has the same basic arithmetic as LLT and no square
roots: p² + 2p per real column or 4p² + 2p per Hermitian complex column.
Its unit-diagonal triangular substitutions and separate real-diagonal solve
cost 4p + 1 operations per real column and right-hand side, or 16p + 2 for
Hermitian complex values. Pivoting block factorizations require their own
reference schedules; these formulas do not describe Bunch–Kaufman or LU.

For `rsua-1` in RunMat 0.7.0, faer's sparse LU and sparse Bunch–Kaufman
numeric factorization, refactorization and substitution kernels are explicitly
excluded from the simulation operation count. Pivot choices and their resulting
work do not have an approved exact reference schedule. The solver classifies
these two kernels by factor kind before admitting a numerical batch; an
unsupported or future kind cannot silently become measured zero. Their
solutions remain available in a measured FEA run. Residual construction, norms,
correction updates, assembly and other independently measured work in the same
run still consume allowance. The hand-written small dense LU kernel has its own
reference batches and is not part of this sparse-library exclusion.

Modal and buckling eigen kernels are also excluded from `rsua-1` in 0.7.0:
faer's dense symmetric eigen-decomposition and the outer Lanczos/LOBPCG,
shift-selection, Ritz, inertia and eigenpair postprocessing algorithms do not
have complete approved reference schedules. This exclusion applies to those
eigen kernels, not to separately metered sparse matrix applications, vector
operations, preconditioners or LLT/LDLT direct solves they invoke. The excluded
kernels consume computing resources without consuming simulation allowance.

An LLT or LDLT numeric factorization, refactorization or substitution is one reference
batch. The solver reserves its count before numeric allocation or changes to
the right-hand side, then records it when that batch succeeds. A refused batch
leaves the factor and right-hand side unchanged. A factorization that fails
before producing a factor does not complete that batch; earlier completed
batches retain their counts. Cancellation observed after a batch completes
does not erase its work. These boundaries apply to admitted simulation work;
ordinary math execution without a simulation ledger needs no spending grant.

### Iterative solver reference arithmetic

PCG, GMRES, MINRES and BiCGStab charge the batches they execute. An early
convergence or rejected proposal does not charge later iterations. Matrix
applications and preconditioner applications retain their own owners; the
recurrence does not add their counts again.

For vectors of positive length n, the common reference batches are:

| Batch | Real operations | Complex operations |
| --- | ---: | ---: |
| Dot product | 2n − 1 | 8n − 2 |
| Euclidean norm, including square root | 2n | 4n |
| Multiply by a real scalar | n | 2n |
| Add a real-scaled vector | 2n | 4n |
| Add a scalar-scaled vector | 2n | 8n |
| Form `rhs − product` | n | 2n |
| Divide by a positive real norm | n | 2n |
| Update solution and residual with a real step | 4n | 8n |
| Gram–Schmidt projection against k retained vectors | k(4n − 1) | k(16n − 2) |

Empty vectors and empty projection sets contribute zero. Scalar recurrence
work is counted separately when its branch executes. A zero-denominator
Anderson proposal, for example, completes its residual difference and two dots
but performs no coefficient division or proposed-image update. Its real
reference totals are 5n − 2 before division, 5n − 1 after division, and 8n − 1
when the image update completes.

Preconditioner setup and application are separate work: reusing a Jacobi, ILU
or AMG setup does not charge that setup again. A V-cycle counts the operations
in its retained hierarchy and selected sweeps; unused maximum iteration counts
are neither reserved nor charged as completed work.

### Structural result recovery

Result recovery counts new numerical work even when it uses an existing solution.
Copying the solution into a field or reading a completed field does not consume
operations. Diagnostic summaries reuse the completed strain field; they do not
repeat element recovery.

The reference recovery kernels use these schedules:

| Kernel | Reference operations |
| --- | ---: |
| Tet4 strain projection | Six length-12 dot products: 138 |
| Tet4 integration-point volume scaling | 1 |
| Symmetric stress tensor from strain and retained Lamé parameters | 17 per tensor |
| Three-dimensional von Mises stress | 17 per tensor |
| Strain-energy density from a strain/stress pair | 12 per tensor |
| Tet10 extrapolation from four integration points to ten nodes | 217 per element |
| Volume-weighted tensor accumulation | 13 per contributing element node |
| Region tensor normalization | 6 per node-region pair |
| Comparing two recovered region tensors by von Mises stress | 34 per comparison |

Geometry, material parameters and quadrature have their own primitive batches;
the recovery schedules do not charge those primitives again. Curved Tet10
geometry uses its selected adaptive validation stages before the fixed recovery
points. Thermal energy integration counts each selected quadrature rule, including
additional rules used for convergence. It does not reserve or charge an unused
maximum-order rule.

These are numerical-kernel schedules. Input and constitutive invariant checks,
field-shape checks, attribution and diagnostic formatting are outside these
reference batches. Their implementation details do not change the charge.

### Structural transient integration

Average-acceleration Newmark integration counts each predictor, completed state
and correction when it executes. For n displacement degrees of freedom:

| Batch | Real operations |
| --- | ---: |
| Displacement and velocity predictor, including time-step coefficients | 6n + 5 |
| Complete displacement and velocity from the solved acceleration | 4n + 3 |
| Apply an acceleration correction to displacement, velocity and acceleration | 5n + 3 |
| Form initial acceleration RHS `load − damping − elastic` | 2n |
| Form an equilibrium residual from load and three force vectors | 3n |
| Form a displacement increment | n |
| Average two load, velocity or reaction vectors | 2n |

Mass, damping and stiffness applications use the CSR row-dot schedule above.
Effective-operator construction counts its executed coefficient scaling and
accumulation. Linear factorization, substitutions and iterative corrections keep
their solver counts; integration does not charge those operations again.

Dynamic-balance histories count the matrix applications, dot products, norms,
energy calculations and work increments needed to produce each snapshot. A
recovered history counts numerical reconstruction it performs. Reading an
already computed snapshot consumes no additional operations.

Constraint projection counts two real operations per retained recovery
coefficient. Expansion counts the reference coefficient products and sums plus
the affine offsets that it applies. Fixed-DOF selection and vector copies have
no floating-point arithmetic charge. Input validation remains outside these
numerical batches.

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
convergence and iterative solves can require additional work. Runs need a
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
