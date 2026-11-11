# Simulation usage accounting

Simulation runs draw on a simulation allowance. RunMat measures that usage as the
floating-point operations the run performs: meshing, assembly, solvers and result
recovery. The same algorithm on the same model counts the same work however fast
it runs, on however many cores or GPUs, so faster execution alone never changes
the charge. Usage does not measure physical quantities, CAD dimensions, accuracy
or elapsed time.

The included allowances are intended for simple brackets and everyday objects
under free use, and larger design-and-simulation workflows under Professional. Usage
depends on the physics, mesh, load cases, iterations and numerical algorithms;
geometry alone does not determine the amount of measured work.

## The unit

Usage is counted in GFLOP. One GFLOP is exactly 1,000,000,000 measured
floating-point operations. GFLOP measures an amount of work; GFLOP/s measures
speed and is not the unit of the allowance.

Larger amounts display on decimal scales of the same count:

| Display unit | Operations |
| --- | ---: |
| GFLOP | 1,000,000,000 |
| TFLOP | 1,000,000,000,000 |
| PFLOP | 1,000,000,000,000,000 |

2,500 GFLOP and 2.5 TFLOP describe the same work. RunMat stores the exact integer
count; the decimal display does not round what is charged.

## What is counted

Each numerical step counts the operations of a defined reference algorithm. The
count does not come from hardware instruction counters or from elapsed time.

| Operation | Measured operations |
| --- | ---: |
| Real addition, subtraction, multiplication or division | 1 each |
| Fused multiply-add | 2 |
| Real square root or transcendental function (`sqrt`, `exp`, `log`, `pow`, `atan2`, …) | 1 per evaluation |
| Complex arithmetic | Its real operations, for example 6 for a complex multiplication |
| Vector, SIMD or GPU arithmetic | The operations for each active element |

Single and double precision count the same. Parallel execution and instruction
fusion add nothing. A different algorithm or a different number of iterations on
the same model can do a different amount of work, and the count reflects it.

Metered steps follow reference schedules. For example, a real dot product
of length n counts 2n − 1, and each conjugate-gradient iteration counts its
matrix application, preconditioner application, dot products and vector updates.
[How each operation is counted](./usage-accounting-reference.md) lists the
schedules for meshing predicates, sparse factorizations and substitutions,
iterative solvers, result recovery and transient integration, and which
schedules are still being written before accounting version 1 is published.

## What is not counted

- OCCT CAD preparation and tessellation (excluded from accounting version `rsua-1`).
- In RunMat 0.7.0 (`rsua-1`), two kinds of kernel whose pivoting or iteration has no
  approved reference schedule yet: sparse LU and Bunch–Kaufman factorizations and
  their substitutions, and the modal and buckling eigen kernels (dense symmetric
  eigen-decomposition, Lanczos and LOBPCG iteration, shift selection and eigenpair
  postprocessing). The solutions are still produced. The matrix applications,
  vector operations, preconditioners, LLT and LDLT solves and assembly around
  them are counted.
- Integer arithmetic, comparisons, memory movement, communication and waiting. Some
  meshing work is therefore uncharged; RunMat assigns it no conversion weight.
- Input validation, field-shape checks and diagnostic formatting.
- Loading, viewing or copying a result that already exists.
- Ordinary math execution outside a simulation run.

Memory, CPU capacity and wall-time limits are separate from the usage meter.
Cluster access and any cluster charges are separate entitlements.

## Reuse, cancellations and retries

Reusing an existing output does not charge its construction again. New work that
uses it is counted. Solving a new load case on an existing mesh counts the new
assembly and solve, not the mesh. Applying an existing factorization to another
right-hand side counts the substitutions, not the factorization.

Work is charged when a step completes:

- A step that is refused before it starts changes nothing and charges nothing.
- A step that fails before it produces a result charges nothing for that step.
  Steps completed earlier keep their charge.
- Cancelling after a step completes does not erase that step's charge.
- An iterative solve that converges early charges only the iterations it ran.
  Unused maximum iteration counts are never charged.
- An abandoned run expires 24 hours after its last usage report. It is charged
  for the completed work last durably reported, and the rest of its reservation
  returns to the allowance.

Reserved operations are not charged work. Recovery after an interruption must keep
the identity of completed work so the same work is never charged twice; that
mechanism remains a release requirement.

## Estimates and spending caps

A pre-run estimate is not the charge. Adaptive meshing, nonlinear convergence and iterative solves can need more work
than planned.

Every run has a fixed spending cap. The default cap is the available allowance
plus eligible packs at admission. You can set a lower cap. Additional work needs
admission before it would exceed the cap, and a grant renewal during the run does
not raise that run's cap.

## Usage receipts

Each run produces a usage receipt with:

- the accounting version;
- the run and attempt identities;
- completed work by category, such as meshing, assembly, factorization and
  solver iterations;
- the exact total operations;
- the settlement state.

Accounting versions stay published so past receipts can be read under the rules
they were charged with. Example totals on RunMat pages come from actual runs.

## Allowances and representative projects

The included simulation allowance and renewal period are published at
https://runmat.com/pricing. The applicable Free Use grant or commercial agreement
identifies which allowance applies. Eligibility verification for academic use is
available at https://runmat.com/contact?type=academic.

Representative projects include measured receipts and enough information to
reproduce their workloads. A full-house design workflow is a calibration target,
not a guarantee about every house model or a measured release claim.

## How each operation is counted

The per-algorithm schedules (meshing predicates, sparse LLT and LDLT, iterative
solvers, structural result recovery and transient integration) are in
[How each operation is counted](./usage-accounting-reference.md).
