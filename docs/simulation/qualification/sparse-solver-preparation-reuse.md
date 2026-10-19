# Process-local sparse solver preparation reuse

The shared algebra owner retains at most one coercivity certificate. It retains
no matrix, numerical factor, prior RHS, mesh, physical history or serialized trust.
The carrier cannot be deserialized. A cold worker starts empty and proves its
original operator in that process.

Reuse requires the existing certificate owner to validate the exact prepared
matrix pattern and values. There is no drop tolerance, approximate matrix match or
rounded time-step key. A miss discards the prior certificate and proves the current
operator. Failed replacement cannot supply the stale certificate to later solves.

Each preparation still derives the authoritative constrained matrix and Dirichlet
lifting. Current RHS and optional physics nullspace provenance are freshly bound
by shared admissibility. Solver compatibility, preconditioner setup, current
resource admission, backend policy, cancellation and final residual checks follow
the ordinary shared path. A cached proof cannot relax a new resource policy.

Ordinary and affine-reduced structural solves use this owner. Affine congruence,
constraint consistency and lifting remain with the shared constraint solver.
Direct transient strategy owns one carrier across accepted increments, initial
acceleration refresh and normalized equilibrium corrections; it is absent from
physical checkpoints and reconstructed after cold continuation. Multiplier solves
retain their existing distinct algorithm and clear this SPD carrier.

Resource model revision 74 reserves the carrier's fixed typed storage in direct
transient workspace in addition to the existing full/reduced proof-construction
bound. Modal policy and authored ceilings are unchanged. Reuse never claims measured
RSS, task distribution, accelerator execution or scientific field accuracy.

Independent tests compare warm and cold original PCG results, Dirichlet RHS and
prepared identities; reject one-bit operator changes and foreign constraint
reductions; preserve current physical nullspace provenance; and reject tightened
work admission before provider execution. Affine lifting and changed congruence
are checked against an independent constrained quadratic-energy minimum.

Nominally uniform schedules can produce different floating point interval lengths.
Their different actual operators require fresh validation and may miss reuse.
This mechanism does not coerce them into a common key or change the equations.
All original continuum/history accuracy gates remain mandatory.
