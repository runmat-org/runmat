# Canonical solid recovery qualification

## Original cantilever gate

The original Tet10 cantilever uses the unit cube, fully fixed root, steel
`E = 200 GPa`, `nu = 0.3`, and a resultant tip load `(0, -1000, 0) N`.
The original uniform divisions remain `1, 2, 3`. The recovery estimator must
fall by at least 10% per rung; the final peak-displacement change must be less
than 0.25 times the preceding change. These are separate mandatory assertions.

The latest original run retains estimator values approximately
`372.95, 270.29, 195.79`, and peak displacements
`3.10020e-8, 3.40295e-8, 3.48526e-8 m`. The estimator assertion passes;
the displacement assertion fails, with change ratio approximately `0.271869`.
Neither the meshes nor the criterion has been replaced.

## Discrete-equilibrium discriminator

A test-only dense Cholesky factorization independently solves the full original
free-DOF systems (54, 300, 882 DOFs). It uses the original canonical stiffness
and physical load, independently of the production sparse solver, ordering,
preconditioner and recovery. Scaling matrix and load by the same maximum
stiffness diagonal avoids tying the calculation to a choice of force units.
A fixed three-by-three system with an analytic solution checks the reference
calculation over 200 orders of magnitude in coefficient scale.

On the original meshes, dense residuals are `1.08e-14` to `2.55e-13` relative;
production displacements differ by at most `6.76e-10` relative. The reference
retains the same `0.271869` peak-change ratio. Separate assertions check the
resultant load `(0, -1000, 0) N`, moment `(500, 0, -1000) N m` about the origin,
zero clamp displacement, and `2 * strain_energy = load_work`.

This evidence rules out sparse-solve error as the explanation for the failed
original displacement ratio. It does **not** independently validate stiffness
assembly or certify continuum error: the dense calculation consumes the same
assembled stiffness. It does not justify changing the integration rule,
preconditioner, solver tolerance or original qualification criterion. Assembly
accuracy and spatial approximation remain candidates for investigation. Rising
near-clamp point-stress peaks alone do not establish a singularity diagnosis.

## Ownership and remaining gates

The qualification coordinator retains original test names and delegates mesh
construction, manufactured solutions, curved patches and heterogeneous material
references to separate modules. Extraction preserves the nine original test and
mesh-helper function bodies after whitespace normalization. The dense reference
is confined to tests; it is neither a new production solver nor a certification
or resource-admission path.

The original cantilever convergence gate remains failed. This diagnostic adds no
production, adaptation, stage, platform or whole-resource qualification. Original
independent material, curved geometry and manufactured field checks remain
mandatory; successful equilibrium alone does not close recovery or adaptation.
