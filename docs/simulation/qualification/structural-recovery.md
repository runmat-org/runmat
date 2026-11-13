# Canonical solid recovery qualification

## Original cantilever gate

The original Tet10 cantilever uses the unit cube, fully fixed root, steel
`E = 200 GPa`, `nu = 0.3`, and a resultant tip load `(0, -1000, 0) N`.
The original uniform divisions remain `1, 2, 3`. The recovery estimator must
fall by at least 10% per rung; the final peak-displacement change must be less
than 0.25 times the preceding change. These are separate mandatory assertions.

The latest original run retains estimator values approximately
`390.13, 287.06, 204.91`, and peak displacements
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
original displacement ratio. It does not certify continuum error: the dense
calculation consumes the same assembled stiffness.

## Independent affine stiffness check

A separate reference constructs physical barycentric gradients from corner
coordinates, identifies midsides by their physical edge coordinates, and
integrates the quadratic stiffness polynomial using closed-form moments:
`integral(lambda_r * lambda_s) = V * (1 + delta_rs) / 20`.
It shares neither production shape gradients nor quadrature rules. The isotropic
elasticity coefficients come directly from the original physical specification.

All global stiffness entries, including zeros, match on the three original
meshes: 6,561, 140,625 and 1,058,841 entries. Relative Frobenius differences are
`1.86e-16, 2.01e-16, 2.40e-16`; maximum coefficient differences normalized to
the largest coefficient are at most `4.62e-16`. This establishes agreement with
the independent affine polynomial integral on these meshes. It does not prove
curved-element integration, arbitrary materials or continuum approximation.
Neither the checked stiffness integration nor the sparse solve explains the
failed peak-change ratio. The spatial approximation remains unqualified; no
singularity diagnosis is inferred from rising near-clamp stress peaks.

## Rotation-invariant recovery metric

Stress components are physical tensor values `[xx, yy, zz, yz, xz, xy]`.
The volume L2 Frobenius norm counts each shear component twice, reflecting its
two entries in the symmetric matrix. Treating those six values as a vector
incorrectly changed the former two-element squared-mean proxy from `0.5` to
`0.4330127` after a 45-degree tensor rotation. Both element and integration-point
paths now use the tensor norm, including the relative roundoff floor. Tests cover
all three coordinate planes, Tet4/Tet10, both layouts, indicators and marking.

The raw total/indicators have units `Pa * m^(3/2)`; they are neither strain energy
nor a dimensionless error. Required `StressTensorFrobeniusL2` metadata names the
norm. Required `reference_norm` retains the original physical stress norm; relative
recovery is the discrepancy divided by that norm. Adaptation iteration schema/domain
V3 binds the norm and shared accuracy plan; old schema, missing norm
and unknown norm records reject. Cold metadata validation is not mathematical
reverification of imported stress fields. Finite source values whose squared
norms overflow reject rather than acquiring a zero-error verdict through an
infinite roundoff floor.

The corrected metric changes the estimator values above. Peak displacements,
meshes, residual targets, resource caps and the original `0.25` convergence
criterion are unchanged; its original failure remains visible.

## Ownership and remaining gates

The qualification coordinator retains original test names and delegates mesh
construction, manufactured solutions, curved patches and heterogeneous material
references to separate modules. Extraction preserves the nine original test and
mesh-helper function bodies after whitespace normalization. The dense and closed-form references
are confined to tests; it is neither a new production solver nor a certification
or resource-admission path.

The original cantilever convergence gate remains failed. This diagnostic adds no
family, adaptation, stage, platform or whole-resource qualification. Original
independent material, curved geometry and manufactured field checks remain
mandatory; successful equilibrium alone does not close recovery or adaptation.

## Element-constant stress: integrate discrepancy before summing

The element-stress path now computes the declared volume L2 norm of the original
constant tensor minus the region-side recovered nodal interpolant. Squaring a cell
average discarded spatial variance and was not this norm. Tensor components and
region-side averaging retain their existing authority; original point-stress recovery
and solver quadrature are unchanged.

Tet4 uses the shared positive four-point rule, exact for the squared linear
interpolant. Tet10 uses the shared five-point tensor Duffy rule: squared quadratic
recovery times a cubic isoparametric Jacobian determinant has degree at most seven.
The estimator streams points with fixed cell scratch, checks cancellation, and rejects
nonfinite norms. This rule evaluates recovery only; it does not change equilibrium.

Independent closed moments check an affine Tet4 and both affine and curved Tet10
maps. For `X=ξ, Y=η, Z=ζ(1+aξ)` and discrepancy `ξ²`, the squared error is
`1/210+a/336`, and the constant reference norm squared is `1/6+a/24`.
The two-cell discontinuity reference is corrected from the invalid squared-mean
proxy `0.5` to `sqrt(4/15)`, retaining its `1e-12` tolerance and adding per-cell
squared errors `8/45` and `4/45`. This is a correction to the declared metric's
unit-test oracle; the original cantilever's `0.9` and `0.25` gates are unchanged.
Verification for this draft is recorded in the evidence ledger; no qualification is
inferred from analytic integration tests.

## Canonical region occurrence identity

Region-side recovery uses borrowed full `PersistentEntityId` keys: entity kind,
source topology ID and ordered assembly path. Source IDs alone can recur in separate
part occurrences and must not identify a common material side. The same value in
separate region declarations still shares recovery; equality is canonical value
equality, not pointer identity. Borrowing removes per-lookup source-string copies.

The regression holds differing constant tensors in cells with shared nodes and the
same source ID but different assembly paths. Each occurrence recovers its own constant
field exactly, whereas the original common-region control must retain a nonzero
recovery discrepancy. Tests cover Tet4/Tet10 and element/point layouts, unchanged
reference norms and exact cold-mesh reevaluation. The old keys
fail both layout tests. This repairs recovery isolation; it is not contact/interface
physics or a resolution of the original cantilever displacement gate.
