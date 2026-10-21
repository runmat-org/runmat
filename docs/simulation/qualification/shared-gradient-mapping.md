# Shared canonical volume-gradient correction

The topology volume quadrature forms a Jacobian with reference axes in rows and
physical axes in columns. Physical shape gradients are `J^-1 grad_reference N`.
The inverse's off-diagonal cofactor previously used `m02*m20` where `m02*m21`
is required. The correction changes exactly that index, not the determinant,
quadrature rule, shape values, positions or integration weights.

Independent rotated acoustic tests exposed the defect: an affine world-z field
reconstructed `[-sqrt(2),0,1]` rather than `[0,0,1]`, and acoustic operator covariance
failed. Both pass after correction. Direct tests now require left and right matrix
products to reproduce identity for nonsymmetric positive-determinant matrices.
Shared Tet4/Tet10 oblique geometry tests reproduce all three world-coordinate affine
gradients at every quadrature point. Topology regressions pass 34/34.

Gradient-sensitive consumers include acoustic stiffness/recovery, thermal diffusion
and flux, electrical fields, thermoelastic loads, prestress geometric stiffness,
and coupled structural recovery. Main structural element stiffness uses separate,
already correct inverses. Shape-only mass/source/interpolation is not directly changed.
Retained thermal-to-structural tests pass 11/11 with unchanged continuum metrics;
prestress integration passes 6/6. Broad FEA regressions pass 1266/1266, explicitly filtering
four transient draft tests, the failed radiation test and two known proof-limit tests.
This is not unrestricted green; further verification is recorded in the project ledger.

This correction does not retrospectively change stored numerical artifacts or qualify
arbitrary geometry. Prior aligned tests could mask the defect. In particular the new
local-radiation envelope still fails a separate coarse Tet10 velocity-convergence gate;
passing rotation checks does not promote that formulation or Stage 8.

## Exact affine Jacobians in the structural Tet10 element

The classical structural Tet10 element now derives its Jacobian from the three
corner differences when every midside coordinate is exactly the arithmetic midpoint
of its corners. The existing exact predicate is owned by element geometry and shared
with stiffness integration. There is no curvature tolerance: even a dyadic midside
perturbation of `2^-40` remains on the full isoparametric path.

This avoids cancellation when evaluating the derivatives of quadratic shape
functions for an exactly affine map. It affects the shared classical Tet10 geometry
used by stiffness, mass and recovery, separately from the canonical topology inverse
correction above. Original curved-element and independent physical regressions remain
mandatory; this is not a change to the integration rule or accuracy criteria.

Independent analytic simplex-derivative tests pass at dyadic and canonical non-dyadic
recovery points, three scales and three origins,
including a dyadic translation of `2^40`. The original heterogeneous Tet10 constitutive
comparison passes with its unchanged `1e-10` criterion. A source-projection rounding
change exposed that comparison's sensitivity; the exact affine Jacobian resolves it
without changing the reference. The full isolated numerical candidate finished with 1,663 passing and 14 failing
tests. Original scientific and resource gates remain open; no whole-suite pass or
qualification promotion is claimed. That run predates the solver-ownership refactor.

The cantilever refinement test contains two independent gates. Its stress-recovery
estimator decreases from about 373 to 270 to 196, satisfying the original estimator
criterion. Its peak-displacement changes still fail the original 0.25 ratio criterion
(actual about 0.272). This remains an open displacement-convergence gate.
