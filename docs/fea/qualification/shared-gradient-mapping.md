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
