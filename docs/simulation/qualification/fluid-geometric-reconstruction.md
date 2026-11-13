# Prepared CFD geometric reconstruction

Velocity and pressure gradients share a geometry-only least-squares metric owner.
Four-face velocity stencils are derived once from immutable cell/face geometry and
compiled boundary kinds. Each evaluation reads current original cell differences
and prescribed face values. Pressure reconstruction retains its bounded one-sided
cell/outlet inventory; velocity outlets retain the original zero normal derivative.
No matrix fit, diagonal floor, fabricated pressure trace or independent solver is added.

Numerical input identity v4 separates this preparation/arithmetic revision from
prior cold results. Resource model revision 75 charges the retained velocity
stencils in addition to existing construction, Krylov, original-field and recovery
bounds. Requested bytes are not measured allocator usage.

Independent affine gradients are tested on original tetrahedral face authorities,
multiple meshes and rotations. General physical-coordinate stencil tests include
rotation/scale covariance, exact affine gradients, first-order gradient truncation
of an independently defined quadratic field, and rank-deficiency rejection.
These stencil checks do not establish study-level spatial error bounds. Original
Couette, convective and Fourier duct references, balances and cold validation remain
mandatory. No CFD qualification or Stage 9 promotion follows from preparation reuse.
