# Physical tetrahedral edge-space prerequisite

## Scope and decision

This is a shared topology/numerical prerequisite for Stage 7 magnetostatics and
harmonic Maxwell, **not a qualified formulation**. The
existing reduced pipeline EM operator is not replaced or qualified by this work.
No native solve, gauge, magnetic source, recovery, restart, or release gate is
claimed. In particular, no scalar electrostatic solve is renamed magnetostatics.

`TetrahedralEdgeSpace` owns deterministic global edge incidence and lowest-order
first-kind Nédélec basis evaluation on canonical Tet4/Tet10 geometry. Edges are
oriented and numbered by ordered persistent node identities, not node-array
positions or numerical node labels. Local corner-edge orientation is retained.
The space is bound to the projection digest and element identity/connectivity.

The six local functions are `N_ab = λ_a grad λ_b − λ_b grad λ_a`; physical curls
are `2 grad λ_a × grad λ_b`. Gradients use the actual isoparametric Jacobian,
including curved Tet10 midside geometry. This implements covariant Piola values
and contravariant Piola curls. Tet10 **geometry order does not imply quadratic
edge approximation**: there are still six corner-edge circulation DOFs.

Shared physical quadrature supplies mass, curl-curl, constant-vector-load and
scalar-gradient-coupling moments. Physics owns material coefficients and sources;
topology does not add heuristic diagonal stabilization, penalties, gauge fixes,
or geometric denominator floors. The discrete scalar gradient is the canonical
oriented endpoint difference, not a recovered field or a second solve.

## Independent evidence

- Affine Tet4 and Tet10 integrals use analytic barycentric moments:
  `∫λ_i=V/4`, `∫λ_i²=V/10`, `∫λ_iλ_j=V/20` for `i≠j`. These independently
  determine every mass, curl-curl, constant-source and gradient-coupling entry.
- The curved map `F(r,s,t)=(r,s,t+0.6rs)` has determinant one. Explicit analytic
  covariant/contravariant transformations check values and curls pointwise.
- All edge line integrals reproduce the Kronecker circulation DOFs on affine and
  curved geometry. Shared-face tangential traces agree per global edge.
- Arbitrary corner scalar data reconstruct its piecewise scalar gradient and have
  zero curl; global edge identities and element moments survive node-storage
  permutation. Invalid cardinality, nonfinite data, foreign element identity,
  duplicate identities, repeated/missing corners and inverted geometry fail closed.
- Shared 15-point volume integrals agree with a separate 8×8×8 tensor rule on the
  named determinant-one polynomial curved map. This is **not** general curved
  rational-integrand quadrature convergence evidence. Affine forms are exact;
  general curved admission remains gated on explicit quadrature convergence.

The basis/mapping contract follows [DefElement's first-kind Nédélec element](https://defelement.org/elements/nedelec1.html).
The physical magnetic weak form and source-compatibility requirements are
described in [MFEM's magnetostatic notes](https://mfem.org/tesla-notes/).
The tests use analytic references, not another RunMat solver as an oracle.

## Remaining vertical chain

The [physical magnetic numerical baseline](magnetostatic-volume-current.md) now
implements distinct intent, element-owned permeability, explicit volume current,
full-exterior tangential grounding and a weak Coulomb gauge using this space.
Its independent square-current series, field/energy convergence, work/flux closure
and signed permeability scaling tests pass within the documented affine envelope.
It does not admit a native magnetic workload or close a formulation gate.
Next finish resource/peak accounting,
retained fields/state, source-bound cold validation and actual authored/native
execution, then decide the formulation gate. Remove superseded reduced paths at
the actual cutover. Maxwell loss, complex sources, ports and boundary forms require
their own evidence. No broad Stage 7 checkbox is closed by this prerequisite.
