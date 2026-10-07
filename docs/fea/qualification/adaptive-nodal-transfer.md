# Adaptive nodal field transfer

## Supported numerical operation

Meshing owns physical point location and source-element interpolation. Interior
CDT refinement/coarsening maps now cover every new target node, including new
Tet10 midsides. A Tet4 source uses its four barycentric weights; an affine Tet10
source uses all ten Lagrange nodal values. The reference basis has one owner in
`runmat-meshing-core::interpolation`, also used by FEA element and volume-quadrature
shape evaluation. Production quadrature points and derivatives are unchanged.

Exact orientation predicates admit containment. Floating determinant ratios
construct the barycentric coordinates, with finite/nonzero and partition checks;
unresolvable coordinates reject. Source elements are searched in stable-identity
order. A containing element on a shared face is chosen deterministically. Source
Tet10 geometry must have straight midpoint nodes within the explicit floating
roundoff bound; curved geometry rejects with `UnsupportedGeometry`. This is an
affine interpolation envelope, not an inverse curved-map implementation.

One work counter and cancellation policy cover affinity checks, new-node point
location, and new-element centroid location. The resource ceiling cannot reset
between these phases. This bounds that search component; it does not establish
whole-task storage admission or measured resident memory.

## Portable map and consumer

Transfer schema/domain V2 names `QuadraticLagrangeInterpolation` explicitly.
Weights may be negative: the four centroid corner weights are `-1/8`, while six
edge weights are `1/4`. Inventories have at most ten nonzero, finite, canonical
sources, weights in the admitted `[-1/8, 1]` roundoff envelope and sum one.
Artifact-bound validation requires the method to retain the source element order.
Legacy schema/domain, inappropriate methods and malformed weights reject.
Decoding and artifact binding establish the portable contract, not an independent
mathematical proof that arbitrary imported weights reconstruct a physical point.

FEA applies that original source inventory to each field component. Stable source
identities copy their original values. Endpoint averaging is available only for a
linear source and is reported as `LinearEdgeInterpolation`; a quadratic source
without a required map rejects instead of losing its higher-order content.

Error measurement has its own FEA module and uses scaled `hypot` reductions.
Finite fields at `1e-200` and `1e200` retain their nonzero absolute error and
relative error; a norm or ratio that cannot be represented finitely rejects.
Naive squaring previously underflowed to false zero or overflowed to infinity.

## Evidence and remaining gates

Independent tests evaluate every constant, linear and quadratic Cartesian mode on
actual CDT-refined target nodes, including centroid and midside nodes. They also
check negative weights, cold map/lineage binding, unsupported curved source,
cancellation and the exact combined work ceiling. FEA tests compare transferred
scalar/vector quadratic fields with their physical-coordinate polynomial values
and reject incomplete maps and linear downgrade. Existing p-elevation/restriction
and Tet4 transfer tests remain mandatory.

This is nodal field transfer. Existing centroid projection of element values does
not authorize transfer of irreversible point history, contact state or temporal
state. The original cantilever target-quantity convergence failure remains open.
Stage 11 still requires validated family estimators, constraint rebuilding,
shared-accuracy stopping, authored adaptive graphs, durable/remote execution and
iteration/result publication. This foundation alone closes none of those gates.
