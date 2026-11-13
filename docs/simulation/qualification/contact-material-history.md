# Original contact material history and smooth friction contribution

This component retains and differentiates original point history on a smooth, held-primary-
face branch. It is not enabled in nonlinear assembly and is not a converged body state,
global checkpoint, native continuation witness, contact benchmark or qualification.

## Original reference inventory and scope

Shared boundary quadrature now retains its original barycentric coordinates and reference
weight directly. Numerical rules, physical fields and acceptance tolerances are unchanged.
The shared point layout is larger; resource estimates use its actual type size and resource
model revision 98. Bounded three/six-node boundary coordinates now use stack arrays instead
of a per-face heap vector. The shared curved-moment scratch estimator is reused.

ContactReferenceInventory compiles canonical secondary points once, with explicit point
budget and exact ordered point payload. It owns its canonical primary face-index list
and borrows only the original projection/interface, so it can outlive temporary surface
selection without a self-referential owner. That list is included in its requested storage. A binding selects an original point from this
inventory; it does not regenerate quadrature for each history point. The complete authored
interface, projection, law, original secondary face/ordinal/coordinates/weights and local
derivative controls bind point scope through shared canonical encoding. The law must match
the actual interface friction model. Requested inventory/construction layout excludes
borrowed source, stack, allocator bookkeeping, binding codec temporaries and measured RSS;
it is not whole-procedure admission.

## State and validation

The opaque carrier retains schema, scope, ordinal, original primary face/material
coordinates, every primary and secondary nodal coordinate, material frame, relative material
vector and the complete original point-law history. Trial updates do not mutate accepted
input. No state field defaults or projection from element averages are used. Ordinals advance
by one with checked overflow; an external expected ordinal/configuration is required for
cold revalidation. Exact present geometry/frame consistency, point-law invariants and scope
are checked. Deserialization alone does not authorize continuation, attest past history or
prove convergence. Initial point history evaluates initial normal penetration; it is not a
stress-free, equilibrated body state.

The current branch preserves the primary face. Face switching, ambiguous minima, edge/corner
features and supported transfer require explicit future transitions; no friction history is
silently transferred to another face. The primary closest material coordinates may move
inside the held face, and their original accepted coordinates remain in the history.

## Objective coordinates, residual and tangent

At a current stationary projection, tangential coordinates compare the SAME accepted
primary material location and secondary material point in the accepted/current material
frames. Fresh closest-point separation is not frictional slip. The point law retains its
original traction/net slip, accumulated sliding and work under this declared corotational
material-frame transport. The current closest point is committed only in the new trial
carrier; the accepted carrier remains intact.

Topology owns analytic first/mixed implicit projection derivatives, including curvature,
material-coordinate motion and normal motion. The same local metric and shape authority
serve both derivative orders. Shared algebra owns scale-aware normalized-vector first/mixed
derivatives; structural frames reuse that normalization authority.

With objective coordinates z=[s1,s2,g], point generalized traction t and local tangent C,
nodal residual is w0 * B^T*t. Its directional derivative includes BOTH
w0 * B^T*C*delta_z and w0 * sum_i(t_i*delta_grad(z_i)). Material-frame/projection terms
therefore contribute nodal forces and tangent, rather than merely rotating the point
matrix. Nominal traction and fixed reference-area measure have the same boundary as the
normal contribution. Secondary query forces scatter using the original secondary shape
values. Trial state, original nodal residuals and directional tangents allocate no heap.

## Evidence and remaining work

Independent normalized-vector references and an independent scalar parabola check mixed
normal/projection/gap derivatives. Both element orders check cold state and tamper/ordinal/
foreign-point rejection, deterministic immutable trials, proper rigid motion, zero rigid
slip/work, resultant/moment balance with friction, full stick/slip residual derivatives at
two step sizes, requested-layout inventory bounds and cancellation.

The canonical tetrahedron point fixtures test local assembly algebra and history, including
initial penetration. They are not independent two-body continuum contact benchmarks. Global
patch/Hertz/sliding, mesh/load/penalty sensitivity, feature transitions, original-state
checkpoint integration, equality conflicts, full sparse/resource admission, native cold
continuation, mandatory accuracy, receipts/publication and release evidence remain open.
Existing nonlinear admission still rejects contact. Exact verification belongs to the ledger.

## Stationary projection and global sparse scatter

`BoundaryStationaryPolicy` declares separate stationarity, feature/metric and bounded
Newton/backtrack controls. The shared surface and canonical surface inventory refine
an existing numerical closest candidate without rerunning its subdivision search.
Each accepted refinement decreases distance and stationarity residual; the original
numerical distance envelope is retained. Canonical outward side follows reference
incidence and the normal follows current geometry. Exhaustion, a nonsmooth feature,
an unresolved/nonpositive distance metric or cancellation returns no refined result.
This is an interior local projection, not an interval proof or within-face uniqueness
certificate. Conservative inter-face separation remains separate from stationarity.

`ContactTangentSupport` reserves all translational couplings for every admitted
primary/secondary face pair, independent of the current active set. Pair and coordinate
caps are checked before coordinate/sparse allocation. It uses shared `SparsePattern`
construction from an owned coordinate inventory: in-place sort/deduplication and exact
CSR buffers, without guessed tree-node storage. Requested construction storage includes
the input coordinate inventory and returned CSR buffers. No symmetry/SPD assumption,
whole-task admission or measured RSS follows from this support.

`ContactPointTrial::assemble_into` scatters original primary forces and secondary
reference-shape forces into the caller's explicit translational rows. Tangent columns
apply the same transformation on both sides, including shared-node contributions,
through the shared planned sparse assembler. It streams fixed scratch without retaining
a dense point tangent. Invalid maps reject before scatter. Cancelled/failed accumulators
must be discarded; assembler finalization checks support and finite values. Finite checks
visit touched residual entries rather than scanning the whole body vector for every point.

Both element orders are checked against original residual differences at two step sizes,
including the sparse matrix action, exact requested payload, zero-heap scatter, cap
rejection before allocation and inventory lifetime. These are assembly tests. Global
Newton integration, deduplicated original body/contact state, governed checkpoint/restart,
feature switching/transfer, equality conflicts, mandatory accuracy, native publication and
independent two-body continuum contact benchmarks remain required. The public nonlinear
contact route remains disabled.


The complete prepared multi-point assembly and compact original global configuration
are described in [contact-global-operator.md](contact-global-operator.md). That layer
reuses this original point law and derivative authority; it does not enable body contact.
