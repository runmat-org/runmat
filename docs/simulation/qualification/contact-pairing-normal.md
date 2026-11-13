# Canonical contact pairing and smooth normal contribution

This slice implements current-configuration face search and a normal penalty contribution
at a stationary interior projection. Nonlinear preparation still rejects interfaces/contact.
It does not qualify a global contact solve, friction assembly, restart or publication.

## Geometry and search authority

The topology owner snapshots original Tri3/Tri6 maps from explicit current canonical nodal
positions. Reference incidence defines the outward side. Face snapshots retain original
stable identities and are sorted deterministically, independently of input index order.
The contact owner selects primary faces through its existing canonical interface inventory;
it delegates geometry and whole-surface search to topology.

For Tri6, the bounding box includes quadratic Bernstein control points, rather than only
nodal positions. A curved map can overshoot its nodal box. Numerical roundoff guards expand
lower-distance comparisons. This is a numerical enclosure, not outward interval proof.
Every selected face is validated when the inventory is constructed. No invalid face is
silently discarded because its bounds appear far from a query.

The query scans bounded inventory in stable order, prunes faces using lower distance bounds,
and shares one explicit total narrow-patch budget. Pruned faces remain in the distance
separation test through their lower bounds. Face count, total work, per-face work/depth,
maximum distance and separation margin are explicit. Unmatched, near-range uncertainty,
unseparated face distances, invalid geometry, cancellation and exhaustion return errors,
never an accepted partial pairing. Coincident or shared-edge face ties are conservatively
rejected; feature-level reconciliation and normal ambiguity remain required work.

A separated face candidate does not establish uniqueness inside a curved face, or guarantee
accuracy of the closest material coordinates and normal merely from distance accuracy.
It is not a physical contact state. Whole-procedure admission and mesh/configuration/history
lineage must bind this immutable geometric snapshot before it can become state authority.
Requested inventory and scratch bytes exclude source coexistence, allocator overhead and RSS.

## Stationary projection differential

At an interior stationary point, write `d = query - X(u,v)` and tangents `t_i = X_,i`.
The local squared-distance metric is `G_ij = t_i dot t_j - d dot X_,ij`. The implicit
variation solves `G delta_xi = t_i dot delta_r + d dot delta_t_i_fixed`. Geometry reuses
first derivatives and differentiates that same shared polynomial owner for second derivatives.
The returned variation includes material-coordinate motion, normal motion, original shape
changes, signed-gap gradient and its directional change for query and every primary node.
The oriented snapshot consistently applies its side to signed quantities.

Explicit stationarity tolerance, interior-feature margin and minimum metric eigenvalue ratio
bound this local operation. Boundary features, nonstationary coordinates and degenerate/focal
metrics reject. Positive local metric is not proof of global uniqueness. A numerical closest
candidate must be refined and checked before requesting this smooth derivative; feature
switches require their own nonsmooth treatment. No fixed-normal surrogate is substituted.

## Normal physical contribution

The structural owner evaluates the existing point law's history-independent normal part.
With a fixed **reference-area** quadrature weight `w0`, potential is
`w0 * kn/2 * min(g,0)^2`. Forces are `w0 * kn*g * grad(g)` in active contact.
The complete directional derivative of this potential is
`w0 * [kn*delta_g*grad(g) + kn*g*delta_grad(g)]`.
This includes projection and normal derivatives. The reported traction is nominal per
reference area; it must not be presented as spatial pressure per deformed area.

The kernel returns original primary nodal forces and query force, not element averages.
It holds reference weight and secondary material interpolation fixed. It does not implement
friction, current-area weight derivatives, global scatter/sparse support, feature switching,
complete material/frame state, equality conflicts or global acceptance. These remain explicit
requirements of the full contact vertical.

Evidence covers independent parabola projection/curvature derivatives, gap-Hessian symmetry,
original force/energy/action-reaction/moment references on both orders, residual finite
differences at two step sizes, full normal tangent symmetry, proper rotations, rigid motions,
cancellation and requested-layout bounds. Global patch/Hertz/sliding, load-step/mesh/penalty
convergence, nonlinear execution, measured resource envelopes, history/restart, mandatory
accuracy and publication remain open. Exact candidate verification is recorded in the ledger.
