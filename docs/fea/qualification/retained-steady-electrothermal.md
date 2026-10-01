# Retained steady Joule heating into canonical thermal assembly

## Supported experimental envelope

One-way, same-canonical-mesh steady electrical conduction into steady CPU thermal
conduction with constant coefficients. This numerical composition is experimental,
not a production formulation or a governed whole-study execution claim.

The electrical result is retained and authenticated against its original workload,
producer, exact result identity, canonical element ordering and mesh. Its element
Joule density (W/m³) becomes a shared element-scalar TopologyField signal with
Imported provenance. No electrical solve runs during transfer or cold import.
Temperature initial fields are not reused as heat-source inputs.

Thermal shape-function integration applies each canonical element density over
its actual physical volume. Region selection retains global row ordering. Source
result, transfer and target workload identities remain distinct; the target
workload includes the transferred source content. Foreign meshes, altered rows,
values, powers and result identities are rejected.

## Independent verification

The unit slab reference uses conductivity σ=3 S/m, ΔV=2 V and length 1 m.
Thus q=σ(ΔV/L)²=12 W/m³ and the unit-volume total power is 12 W.
With thermal conductivity k=1 W/(m K), end temperatures 300 K and insulated
remaining faces, the analytic temperature is T(x)=300+6x(1−x) K.

Tests compare terminal electrical power, retained Joule power, integrated thermal
source and prescribed-boundary outflow. They check the Tet10 quadratic patch,
Tet4 cell-interior refinement, voltage-sign and gauge invariance, and retained
transfer canonical roundtrips after dropping the original electrical result.
Malformed or unsupported sources and production admission fail closed.

## Remaining completion gates

This does not implement temperature-dependent conductivity feedback, harmonic
electromagnetic loss, cross-mesh transfer, transient coupling, or coupled native
worker scheduling/publication. Complete Stage 7 still requires a documented
iteration policy and electrical, thermal, field-change and energy closure,
authored terminal selection, independent wider references, accuracy reports,
restart/execution/resource evidence and release qualification.

Resource reservations are analytical requested-heap bounds, not measured RSS.
Cold reopening is not numerical restart. Existing graph/synthesized-temperature
electrothermal proxies provide no qualification evidence for this formulation.
