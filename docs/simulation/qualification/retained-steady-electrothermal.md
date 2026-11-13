# Retained steady Joule heating into canonical thermal assembly

## Supported envelope

One-way, same-canonical-mesh steady electrical conduction into steady CPU thermal
conduction with constant coefficients. This numerical composition is
not a qualified formulation or a governed whole-study execution claim.

The electrical result is retained and authenticated against its original workload,
producer, exact result identity, canonical element ordering and mesh. Its element
Joule density (W/m³) becomes a shared element-scalar TopologyField signal with
Retained provenance. Ordinary external imports remain distinct: a retained heat
field requires the authenticated original electrical source binding, even if its
thermal workload is rehashed. No electrical solve runs during transfer or cold import.
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
Malformed or unsupported sources fail closed.

## Governed native consumer and cold completion

The original electrical workload/result, expected result identity, unbound thermal
intent and transfer travel as a chunked FieldSourceBinding object through the
existing SolveThermal task and SelectedStep thermal coordinator. Workers rederive
the resource plan before scientific binding and reject missing, foreign, malformed
or substituted source closures. Source decode/transport and validation workspace
are separately reserved; result import verifies the original task inputs and power.

Actual child-process tests execute the thermal consumer for Tet4 and Tet10, publish
named states and field observations, and satisfy their requested algebraic and
conservation accuracy criteria. The original electrical solve is a fixture producer,
not a claimed native coupled electrical task. Schema3 completion roots retain exact
source and catalog authority. Declared objects are copied to a separate store;
after dropping producer handles, cold imports preserve exact manifests, accuracy,
states and publication and independently check the Tet10 analytic temperature.

The source-retaining local completion API holds a read-only object-store lease,
not the producing worker pool. This prevents the session store from disappearing
before callers persist the complete declared closure.

## Remaining completion gates

This does not implement temperature-dependent conductivity feedback, harmonic
electromagnetic loss, cross-mesh transfer, transient coupling, or coupled native
electrical-plus-thermal whole-study scheduling. Complete Stage 7 still requires a documented
iteration policy and electrical, thermal, field-change and energy closure,
authored terminal selection, independent wider references, accuracy reports,
restart/execution/resource evidence and release qualification.

Resource reservations are analytical requested-heap bounds, not measured RSS.
Cold reopening is not numerical restart. Existing graph/synthesized-temperature
electrothermal proxies provide no qualification evidence for this formulation.
