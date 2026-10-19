# Canonical contact surface geometry — qualification boundary

This foundation selects authored primary and secondary face occurrences from the shared
analysis projection and evaluates original Tri3/Tri6 geometry. It does not implement
normal/friction enforcement, nonlinear equilibrium, complete contact history, restart,
transfer, accuracy receipts or publication. Existing nonlinear admission still rejects
contact. Stage 6 remains open.

## Geometry and search

Canonical targets use the existing persistent-entity matcher. Retained face-index
inventories sort by stable face identity. Empty, overlapping, conformal and conflicting
contact-side selections reject. The interface and projection are borrowed; no second
surface identity or meshing authority is created. Current nodal positions are explicit
and follow the canonical node inventory. Outward orientation follows original incidence;
world-space normals come from current tangents, preserving proper rigid rotations.
Snapshots validate surface maps, not volume deformation or physical state lineage.

The narrow-phase provider evaluates the original shared boundary shape functions and
analytic derivatives. A planar triangle minimizes over its interior and three edges.
A quadratic subpatch uses its chord triangle and exact edge-midpoint samples. Its
quadratic Bernstein control offset from that chord bounds the spatial discrepancy in
real arithmetic; four-way subdivision reduces that offset. Depth-first traversal stops
only after all retained branches meet the distance target, or returns typed exhaustion.
The result retains the selected barycentric coordinate, original shape weights, current
position/tangents/normal, Euclidean distance, signed normal offset and work count.
The normal offset alone is not penetration or complementarity evidence.

Floating arithmetic uses a scale-based resolution guard. The returned distance error
is a numerical convergence envelope, not a rigorously outward interval certificate,
a guarantee of unique projection, or a bound on position/normal error. Whole-surface
broad phase, tied minima and normal ambiguity, derivative consistency across search
features and contact-parameter/mesh convergence remain necessary before contact use
can be qualified. Curved geometry is never replaced by its corner triangle.

Algorithm references: [point/triangle constrained distance](https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf)
and [Bezier patch convex-hull bounds](https://cs.nyu.edu/media/publications/Morse-thesis.pdf).
These support the geometric construction; repository tests establish the scoped
implementation evidence and do not confer general contact qualification.

## Resources and independent evidence

Controls explicitly declare distance tolerance, maximum visited patches and maximum
depth (at most 32). A depth-first four-child frontier needs at most `1+3*depth` slots.
The component exposes requested frontier heap and selected-face inventory heap before
construction. These exclude source coexistence, whole-task admission, stack, allocator
overhead, retained physical history and RSS. Work/depth exhaustion and cancellation
return no successful candidate. Requested-layout instrumentation checks success,
exhaustion and cancellation; this is not measured process RSS.

Independent references cover triangle interior/edges/vertices, a quadratic graph's
closed position/tangents, a strictly convex known normal projection, proper rigid
rotation/translation and scale, folded/invalid geometry, low precision and explicit
work/depth exhaustion. Both affine element orders exercise canonical current face
snapshots and target binding. Full operator and physics evidence remains outstanding.
Verification receipts and exact candidate scope are recorded in the progress ledger.
