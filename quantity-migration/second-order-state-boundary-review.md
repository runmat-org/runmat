# Private Newmark state boundary review

Status: proposed narrow raw-kernel disposition; pending the retained-basis equality guards and combined semantic qualification. This is not an approval of arbitrary structural vectors or public response fields.

`solve/transient/state.rs::SecondOrderState` is a private numerical integration record containing three equally sized finite buffers. It does not implement Serde, does not define units, and is not a durable or transport representation. `newmark.rs::{prepare_average_acceleration_step,NewmarkPredictor}` operates on these canonical buffers and checked canonical time-step coefficients. Giving each private coefficient a new unit enum would create a competing authority rather than improve the public contract.

The physical authority must remain outside this numerical record:

- `AssemblySummary::structural_solve_layout` supplies the full layout or actual retained affine reduction. Reduced buffers must never be labeled with the unreduced layout merely because an elimination flag has been cleared.
- `effective_solve/system.rs::{effective_system,reduced_effective_system}` carries that exact layout/transform into `StructuralLinearSystemView`; its physical input validation precedes numerical execution.
- `solve/transient/mod.rs::require_stable_system` and the corresponding snapshot/reconstruction guards must compare the actual retained reduction as well as full DOF layout. Equal buffer lengths or equal reduced dimensions are insufficient evidence that coordinate bases agree.
- Modal entry/exit uses `modal_block.rs::{project_initial_conditions,project_representable_state,coordinate_state}` and the admitted modal basis. Full physical continuation and numerical modal coordinates must be reconstructed/projected through that basis, not relabeled.
- `advance.rs::{initialize,resume}` and `segment.rs::into_canonical_segment_parts` are the reviewed state ingress/egress. Durable continuation identity, full/reduced scope, and typed physical sample fields remain authoritative; a raw Newmark record must never be serialized as a substitute for those contracts.

The proposed disposition applies only to the private buffers and arithmetic in the Newmark kernel and these checked conversion seams. It does not apply to raw public frequency/PSD inputs, physical outputs, arbitrary `Vec<f64>` fields elsewhere, or a future public constructor accepting anonymous coordinate vectors.

Qualification required before approval: prescribed support-motion direct/reduced cases, modal reconstruction/restart, cold continuation, nonfinite/mismatched state rejection, and an equal-size different-reduction substitution regression. Root is implementing the retained-basis comparison guards and owns their focused regression. Until these pass, the residual remains explicitly open. If the record is later exported or serialized, it must gain the shared field/basis header at that boundary; this review cannot be reused to exempt a new public representation.
