//! Additive publication residency using shared canonical materialization estimates.
use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::SemanticStatePublicationResourceDemand;

pub(super) fn build_state_publication_plan(
    input: FeaResourcePlanningInput,
    demand: &SemanticStatePublicationResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let state_bytes = demand.state_encoded_bytes_upper_bound.0;
    let materialization = demand.state_materialization_resident_bytes_estimate.0;
    let construction = add(
        materialization,
        demand.output.transport_closure_bytes_upper_bound.0,
        "publication encoding and output transport",
    )?;
    if state_bytes == 0
        || materialization < state_bytes
        || demand.inputs.transport_closure_bytes < demand.inputs.encoded_payload_bytes
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || demand.source_validation_workspace_bytes
            != demand.inputs.decoded_resident_bytes_admission_budget
        || demand.output.encoded_payload_bytes_upper_bound.0 != state_bytes
        || demand.output.transport_closure_bytes_upper_bound.0 < state_bytes
        || demand.output.construction_resident_bytes_admission_budget.0 < construction
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "publication requires source decoding and source-bound state construction envelopes on host".into(),
        ));
    }
    let retained_transport = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "publication retained input transport",
    )?);
    // Sources remain decoded while constructors clone/move their selected payload. Charge a
    // separate state materialization envelope for that owned state and its validation/content
    // digest construction. Output construction independently covers canonical encoding and
    // retained prepared chunks. Repeated digest/validation passes are sequential, not multiplied
    // by their invocation count. These are codec admission estimates, not heap measurements.
    // Source validation also remains live: response-spectrum verification clones direction
    // contributions; random-response verification clones interval contributions and creates
    // total/compensation/flat vectors plus variance/RMS reconstruction. These are linear copies
    // of arrays already present in the result, not a numerical solve. Reserve an additional
    // shared canonical source materialization envelope, whose two Value-sized representations
    // dominate these scalar copies and their collection metadata. Modal governance adds two
    // scalar vectors and linear eigenspace metadata; thermal/linearization use uniqueness sets.
    let workspace = ByteCount(add(
        materialization,
        demand.source_validation_workspace_bytes.0,
        "publication source validation and owned state",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            workspace,
            retained_transport,
            &[&demand.output],
        ),
    )
}
