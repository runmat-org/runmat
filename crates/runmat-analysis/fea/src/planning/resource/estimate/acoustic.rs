//! Additive host accounting; scientific qualification is not a resource estimate.
use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::AcousticHarmonicResourceDemand;

pub(super) fn build_acoustic_plan(
    input: FeaResourcePlanningInput,
    demand: &AcousticHarmonicResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.node_count.0 == 0
        || demand.element_count.0 == 0
        || demand.block_dimension_upper_bound.0
            != mul(demand.node_count.0, 2, "acoustic real block")?
        || demand.krylov_restart_upper_bound.0 == 0
        || demand.krylov_restart_upper_bound.0 > demand.block_dimension_upper_bound.0
        || demand.numerical_reservation_bytes.0 == 0
        || !input.worker.accelerators.is_empty()
        // Content-addressed repeated chunks can make unique transport smaller than logical
        // payload. The bridge/worker derive its exact closure; decoded residency is separate.
        || demand.inputs.transport_closure_bytes.0 == 0
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || demand
            .result_output
            .construction_resident_bytes_admission_budget
            < demand.result_output.transport_closure_bytes_upper_bound
    {
        return Err(ResourcePlanningError::Invalid(
            "acoustic CPU dimensions or prepared byte facts are inconsistent".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "acoustic retained input transport",
    )?);
    let outputs = [&demand.result_output];
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            demand.numerical_reservation_bytes,
            retained,
            &outputs,
        ),
    )
}
