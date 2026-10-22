use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::AcousticResponseResourceDemand;

pub(super) fn build(
    input: FeaResourcePlanningInput,
    demand: &AcousticResponseResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.node_count.0 == 0
        || demand.element_count.0 == 0
        || demand.sample_count.0 == 0
        || demand.operator_support_upper_bound.0 == 0
        || demand.numerical_reservation_bytes.0 == 0
        || demand.state_profiles_resident_bytes.0 == 0
        || demand.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::ACOUSTIC_RESPONSE_RECEIPT_MAX_ENCODED_BYTES
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || [
            &demand.result_output,
            &demand.receipt_output,
            &demand.state_profiles_output,
        ]
        .iter()
        .any(|o| {
            o.construction_resident_bytes_admission_budget < o.transport_closure_bytes_upper_bound
        })
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "acoustic response sparse or prepared host facts are inconsistent".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "acoustic response retained input transport",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            ByteCount(add(
                demand.numerical_reservation_bytes.0,
                demand.state_profiles_resident_bytes.0,
                "acoustic response profiles resident",
            )?),
            retained,
            &[
                &demand.result_output,
                &demand.receipt_output,
                &demand.state_profiles_output,
            ],
        ),
    )
}
