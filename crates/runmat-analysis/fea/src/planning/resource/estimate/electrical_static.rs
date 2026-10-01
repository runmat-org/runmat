use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::ElectricalStaticResourceDemand;

pub(super) fn build_electrical_static_plan(
    input: FeaResourcePlanningInput,
    demand: &ElectricalStaticResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.node_count.0 == 0
        || demand.element_count.0 == 0
        || demand.operator_nonzeros_upper_bound.0 < demand.node_count.0
        || demand.numerical_reservation_bytes.0 == 0
        || demand.solver_admission.dimension != demand.node_count
        || demand.solver_admission.factor_nonzero_upper_bound < demand.node_count
        || demand.solver_admission.update_count_upper_bound.0 == 0
        || demand.solver_admission.resident_bytes_upper_bound.0 == 0
        || demand.numerical_reservation_bytes < demand.solver_admission.resident_bytes_upper_bound
        || demand.inputs.transport_closure_bytes.0 == 0
        || demand.state_profiles_resident_bytes.0 == 0
        || demand.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES
        || demand
            .receipt_output
            .construction_resident_bytes_admission_budget
            < demand.receipt_output.transport_closure_bytes_upper_bound
        || demand
            .state_profiles_output
            .construction_resident_bytes_admission_budget
            < demand
                .state_profiles_output
                .transport_closure_bytes_upper_bound
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || demand
            .result_output
            .construction_resident_bytes_admission_budget
            < demand.result_output.transport_closure_bytes_upper_bound
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "electrical static CPU dimensions or prepared facts are inconsistent".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "electrical retained input transport",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            ByteCount(add(
                demand.numerical_reservation_bytes.0,
                demand.state_profiles_resident_bytes.0,
                "electrical retained publication profiles",
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
