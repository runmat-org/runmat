//! Thermal numerical reservation uses the same host-envelope accounting as other phases.
use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::ThermalSolveResourceDemand;

pub(super) fn build_thermal_plan(
    input: FeaResourcePlanningInput,
    demand: &ThermalSolveResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.numerical_reservation_bytes.0 == 0
        || demand.sample_count.0 == 0
        || !input.worker.accelerators.is_empty()
        || demand.inputs.transport_closure_bytes < demand.inputs.encoded_payload_bytes
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || [&demand.result_output, &demand.receipt_output]
            .iter()
            .any(|output| {
                output.construction_resident_bytes_admission_budget
                    < output.transport_closure_bytes_upper_bound
            })
    {
        return Err(ResourcePlanningError::Invalid("thermal host demand requires numerical storage, samples and valid inputs without accelerators".into()));
    }
    let outputs = [&demand.result_output, &demand.receipt_output];
    // Imported chunk buffers coexist with decoded numerical inputs during execution.
    let retained_transport = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "thermal retained input transport",
    )?);
    let mut accounting = HostPhaseAccounting::host_only(
        demand.topology_digest,
        &demand.inputs,
        demand.numerical_reservation_bytes,
        retained_transport,
        &outputs,
    );
    accounting.accuracy_demand.samples = demand.sample_count;
    build_host_phase_resource_plan(input, accounting)
}
