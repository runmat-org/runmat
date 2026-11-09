//! Thermal numerical reservation uses the same host-envelope accounting as other phases.
use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::ThermalSolveResourceDemand;

pub(super) fn build_thermal_plan(
    input: FeaResourcePlanningInput,
    demand: &ThermalSolveResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let invalid = |reason: String| {
        Err(ResourcePlanningError::Invalid(format!(
            "thermal host demand requires numerical storage, samples and valid inputs without accelerators: {reason}"
        )))
    };
    if demand.numerical_reservation_bytes.0 == 0 {
        return invalid("numerical reservation is 0 bytes".into());
    }
    if demand.sample_count.0 == 0 {
        return invalid("sample count is 0".into());
    }
    if !input.worker.accelerators.is_empty() {
        return invalid(format!(
            "{} accelerators requested",
            input.worker.accelerators.len()
        ));
    }
    // Content-addressed repeated chunks can make unique transport smaller than the logical
    // payload (a large mesh repeats region and provenance runs); decoded residency is separate.
    if demand.inputs.transport_closure_bytes.0 == 0 && demand.inputs.encoded_payload_bytes.0 > 0 {
        return invalid(format!(
            "input transport closure is empty for an encoded payload of {} B",
            demand.inputs.encoded_payload_bytes.0
        ));
    }
    if demand.inputs.decoded_resident_bytes_admission_budget < demand.inputs.encoded_payload_bytes {
        return invalid(format!(
            "decoded input budget {} B is below the encoded payload {} B",
            demand.inputs.decoded_resident_bytes_admission_budget.0,
            demand.inputs.encoded_payload_bytes.0
        ));
    }
    for (name, output) in [
        ("result", &demand.result_output),
        ("receipt", &demand.receipt_output),
        ("state profiles", &demand.state_profiles_output),
    ] {
        if output.construction_resident_bytes_admission_budget
            < output.transport_closure_bytes_upper_bound
        {
            return invalid(format!(
                "{name} output construction budget {} B is below its transport closure {} B",
                output.construction_resident_bytes_admission_budget.0,
                output.transport_closure_bytes_upper_bound.0
            ));
        }
    }
    let outputs = [
        &demand.result_output,
        &demand.receipt_output,
        &demand.state_profiles_output,
    ];
    let field_validation = match &demand.field_source {
        Some(source)
            if source.identity != Digest::sha256([])
                && source.encoded_payload_bytes.0 > 0
                && source.validation_workspace_bytes.0 > 0
                && source.encoded_payload_bytes <= demand.inputs.encoded_payload_bytes =>
        {
            source.validation_workspace_bytes
        }
        Some(_) => {
            return Err(ResourcePlanningError::Invalid(
                "thermal field-source resource closure is invalid".into(),
            ))
        }
        None => ByteCount(0),
    };
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
        ByteCount(add(
            demand.numerical_reservation_bytes.0,
            field_validation.0,
            "thermal numerical and source-validation workspace",
        )?),
        retained_transport,
        &outputs,
    );
    accounting.accuracy_demand.samples = demand.sample_count;
    build_host_phase_resource_plan(input, accounting)
}
