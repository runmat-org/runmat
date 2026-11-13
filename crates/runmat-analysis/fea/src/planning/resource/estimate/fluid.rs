//! Genuine monolithic CPU FP64 phases, using shared transport/host accounting.
use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::{FluidFieldRecoveryResourceDemand, FluidSteadySolveResourceDemand};

pub(super) fn solve(
    input: FeaResourcePlanningInput,
    demand: &FluidSteadySolveResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let p = &demand.projection;
    let receipt = crate::execution::fluid_receipt_encoding_profile()
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    if p.numerical.cells == 0
        || p.numerical.faces == 0
        || p.numerical.sparse_entries == 0
        || p.numerical.linear_iterations_upper_bound == 0
        || p.numerical.operator_work_upper_bound == 0
        || p.phase_resident_bytes_upper_bound < ByteCount(p.numerical.resident_bytes_upper_bound)
        || p.phase_resident_bytes_upper_bound < p.source_preparation_resident_bytes
        || p.numerical.verification_resident_bytes_upper_bound
            > p.numerical.resident_bytes_upper_bound
        || p.verification_phase_resident_bytes_upper_bound > p.phase_resident_bytes_upper_bound
        || p.verification_phase_resident_bytes_upper_bound < p.source_preparation_resident_bytes
        || demand.inputs.encoded_payload_bytes != p.canonical_input_bytes
        || demand.inputs.decoded_resident_bytes_admission_budget < p.actual_decode_resident_bytes
        || demand.result_output.encoded_payload_bytes_upper_bound
            != p.result_payload_bytes_upper_bound
        || demand
            .result_output
            .construction_resident_bytes_admission_budget
            < p.result_materialization_resident_bytes
        || demand.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::FLUID_RECEIPT_MAX_ENCODED_BYTES
        || demand
            .receipt_output
            .construction_resident_bytes_admission_budget
            .0
            < receipt.materialization_resident_bytes_estimate
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "steady fluid source/numerical/wire host demand is inconsistent".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "fluid retained input transport",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            p.topology_digest,
            &demand.inputs,
            p.phase_resident_bytes_upper_bound,
            retained,
            &[&demand.result_output, &demand.receipt_output],
        ),
    )
}

pub(super) fn recovery(
    input: FeaResourcePlanningInput,
    demand: &FluidFieldRecoveryResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.construction_workspace_bytes.0 == 0
        || demand.source_validation_workspace_bytes.0 == 0
        || demand.actual_decode_resident_bytes.0 == 0
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.actual_decode_resident_bytes
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "fluid native recovery needs complete original source/host workspace".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "fluid recovery retained input transport",
    )?);
    let workspace = ByteCount(add(
        demand.construction_workspace_bytes.0,
        demand.source_validation_workspace_bytes.0,
        "fluid native recovery workspace",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            workspace,
            retained,
            &[&demand.output],
        ),
    )
}

pub(super) fn transient(
    input: FeaResourcePlanningInput,
    demand: &crate::planning::resource::FluidTransientBlockResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let p = &demand.projection;
    if p.start_interval >= p.end_interval
        || p.charged_operator_work == 0
        || p.phase_resident_bytes.0 == 0
        || demand.inputs.encoded_payload_bytes != p.canonical_input_bytes
        || demand.inputs.decoded_resident_bytes_admission_budget < p.actual_decode_resident_bytes
        || demand.output.encoded_payload_bytes_upper_bound != p.checkpoint_encoded_bytes
        || demand.output.construction_resident_bytes_admission_budget < p.checkpoint_resident_bytes
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "fluid transient original host demand differs".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "fluid transient input transport",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            p.topology_digest,
            &demand.inputs,
            p.phase_resident_bytes,
            retained,
            &[&demand.output],
        ),
    )
}
