use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::ResultPublicationResourceDemand;

pub(super) fn build_result_publication_plan(
    input: FeaResourcePlanningInput,
    demand: &ResultPublicationResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let validation = sum(
        &[
            demand.inputs.decoded_resident_bytes_admission_budget.0,
            demand.accounting_interval_workspace_bytes.0,
            demand.nested_plan_workspace_bytes.0,
            demand.observation_task_index_workspace_bytes.0,
            demand.observation_execution_workspace_bytes.0,
        ],
        "result publication validation workspace",
    )?;
    let construction = add(
        demand.receipt_materialization_bytes.0,
        demand.output.transport_closure_bytes_upper_bound.0,
        "result publication construction",
    )?;
    if demand.receipt_encoded_bytes_upper_bound.0 == 0
        || demand.receipt_materialization_bytes < demand.receipt_encoded_bytes_upper_bound
        || demand.accounting_interval_workspace_bytes.0 == 0
        || demand.nested_plan_workspace_bytes.0 == 0
        || demand.source_validation_workspace_bytes.0 != validation
        // Content-addressed repeated chunks can make unique transport smaller than the payload.
        || demand.inputs.transport_closure_bytes.0 == 0
        || demand.inputs.decoded_resident_bytes_admission_budget
            < demand.inputs.encoded_payload_bytes
        || demand.output.encoded_payload_bytes_upper_bound
            != demand.receipt_encoded_bytes_upper_bound
        || demand.output.transport_closure_bytes_upper_bound
            < demand.receipt_encoded_bytes_upper_bound
        || demand.output.construction_resident_bytes_admission_budget.0 < construction
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "result publication requires bound receipt and source validation envelopes on host"
                .into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "result publication retained transport",
    )?);
    let workspace = ByteCount(add(
        validation,
        demand.receipt_materialization_bytes.0,
        "result publication validation and owned receipt",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.source_result.digest(),
            &demand.inputs,
            workspace,
            retained,
            &[&demand.output],
        ),
    )
}
