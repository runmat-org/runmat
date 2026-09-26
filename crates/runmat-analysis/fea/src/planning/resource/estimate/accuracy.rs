use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::AccuracyEvaluationResourceDemand;

pub(super) fn build_accuracy_plan(
    input: FeaResourcePlanningInput,
    demand: &AccuracyEvaluationResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.criterion_count.0 == 0
        || demand.evaluator_workspace_bytes_upper_bound.0 == 0
        || !input.worker.accelerators.is_empty()
        || demand.inputs.transport_closure_bytes < demand.inputs.encoded_payload_bytes
    {
        return Err(ResourcePlanningError::Invalid(
            "accuracy requires criteria, evaluator workspace and host-only input storage".into(),
        ));
    }
    let retained_transport = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "accuracy retained input transport",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.source_result.digest(),
            &demand.inputs,
            demand.evaluator_workspace_bytes_upper_bound,
            retained_transport,
            &[&demand.output],
        ),
    )
}
