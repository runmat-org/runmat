use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::AccuracyEvaluationResourceDemand;

pub(super) fn build_accuracy_plan(
    input: FeaResourcePlanningInput,
    demand: &AccuracyEvaluationResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    // The transport closure counts each shared object once, so inputs that share the mesh or
    // workload have a closure below their summed encoded payloads; only an empty closure is
    // invalid.
    for (refused, reason) in [
        (demand.criterion_count.0 == 0, "no criteria"),
        (
            demand.evaluator_workspace_bytes_upper_bound.0 == 0,
            "no evaluator workspace",
        ),
        (
            !input.worker.accelerators.is_empty(),
            "an accelerator worker",
        ),
        (
            demand.inputs.transport_closure_bytes.0 == 0,
            "no input transport closure",
        ),
    ] {
        if refused {
            return Err(ResourcePlanningError::Invalid(format!(
                "accuracy requires host-only inputs, criteria and evaluator workspace: {reason}"
            )));
        }
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
