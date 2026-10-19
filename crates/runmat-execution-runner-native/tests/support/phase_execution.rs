//! Negative fixtures for the sealed original scheduler history.
use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_execution::resource::ResourceAccounting;

pub fn assert_foreign_allocation_cannot_be_sealed(execution: &AnalysisPhaseExecutionEvidence) {
    let mut original = execution.task_evidence().clone();
    let accounting = original
        .task
        .committed
        .as_mut()
        .unwrap()
        .resources
        .as_mut()
        .unwrap();
    let mut allocation = accounting.resources.allocation.clone();
    allocation.memory_bytes += 1;
    accounting.resources = ResourceAccounting::for_reserved_interval(
        allocation,
        accounting.resources.usage.wall_millis,
        accounting.resources.artifact_bytes,
        accounting.resources.usage.egress_bytes,
        accounting.resources.usage.relay_bytes,
    )
    .unwrap();
    // The scalar charge remains internally valid; it is foreign to the immutable
    // task/request/attempt association, which cannot be rewritten in a replay.
    accounting.validate().unwrap();
    let wire = serde_json::json!({
        "task": &original,
        "graph_wall_millis": execution.graph_wall_millis(),
    });
    assert!(serde_json::from_value::<AnalysisPhaseExecutionEvidence>(wire).is_err());
    assert!(AnalysisPhaseExecutionEvidence::new(original, execution.graph_wall_millis()).is_err());
}
