//! Negative fixtures for the sealed original scheduler history.
#[cfg(feature = "occt-native")]
use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
#[cfg(feature = "occt-native")]
use runmat_execution::resource::ResourceAccounting;

pub fn host_requirement(
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) -> runmat_execution::host::ExecutionHostRequirement {
    request
        .artifact
        .execution_host_requirement(
            &request.recipe,
            std::collections::BTreeSet::from([
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            ]),
        )
        .unwrap()
}

pub fn worker_host_inventory() -> runmat_execution::host::ExecutionHostInventory {
    runmat_core::RunMatSession::with_options(false, false)
        .unwrap()
        .execution_host_inventory(runmat_execution::security::ExecutionTrustTier::CustomerTrusted)
        .unwrap()
}

pub fn winning_accounting_mut(
    execution: &mut runmat_execution_runner::TaskExecutionEvidence,
) -> &mut runmat_execution_runner::AttemptResourceAccounting {
    let runmat_execution_runner::AttemptSuccess::Values {
        resources: Some(resources),
        ..
    } = &mut execution
        .task
        .committed
        .as_mut()
        .expect("original commit")
        .result
    else {
        panic!("original analysis value accounting");
    };
    resources
}

#[cfg(feature = "occt-native")]
pub fn assert_foreign_allocation_cannot_be_sealed(execution: &AnalysisPhaseExecutionEvidence) {
    let mut original = execution.task_evidence().clone();
    let accounting = winning_accounting_mut(&mut original);
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
