use super::*;
use crate::FluidStudyCoordinator;

#[test]
fn cold_phase_preserves_known_retry_and_observed_interval_independently_of_leases() {
    let mut study = FluidStudyCoordinator::new(crate::fluid_study::tests::preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![],
        result_objects: vec![],
    };
    let original = phase_execution_with_retry(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &response,
        Some(true),
        29,
    )
    .unwrap();
    original
        .validate_for(&task.host, &task.resource_plan, &response)
        .unwrap();
    let cold: AnalysisPhaseExecutionEvidence =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    cold.validate_for(&task.host, &task.resource_plan, &response)
        .unwrap();
    let graph = cold.resource_graph(&task.resource_plan).unwrap();
    assert_eq!(graph.wall_millis, 29);
    assert_eq!(graph.attempts.len(), 2);
    assert_eq!(
        graph
            .attempts
            .iter()
            .map(|a| a.accounting.resources.usage.wall_millis)
            .sum::<u64>(),
        3
    );
    assert_eq!(
        graph
            .attempts
            .iter()
            .map(|a| a.accounting.resources.artifact_bytes)
            .sum::<u64>(),
        7
    );
    assert_eq!(cold, original);
    let mut omitted = cold.task_evidence().clone();
    omitted.attempts.remove(0);
    assert!(AnalysisPhaseExecutionEvidence::new(omitted, 29).is_err());
    let swapped = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![task.program.request.arguments[0].clone()],
        result_objects: vec![],
    };
    assert!(cold
        .validate_for(&task.host, &task.resource_plan, &swapped)
        .is_err());
}

#[test]
fn unknown_retry_blocks_completion_and_cold_decode_rejects_missing_or_zero_interval() {
    let mut study = FluidStudyCoordinator::new(crate::fluid_study::tests::preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![],
        result_objects: vec![],
    };
    let unknown = phase_execution_with_retry(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &response,
        Some(false),
        29,
    );
    assert!(
        unknown.is_err(),
        "an indeterminate attempt cannot become completed phase evidence"
    );
    let valid = phase_execution(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &response,
    );
    let mut wire = serde_json::to_value(&valid).unwrap();
    wire["graph_wall_millis"] = 0.into();
    assert!(serde_json::from_value::<AnalysisPhaseExecutionEvidence>(wire).is_err());
    let mut wire = serde_json::to_value(valid).unwrap();
    wire.as_object_mut().unwrap().remove("task");
    assert!(serde_json::from_value::<AnalysisPhaseExecutionEvidence>(wire).is_err());
}

#[test]
fn phase_rejects_foreign_host_and_original_callable_tampering() {
    let mut study = FluidStudyCoordinator::new(crate::fluid_study::tests::preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![],
        result_objects: vec![],
    };
    let valid = phase_execution(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &response,
    );
    let mut host = task.host.clone();
    host.artifact_access
        .authorization_scope
        .push_str("/foreign");
    assert!(valid
        .validate_for(&host, &task.resource_plan, &response)
        .is_err());
    let mut original = valid.task_evidence().clone();
    original.task.submission.request.callable.program =
        runmat_execution::ProgramCallable::builtin("__foreign").unwrap();
    assert!(AnalysisPhaseExecutionEvidence::new(original, 1).is_err());
}
