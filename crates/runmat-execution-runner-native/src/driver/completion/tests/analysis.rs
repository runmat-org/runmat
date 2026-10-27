//! The placement adapter transports original evidence without rewriting delivery failure.
use super::*;
use crate::analysis_task_pool::AnalysisTaskCompletion;

#[test]
fn analysis_completion_requires_original_commit_and_terminal_inventory() {
    let (mut driver, request) = assigned();
    let live = driver
        .snapshot()
        .task_execution_evidence(request.task_id)
        .unwrap();
    let success = AttemptSuccess::new(&request, Vec::new(), Vec::new(), 1, 0, 0).unwrap();
    assert!(AnalysisTaskCompletion::new(live, Ok(success.clone())).is_err());
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Succeeded {
                result: success.clone(),
            },
        )))
        .unwrap();
    let original = driver
        .snapshot()
        .task_execution_evidence(request.task_id)
        .unwrap();
    let completed = AnalysisTaskCompletion::new(original.clone(), Ok(success.clone())).unwrap();
    assert_eq!(completed.evidence, original);
    assert_eq!(completed.attempt, Some(success.clone()));
    let mut substituted = success;
    let AttemptSuccess::Values { resources, .. } = &mut substituted else {
        panic!("value accounting fixture")
    };
    *resources = None;
    assert!(AnalysisTaskCompletion::new(original.clone(), Ok(substituted)).is_err());
    let delivery =
        AnalysisTaskCompletion::new(original.clone(), Err("original storage failure".into()))
            .unwrap();
    assert_eq!(delivery.evidence, original);
    assert!(delivery.attempt.is_none());
    assert!(matches!(crate::analysis_task_pool::delivered_attempt(
        &delivery.response, delivery.attempt.as_ref(), delivery.failure.as_ref()),
        Err(crate::NativeExecutionError::Protocol(message)) if message == "original storage failure"));
    assert!(
        matches!(delivery.response, runmat_execution_artifact::ProgramExecutionResponse::Failure { message }
        if message == "original storage failure")
    );
}

#[test]
fn analysis_completion_keeps_original_lost_attempt_without_invented_usage() {
    let (mut driver, request) = assigned();
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Lost {
                message: "original worker loss".into(),
            },
        )))
        .unwrap();
    let original = driver
        .snapshot()
        .task_execution_evidence(request.task_id)
        .unwrap();
    let completed =
        AnalysisTaskCompletion::new(original.clone(), Err("original worker loss".into())).unwrap();
    assert_eq!(completed.evidence, original);
    assert!(completed.evidence.resource_aggregate(1).is_err());
    assert_eq!(
        completed.evidence.attempts[0]
            .unverified_retirement
            .as_ref()
            .unwrap()
            .diagnostic
            .as_str(),
        "original worker loss"
    );
    assert!(
        matches!(completed.response, runmat_execution_artifact::ProgramExecutionResponse::Failure { message }
        if message == "original worker loss")
    );
}
