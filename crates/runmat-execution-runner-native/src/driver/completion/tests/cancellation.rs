//! Typed scheduler cause and physical termination are independent of delivery durability.
use super::*;
use crate::analysis_task_pool::{delivered_attempt, AnalysisTaskCompletion};
use crate::{NativeExecutionError, NativeTaskQuiescence};
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, TaskCancellationCause,
};

fn cancelled(
    termination: Option<AttemptExecutionTermination>,
    checkpoint: NativeCheckpointEvidence,
) -> crate::NativeProgramCompletion {
    let (mut driver, request) = assigned();
    driver
        .handle(DriverCommand::Tick { now_millis: 1_000 })
        .unwrap();
    let mut report = BackendReport::for_request(&request, AttemptReport::Cancelled);
    report.execution = termination.map(|termination| {
        Box::new(AttemptExecutionObservation {
            attempt_id: request.id,
            worker_id: request.worker_id,
            wall_millis: 1,
            termination,
            local_ipc_sent_bytes: Some(0),
            local_ipc_received_bytes: Some(0),
        })
    });
    driver.handle(DriverCommand::BackendReport(report)).unwrap();
    let original = from_snapshot(
        &driver.snapshot(),
        &request,
        Err(TransferFailure::Execution(
            "an arbitrary cancellation diagnostic".into(),
        )),
    )
    .unwrap();
    let completion = TaskCompletion::new();
    completion.complete(original, checkpoint);
    completion.try_native_completion().unwrap()
}

#[test]
fn native_cancellation_uses_original_cause_and_requires_physical_evidence_for_quiescence() {
    for (termination, expected_quiescence) in [
        (Some(AttemptExecutionTermination::NotStarted), true),
        (
            Some(AttemptExecutionTermination::ProcessExited {
                code: Some(137),
                success: false,
            }),
            true,
        ),
        (Some(AttemptExecutionTermination::Unconfirmed), false),
        (None, false),
    ] {
        let completion = cancelled(termination, NativeCheckpointEvidence::Durable);
        let original = completion.execution_evidence().unwrap().clone();
        assert!(
            matches!(completion.delivery_result(), Err(NativeExecutionError::TaskCancelled {
            task_id, cancellation, quiescence
        }) if task_id == original.task.id()
            && cancellation == original.task.cancellation.unwrap()
            && matches!(cancellation.cause, TaskCancellationCause::WallTime { .. })
            && quiescence == if expected_quiescence { NativeTaskQuiescence::Confirmed } else { NativeTaskQuiescence::Unconfirmed })
        );
        let phase = AnalysisTaskCompletion::from_native(completion).unwrap();
        assert_eq!(phase.evidence, original);
        assert!(
            matches!(delivered_attempt(&phase.response, phase.attempt.as_ref(), phase.failure.as_ref()),
            Err(NativeExecutionError::TaskCancelled { quiescence, .. }) if quiescence == if expected_quiescence { NativeTaskQuiescence::Confirmed } else { NativeTaskQuiescence::Unconfirmed })
        );
    }
}

#[test]
fn atomic_checkpoint_failure_keeps_priority_over_an_original_confirmed_cancellation() {
    for checkpoint in [
        NativeCheckpointEvidence::NotPublished {
            diagnostic: "write failed".into(),
        },
        NativeCheckpointEvidence::DurabilityIndeterminate {
            diagnostic: "directory flush failed".into(),
        },
    ] {
        let completion = cancelled(
            Some(AttemptExecutionTermination::NotStarted),
            checkpoint.clone(),
        );
        let original = completion.execution_evidence().unwrap().clone();
        assert_eq!(completion.checkpoint_evidence(), &checkpoint);
        assert!(
            matches!(completion.delivery_result(), Err(NativeExecutionError::CheckpointDelivery { evidence, diagnostic })
            if *evidence == checkpoint && checkpoint.delivery_error().as_deref() == Some(diagnostic.as_str()))
        );
        let phase = AnalysisTaskCompletion::from_native(completion).unwrap();
        assert_eq!(phase.evidence, original);
        assert!(phase.attempt.is_none());
        assert!(
            matches!(delivered_attempt(&phase.response, phase.attempt.as_ref(), phase.failure.as_ref()),
            Err(NativeExecutionError::CheckpointDelivery { evidence, .. }) if *evidence == checkpoint)
        );
    }
}

#[test]
fn a_failed_task_diagnostic_cannot_impersonate_typed_cancellation() {
    let (mut driver, request) = assigned();
    let message = "execution was cancelled";
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Failed {
                kind: runmat_execution_runner::AttemptFailureKind::Execution,
                message: message.into(),
            },
        )))
        .unwrap();
    let original = from_snapshot(
        &driver.snapshot(),
        &request,
        Err(TransferFailure::Execution(message.into())),
    ).unwrap();
    let completion = TaskCompletion::new();
    completion.complete(original, NativeCheckpointEvidence::Durable);
    let completion = completion.try_native_completion().unwrap();
    assert!(completion
        .execution_evidence()
        .unwrap()
        .task
        .cancellation
        .is_none());
    assert!(
        matches!(completion.delivery_result(), Err(NativeExecutionError::Protocol(diagnostic)) if diagnostic == message)
    );
}
