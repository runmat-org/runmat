//! Typed scheduler cause and physical termination are independent of delivery durability.
use super::*;
use crate::analysis_task_pool::{delivered_attempt, AnalysisTaskCompletion};
use crate::{NativeExecutionError, NativeTaskQuiescence};
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, TaskCancellationCause,
};

fn cancelled(
    termination: Option<AttemptExecutionTermination>,
    local_owner_retired: bool,
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
    let mut retirement = report.clone();
    retirement.report = AttemptReport::Retired;
    driver.handle(DriverCommand::BackendReport(report)).unwrap();
    if local_owner_retired {
        // This is the trusted local process-owner port, not a remote report.
        // Operation outcome and a ProcessExited observation are not this proof.
        driver
            .handle(DriverCommand::BackendReport(retirement))
            .unwrap();
    }
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
    for (termination, local_owner_retired, expected_quiescence) in [
        (Some(AttemptExecutionTermination::NotStarted), true, true),
        (
            Some(AttemptExecutionTermination::ProcessExited {
                code: Some(137),
                success: false,
            }),
            true,
            true,
        ),
        (Some(AttemptExecutionTermination::NotStarted), false, false),
        (
            Some(AttemptExecutionTermination::ProcessExited {
                code: Some(137),
                success: false,
            }),
            false,
            false,
        ),
        (Some(AttemptExecutionTermination::Unconfirmed), false, false),
        (None, false, false),
    ] {
        let completion = cancelled(
            termination,
            local_owner_retired,
            NativeCheckpointEvidence::Durable,
        );
        let original = completion.execution_evidence().unwrap().clone();
        assert_eq!(
            original.task.state,
            if local_owner_retired {
                runmat_execution::state::TaskState::Cancelled
            } else {
                runmat_execution::state::TaskState::Indeterminate
            }
        );
        assert_eq!(original.attempts[0].terminal.is_some(), local_owner_retired);
        assert!(original.resource_aggregate(1).is_err());
        if !local_owner_retired {
            assert!(original.attempts[0]
                .unverified_retirement
                .as_ref()
                .unwrap()
                .retired_at_driver_millis
                .is_none());
        }
        assert!(
            matches!(completion.delivery_result(), Err(NativeExecutionError::TaskCancelled {
            task_id, cancellation, quiescence
        }) if task_id == original.task.id()
            && cancellation == original.task.cancellation.unwrap()
            && matches!(cancellation.cause, TaskCancellationCause::WallTime { .. })
            && quiescence == if expected_quiescence { NativeTaskQuiescence::Confirmed } else { NativeTaskQuiescence::Unconfirmed })
        );
        let mut phase = AnalysisTaskCompletion::from_native(completion).unwrap();
        assert_eq!(phase.evidence, original);
        assert!(
            matches!(delivered_attempt(&phase.response, phase.attempt.as_ref(), phase.failure.as_ref()),
            Err(NativeExecutionError::TaskCancelled { quiescence, .. }) if quiescence == if expected_quiescence { NativeTaskQuiescence::Confirmed } else { NativeTaskQuiescence::Unconfirmed })
        );
        let failure = phase.failure.as_mut().unwrap();
        failure.observe_remote_control_acknowledgement(true);
        assert!(
            matches!(failure.error(), NativeExecutionError::TaskCancelled { quiescence, .. }
            if quiescence == if expected_quiescence { NativeTaskQuiescence::Confirmed } else { NativeTaskQuiescence::Unconfirmed })
        );
        failure.observe_remote_control_acknowledgement(false);
        failure.observe_remote_control_acknowledgement(true);
        assert!(matches!(
            failure.error(),
            NativeExecutionError::TaskCancelled {
                quiescence: NativeTaskQuiescence::Unconfirmed,
                ..
            }
        ));
    }
}

#[test]
fn indeterminate_worker_loss_without_original_cancellation_remains_protocol_failure() {
    let (mut driver, request) = assigned();
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Lost {
                message: "original worker transport lost".into(),
            },
        )))
        .unwrap();
    let original = from_snapshot(
        &driver.snapshot(),
        &request,
        Err(TransferFailure::WorkerLost(
            "original worker transport lost".into(),
        )),
    )
    .unwrap();
    let completion = TaskCompletion::new();
    completion.complete(original, NativeCheckpointEvidence::Durable);
    let completion = completion.try_native_completion().unwrap();
    let evidence = completion.execution_evidence().unwrap();
    assert_eq!(
        evidence.task.state,
        runmat_execution::state::TaskState::Indeterminate
    );
    assert!(evidence.task.cancellation.is_none());
    assert!(matches!(
        completion.delivery_result(),
        Err(NativeExecutionError::Protocol(_))
    ));
    let phase = AnalysisTaskCompletion::from_native(completion).unwrap();
    assert!(matches!(
        delivered_attempt(
            &phase.response,
            phase.attempt.as_ref(),
            phase.failure.as_ref()
        ),
        Err(NativeExecutionError::Protocol(_))
    ));
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
            true,
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
    )
    .unwrap();
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
