//! Preserve original scheduler causes at the native-to-meshing domain boundary.
use super::*;
use crate::NativeExecutionError;
use runmat_execution_runner::TaskCancellationCause;
use runmat_meshing_core::{
    MeshingDiagnosticEntry, MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory,
    MeshingStageKind, MESHING_FAILURE_SCHEMA_VERSION,
};

pub(super) fn stage_failure(
    error: NativeExecutionError,
    stage: MeshingStageKind,
) -> MeshingStageExecutionError {
    if let NativeExecutionError::TaskCancelled { cancellation, .. } = &error {
        let category = match cancellation.cause {
            TaskCancellationCause::Deadline | TaskCancellationCause::WallTime { .. } => {
                MeshingFailureCategory::TimeBudgetExceeded
            }
            _ => MeshingFailureCategory::Cancelled,
        };
        return MeshingStageExecutionError::Failure(Box::new(MeshingFailure {
            schema_version: MESHING_FAILURE_SCHEMA_VERSION,
            category,
            stage,
            operation: stage.operation(),
            entity_ids: Vec::new(),
            witnesses: Vec::new(),
            request_values: Vec::new(),
            achieved_values: vec![MeshingDiagnosticEntry {
                name: "native_task_cancellation".into(),
                value: MeshingDiagnosticValue::Text(error.to_string()),
                unit: None,
            }],
            accounting_error: None,
            remediation: "inspect the original task time envelope and stage performance".into(),
        }));
    }
    // Domain worker failures retain their authenticated meshing message. Delivery
    // or transport failures must not be mistaken for a worker physics verdict.
    match error {
        NativeExecutionError::Protocol(message) => {
            match MeshingHostResponse::failure_from_program_message(&message) {
                Some(failure) => MeshingStageExecutionError::Failure(Box::new(failure)),
                None => MeshingStageExecutionError::new(message),
            }
        }
        other => MeshingStageExecutionError::new(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{identity::AttemptId, TaskId};
    use runmat_execution_runner::TaskCancellation;

    #[test]
    fn scheduler_time_expiry_keeps_cause_and_quiescence_in_meshing_failure() {
        for cause in [
            TaskCancellationCause::Deadline,
            TaskCancellationCause::WallTime {
                attempt_id: AttemptId::derive(&[b"test"]),
            },
            TaskCancellationCause::Provider {
                attempt_id: AttemptId::derive(&[b"test"]),
            },
        ] {
            let original = NativeExecutionError::TaskCancelled {
                task_id: TaskId::derive(&[b"test"]),
                cancellation: TaskCancellation {
                    cause,
                    requested_at_driver_millis: 1000,
                },
                quiescence: crate::NativeTaskQuiescence::Confirmed,
            };
            let diagnostic = original.to_string();
            let MeshingStageExecutionError::Failure(failure) =
                stage_failure(original, MeshingStageKind::Tetrahedralization)
            else {
                panic!("typed failure required")
            };
            failure.validate().unwrap();
            assert_eq!(
                failure.category,
                if matches!(cause, TaskCancellationCause::Provider { .. }) {
                    MeshingFailureCategory::Cancelled
                } else {
                    MeshingFailureCategory::TimeBudgetExceeded
                }
            );
            assert_eq!(
                failure.achieved_values[0].value,
                MeshingDiagnosticValue::Text(diagnostic)
            );
        }
        assert!(matches!(
            stage_failure(
                NativeExecutionError::Protocol("execution was cancelled".into()),
                MeshingStageKind::Tetrahedralization
            ),
            MeshingStageExecutionError::Message(_)
        ));
    }
}
