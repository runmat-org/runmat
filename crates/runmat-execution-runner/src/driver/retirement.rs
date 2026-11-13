//! Logical failure cannot surrender a still-owned physical reservation.
use runmat_execution::state::TaskState;

use crate::pool::WorkerLifecycle;
use crate::task::{AttemptRecord, AttemptRetirementUnverified};
use crate::RunnerResult;

use super::reports::ProviderEvidence;
use super::{Driver, DriverEventKind};

impl Driver {
    pub(super) fn retain_unverified_attempt(
        &mut self,
        attempt: &AttemptRecord,
        evidence: ProviderEvidence,
        diagnostic: String,
        failure: Option<Box<runmat_execution::ProgramRuntimeFailure>>,
        failure_kind: Option<crate::task::AttemptFailureKind>,
    ) -> RunnerResult<()> {
        if diagnostic.is_empty() {
            return Err(crate::RunnerError::Invalid(
                "retirement uncertainty lacks an original diagnostic".into(),
            ));
        }
        let record = self
            .snapshot
            .attempts
            .get_mut(&attempt.request.id)
            .expect("original attempt was validated");
        if let Some(original) = &mut record.unverified_retirement {
            if original.failure.is_none() {
                original.failure = failure;
            }
            if original.failure_kind.is_none() {
                original.failure_kind = failure_kind;
            }
            // Retain one actual observation tuple, never combine accounting
            // from one interval with a later execution/console observation.
            if original.resources.is_none()
                && original.execution.is_none()
                && original.console.is_none()
                && original.console_scopes.is_empty()
                && original.delegated_console_root.is_none()
            {
                original.resources = evidence.resources;
                original.execution = evidence.execution;
                original.console = evidence.console;
                original.console_scopes = evidence.console_scopes;
                original.delegated_console_root = evidence.delegated_console_root;
            }
            return Ok(());
        }
        record.unverified_retirement = Some(Box::new(AttemptRetirementUnverified {
            observed_at_driver_millis: self.snapshot.now_millis,
            diagnostic,
            failure,
            failure_kind,
            resources: evidence.resources,
            execution: evidence.execution,
            console: evidence.console,
            console_scopes: evidence.console_scopes,
            delegated_console_root: evidence.delegated_console_root,
            retired_at_driver_millis: None,
        }));
        self.worker_mut(attempt.request.worker_id)?.lifecycle = WorkerLifecycle::Lost;
        self.emit(DriverEventKind::AttemptLost {
            task_id: attempt.request.task_id,
            attempt_id: attempt.request.id,
        });
        self.transition_task(attempt.request.task_id, TaskState::Indeterminate)?;
        self.task_mut(attempt.request.task_id)?.active_attempt = Some(attempt.request.id);
        Ok(())
    }
}
