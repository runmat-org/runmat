use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::TaskId;

use crate::port::BackendReport;
use crate::scheduler;
use crate::task::{
    retry_decision, AttemptDisposition, AttemptFailureKind, AttemptRecord, AttemptReport,
    AttemptTerminal, CommitDecision, ResultCommit, RetryCause, RetryDecision,
};
use crate::{RunnerError, RunnerResult};

use super::state::attempt_state_is_terminal;
use super::{Driver, DriverAction, DriverEventKind};

#[derive(Default)]
pub(super) struct ProviderEvidence {
    pub resources: Option<Box<crate::task::AttemptResourceAccounting>>,
    pub execution: Option<Box<crate::task::AttemptExecutionObservation>>,
    pub console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
}

impl Driver {
    pub(super) fn apply_backend_report(
        &mut self,
        mut report: BackendReport,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        if let Some(reason) = self.backend_report_rejection_reason(&report) {
            self.discard_backend_report(report, reason, actions);
            return Ok(());
        }
        let attempt = self.snapshot.attempts[&report.attempt_id].clone();
        let resources = report
            .unsuccessful_resources
            .clone()
            .or_else(|| match &report.report {
                AttemptReport::Succeeded { result } => result
                    .resources()
                    .filter(|resources| resources.validate_against(&attempt.request).is_ok())
                    .cloned()
                    .map(Box::new),
                _ => None,
            });
        let execution = report.execution.clone();
        let console = report.console.take();
        let cancelling = attempt.cancellation_requested_at.is_some();
        let unresolved = attempt
            .unverified_retirement
            .as_ref()
            .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none());
        if (cancelling || unresolved)
            && matches!(&report.report,
            AttemptReport::Succeeded { result } if result.resources().is_some_and(|resources|
                resources.validate_against(&attempt.request).is_err()
                    || report.execution.as_ref().is_some_and(|execution|
                        resources.resources.usage.wall_millis != execution.wall_millis)))
        {
            self.discard_backend_report(report,
                "unresolved success accounting contradicts its original attempt or execution interval", actions);
            return Ok(());
        }
        if unresolved && matches!(report.report, AttemptReport::Started) {
            self.discard_backend_report(report, "unresolved attempt cannot restart", actions);
            return Ok(());
        }
        if matches!(&report.report, AttemptReport::Retired) && !cancelling && !unresolved {
            self.discard_backend_report(
                report,
                "retirement certificate lacks an active cancellation or uncertainty",
                actions,
            );
            return Ok(());
        }
        if (cancelling || unresolved)
            && !matches!(
                &report.report,
                AttemptReport::Started
                    | AttemptReport::Retired
                    | AttemptReport::RetirementUnverified { .. }
            )
        {
            // No operation outcome or transport loss proves cancellation
            // retirement. Retain the original attempt and all its capacity.
            if let AttemptReport::Succeeded { result } = &report.report {
                actions.push(DriverAction::GarbageCollectResults {
                    task_id: report.task_id,
                    objects: result.result_objects().to_vec(),
                });
            }
            let (diagnostic, failure, failure_kind) =
                match std::mem::replace(&mut report.report, AttemptReport::Started) {
                    AttemptReport::RuntimeFailed { failure } => {
                        (failure.message.clone(), Some(Box::new(failure)), None)
                    }
                    AttemptReport::Failed { kind, message } => (message, None, Some(kind)),
                    AttemptReport::Lost { message } => (message, None, None),
                    _ => (
                        "worker retirement was not certified after cancellation or uncertainty"
                            .into(),
                        None,
                        None,
                    ),
                };
            self.retain_unverified_attempt(
                &attempt,
                ProviderEvidence {
                    resources,
                    execution,
                    console,
                },
                diagnostic,
                failure,
                failure_kind,
            )?;
            return Ok(());
        }
        if let Some(contract) = &attempt.request.task.outputs.owned_work {
            let invalid_failure = matches!(
                &report.report,
                AttemptReport::RuntimeFailed { failure } if failure.validate().is_err()
            );
            if invalid_failure || contract.validate_json_wire(&report.report).is_err() {
                self.fail_attempt(
                    &attempt,
                    AttemptFailureKind::Rejected,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    Some("owned builtin report differs from its output contract".into()),
                )?;
                return Ok(());
            }
        }
        match report.report {
            AttemptReport::Started => self.start_attempt(&report)?,
            AttemptReport::Succeeded { result } => {
                if (attempt.request.task.console.is_some()
                    && console.as_ref().is_none_or(|console| {
                        console.completion()
                            != runmat_execution::console::ConsoleCompletion::Complete
                    }))
                    || result.validate_against(&attempt.request).is_err()
                    || execution.as_ref().is_some_and(|execution| {
                        !matches!(
                            execution.termination,
                            crate::task::AttemptExecutionTermination::ProcessExited {
                                success: true,
                                ..
                            }
                        ) || result.resources().is_some_and(|resources| {
                            resources.resources.usage.wall_millis != execution.wall_millis
                        })
                    })
                {
                    self.fail_attempt(
                        &attempt,
                        AttemptFailureKind::Rejected,
                        ProviderEvidence {
                            resources: result
                                .resources()
                                .filter(|resources| {
                                    resources.validate_against(&attempt.request).is_ok()
                                })
                                .cloned()
                                .map(Box::new),
                            execution,
                            console,
                        },
                        Some(
                            "worker success contradicts admitted accounting or execution evidence"
                                .into(),
                        ),
                    )?;
                    return Ok(());
                }
                if result
                    .validate_for_portable_transport(&attempt.request.task)
                    .is_err()
                {
                    self.fail_attempt(
                        &attempt,
                        AttemptFailureKind::Rejected,
                        ProviderEvidence {
                            resources: None,
                            execution,
                            console,
                        },
                        Some("worker success violates its portable transport contract".into()),
                    )?;
                    return Ok(());
                }
                let uncommitted_objects = result.result_objects().to_vec();
                self.finish_attempt(
                    &attempt,
                    AttemptDisposition::Succeeded,
                    ProviderEvidence {
                        resources: result.resources().cloned().map(Box::new),
                        execution,
                        console,
                    },
                    None,
                )?;
                let decision = self.commit_result(&attempt, result)?;
                if let CommitDecision::Accepted(commit) = decision {
                    self.emit(DriverEventKind::ResultCommitted {
                        task_id: report.task_id,
                        attempt_id: report.attempt_id,
                        commit_id: commit.id,
                    });
                    self.activate_dependents(report.task_id)?;
                } else {
                    actions.push(DriverAction::GarbageCollectResults {
                        task_id: report.task_id,
                        objects: uncommitted_objects,
                    });
                    self.emit(DriverEventKind::ReportDiscarded {
                        task_id: report.task_id,
                        attempt_id: report.attempt_id,
                        reason: "result lost its task or driver fence".into(),
                    });
                }
            }
            AttemptReport::Failed { kind, message } => {
                self.fail_attempt(
                    &attempt,
                    kind,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    Some(message),
                )?;
            }
            AttemptReport::RuntimeFailed { .. } => {
                self.fail_attempt(
                    &attempt,
                    AttemptFailureKind::Execution,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    None,
                )?;
            }
            AttemptReport::Lost { message } => {
                self.retain_unverified_attempt(
                    &attempt,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    message,
                    None,
                    None,
                )?;
            }
            AttemptReport::RetirementUnverified { message } => {
                self.retain_unverified_attempt(
                    &attempt,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    message,
                    None,
                    None,
                )?;
            }
            AttemptReport::Retired => {
                self.finish_attempt(
                    &attempt,
                    if unresolved {
                        AttemptDisposition::Lost
                    } else {
                        AttemptDisposition::Cancelled
                    },
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    None,
                )?;
                if unresolved {
                    // Physical proof releases capacity, not an indeterminate
                    // operation's result or permission to replay it.
                    self.task_mut(report.task_id)?.active_attempt = None;
                } else {
                    self.emit(DriverEventKind::AttemptCancelled {
                        task_id: report.task_id,
                        attempt_id: report.attempt_id,
                    });
                    self.transition_task(report.task_id, TaskState::Cancelled)?;
                }
            }
            AttemptReport::Cancelled => {
                self.finish_attempt(
                    &attempt,
                    AttemptDisposition::Cancelled,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                    },
                    None,
                )?;
                self.emit(DriverEventKind::AttemptCancelled {
                    task_id: report.task_id,
                    attempt_id: report.attempt_id,
                });
                if !self.task_is_terminal(report.task_id)? {
                    let requested_at_driver_millis = self.snapshot.now_millis;
                    self.task_mut(report.task_id)?.cancellation = Some(crate::TaskCancellation {
                        cause: crate::TaskCancellationCause::Provider {
                            attempt_id: attempt.request.id,
                        },
                        requested_at_driver_millis,
                    });
                    self.transition_task(report.task_id, TaskState::Cancelled)?;
                }
            }
        }
        Ok(())
    }

    /// Advisory validation under the caller's scheduler lock. Hosts must retain
    /// that lock through a related authority transition; this is not a lease.
    pub fn accepts_backend_report(&self, report: &BackendReport) -> bool {
        self.backend_report_rejection_reason(report).is_none()
    }

    pub(super) fn backend_report_rejection_reason(
        &self,
        report: &BackendReport,
    ) -> Option<&'static str> {
        let Some(attempt) = self.snapshot.attempts.get(&report.attempt_id) else {
            return Some("unknown attempt");
        };
        let original_retirement = matches!(report.report, AttemptReport::Retired)
            && attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none());
        if report.driver_fence != attempt.request.driver_fence
            || (report.driver_fence != self.snapshot.driver_fence && !original_retirement)
        {
            return Some("stale driver fence");
        }
        if report.task_id != attempt.request.task_id
            || report.worker_id != attempt.request.worker_id
        {
            return Some("attempt identity mismatch");
        }
        if attempt_state_is_terminal(attempt.state) {
            return Some("attempt is already terminal");
        }
        if matches!(&report.report, AttemptReport::RuntimeFailed { failure } if failure.validate().is_err())
        {
            return Some("invalid original runtime failure");
        }
        if let Some(console) = &report.console {
            if matches!(report.report, AttemptReport::Started)
                || console.validate().is_err()
                || Some(console.limits()) != attempt.request.task.console
            {
                return Some("invalid original console evidence");
            }
        }
        if let Some(execution) = &report.execution {
            if matches!(report.report, AttemptReport::Started)
                || execution.validate_against(&attempt.request).is_err()
            {
                return Some("invalid provider execution observation");
            }
        }
        if let Some(resources) = &report.unsuccessful_resources {
            if matches!(
                report.report,
                AttemptReport::Started | AttemptReport::Succeeded { .. }
            ) || resources.validate_against(&attempt.request).is_err()
                || report.execution.as_ref().is_some_and(|execution| {
                    resources.resources.usage.wall_millis != execution.wall_millis
                })
            {
                return Some("invalid unsuccessful accounting");
            }
        }
        None
    }

    fn start_attempt(&mut self, report: &BackendReport) -> RunnerResult<()> {
        self.snapshot
            .attempts
            .get_mut(&report.attempt_id)
            .expect("attempt was validated")
            .state = AttemptState::Running;
        let task = self.task_mut(report.task_id)?;
        if task.active_attempt == Some(report.attempt_id) {
            task.state = TaskState::Running;
        }
        self.emit(DriverEventKind::AttemptStarted {
            task_id: report.task_id,
            attempt_id: report.attempt_id,
        });
        Ok(())
    }

    fn fail_attempt(
        &mut self,
        attempt: &AttemptRecord,
        kind: AttemptFailureKind,
        evidence: ProviderEvidence,
        diagnostic: Option<String>,
    ) -> RunnerResult<()> {
        let task_id = attempt.request.task_id;
        let attempt_id = attempt.request.id;
        self.finish_attempt(
            attempt,
            AttemptDisposition::Failed { kind },
            evidence,
            diagnostic,
        )?;
        self.emit(DriverEventKind::AttemptFailed {
            task_id,
            attempt_id,
            kind,
        });
        let cause = match kind {
            AttemptFailureKind::Infrastructure => RetryCause::Infrastructure,
            AttemptFailureKind::Execution => RetryCause::Execution,
            AttemptFailureKind::Rejected => RetryCause::Rejected,
        };
        self.finish_unsuccessful(task_id, cause)
    }

    pub(super) fn finish_attempt(
        &mut self,
        attempt: &AttemptRecord,
        disposition: AttemptDisposition,
        evidence: ProviderEvidence,
        diagnostic: Option<String>,
    ) -> RunnerResult<()> {
        let terminal = AttemptTerminal {
            disposition,
            closed_at_driver_millis: self.snapshot.now_millis,
            resources: evidence.resources,
            execution: evidence.execution,
            console: evidence.console,
            diagnostic,
        };
        terminal
            .validate_against(attempt)
            .map_err(RunnerError::Invalid)?;
        let record = self
            .snapshot
            .attempts
            .get_mut(&attempt.request.id)
            .expect("attempt was validated");
        record.state = disposition.state();
        record.terminal = Some(terminal);
        if let Some(uncertainty) = &mut record.unverified_retirement {
            uncertainty.retired_at_driver_millis = Some(self.snapshot.now_millis);
        }
        let pool = self.pool_mut(attempt.request.task.pool_id)?;
        let worker = pool
            .workers
            .get_mut(&attempt.request.worker_id)
            .ok_or(RunnerError::UnknownWorker(attempt.request.worker_id))?;
        scheduler::release(
            &mut worker.allocated,
            &attempt.request.task.resources,
            &attempt.request.resource_assignment,
        );
        worker.active_attempts = worker.active_attempts.saturating_sub(1);
        scheduler::release_scalar(&mut pool.allocated, &attempt.request.task.resources);
        pool.active_attempts = pool.active_attempts.saturating_sub(1);
        Ok(())
    }

    fn commit_result(
        &mut self,
        attempt: &AttemptRecord,
        result: crate::task::AttemptSuccess,
    ) -> RunnerResult<CommitDecision> {
        if attempt.request.driver_fence != self.snapshot.driver_fence {
            return Ok(CommitDecision::StaleFence);
        }
        let task = self.task_mut(attempt.request.task_id)?;
        if task.committed.is_some() {
            return Ok(CommitDecision::Duplicate);
        }
        if task.active_attempt != Some(attempt.request.id) || task.state == TaskState::Cancelled {
            return Ok(CommitDecision::StaleAttempt);
        }
        let commit =
            ResultCommit::from_success(attempt.request.id, attempt.request.driver_fence, result);
        task.committed = Some(commit.clone());
        task.active_attempt = None;
        task.state = TaskState::Succeeded;
        self.remove_deadline(attempt.request.task_id);
        Ok(CommitDecision::Accepted(commit))
    }

    fn finish_unsuccessful(&mut self, task_id: TaskId, cause: RetryCause) -> RunnerResult<()> {
        if self.task_is_terminal(task_id)? {
            return Ok(());
        }
        let enqueue_sequence = self.snapshot.next_event_sequence;
        let task = self.task_mut(task_id)?;
        task.active_attempt = None;
        let decision = retry_decision(task.submission.request.retry, task.attempt_count, cause);
        match decision {
            RetryDecision::Retry => {
                task.state = TaskState::Ready;
                task.enqueued_sequence = enqueue_sequence;
                let record = task.clone();
                self.enqueue(&record);
                self.emit(DriverEventKind::TaskStateChanged {
                    task_id,
                    state: TaskState::Ready,
                });
            }
            RetryDecision::Fail => self.transition_task(task_id, TaskState::Failed)?,
            RetryDecision::Indeterminate => {
                self.transition_task(task_id, TaskState::Indeterminate)?
            }
        }
        Ok(())
    }

    pub(super) fn discard_backend_report(
        &mut self,
        report: BackendReport,
        reason: &str,
        actions: &mut Vec<DriverAction>,
    ) {
        if let AttemptReport::Succeeded { result } = report.report {
            actions.push(DriverAction::GarbageCollectResults {
                task_id: report.task_id,
                objects: result.result_objects().to_vec(),
            });
        }
        self.emit(DriverEventKind::ReportDiscarded {
            task_id: report.task_id,
            attempt_id: report.attempt_id,
            reason: reason.into(),
        });
    }
}
