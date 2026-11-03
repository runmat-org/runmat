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
    pub console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    pub delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
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
        let console_scopes = std::mem::take(&mut report.console_scopes);
        let delegated_console_root = report.delegated_console_root;
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
                    | AttemptReport::CertifiedNotStarted {}
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
                    console_scopes,
                    delegated_console_root,
                },
                diagnostic,
                failure,
                failure_kind,
            )?;
            return Ok(());
        }
        if let Some(contract) = attempt
            .request
            .task
            .outputs
            .owned_work
            .as_ref()
            .filter(|_| !matches!(report.report, AttemptReport::CertifiedNotStarted {}))
        {
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
                        console_scopes,
                        delegated_console_root,
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
                    && runmat_execution::console::original_root(
                        console.as_deref(),
                        &console_scopes,
                        delegated_console_root,
                    )
                    .is_none_or(|console| {
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
                            console_scopes,
                            delegated_console_root,
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
                            console_scopes,
                            delegated_console_root,
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
                        console_scopes,
                        delegated_console_root,
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
                        console_scopes,
                        delegated_console_root,
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
                        console_scopes,
                        delegated_console_root,
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
                        console_scopes,
                        delegated_console_root,
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
                        console_scopes,
                        delegated_console_root,
                    },
                    message,
                    None,
                    None,
                )?;
            }
            AttemptReport::Retired => {
                if unresolved {
                    let original = attempt
                        .unverified_retirement
                        .as_ref()
                        .expect("unresolved retirement was checked");
                    let evidence = ProviderEvidence {
                        resources: resources.or_else(|| original.resources.clone()),
                        execution: execution.or_else(|| original.execution.clone()),
                        console: console.or_else(|| original.console.clone()),
                        delegated_console_root: delegated_console_root
                            .or(original.delegated_console_root),
                        console_scopes: if console_scopes.is_empty() {
                            original.console_scopes.clone()
                        } else {
                            console_scopes
                        },
                    };
                    let cause = match (original.failure_kind, original.failure.as_ref()) {
                        (Some(AttemptFailureKind::Infrastructure), None) => {
                            Some(RetryCause::Infrastructure)
                        }
                        (Some(AttemptFailureKind::Execution), None) | (None, Some(_)) => {
                            Some(RetryCause::Execution)
                        }
                        (Some(AttemptFailureKind::Rejected), None) => Some(RetryCause::Rejected),
                        (None, None) => Some(RetryCause::Lost),
                        (Some(_), Some(_)) => None,
                    };
                    let complete = evidence.resources.is_some() && cause.is_some();
                    let disposition = if complete {
                        match original.failure_kind {
                            Some(kind) => AttemptDisposition::Failed { kind },
                            None if original.failure.is_some() => AttemptDisposition::Failed {
                                kind: AttemptFailureKind::Execution,
                            },
                            None => AttemptDisposition::Lost,
                        }
                    } else {
                        AttemptDisposition::Lost
                    };
                    self.finish_attempt(
                        &attempt,
                        disposition,
                        evidence,
                        Some(original.diagnostic.clone()),
                    )?;
                    let cancel_or_expire = self.task_record(report.task_id)?.cancellation.is_some()
                        || self
                            .task_record(report.task_id)?
                            .submission
                            .request
                            .deadline_unix_millis
                            .is_some_and(|at| at <= self.snapshot.now_millis);
                    if complete || cancel_or_expire {
                        self.resolve_certified_original(
                            report.task_id,
                            attempt.request.id,
                            cause.unwrap_or(RetryCause::Lost),
                            actions,
                        )?;
                    } else {
                        // Cessation can release physical capacity, but absent or
                        // contradictory original usage cannot authorize retry.
                        self.task_mut(report.task_id)?.active_attempt = None;
                    }
                } else {
                    self.finish_attempt(
                        &attempt,
                        AttemptDisposition::Cancelled,
                        ProviderEvidence {
                            resources,
                            execution,
                            console,
                            console_scopes,
                            delegated_console_root,
                        },
                        None,
                    )?;
                    self.emit(DriverEventKind::AttemptCancelled {
                        task_id: report.task_id,
                        attempt_id: report.attempt_id,
                    });
                    self.transition_task(report.task_id, TaskState::Cancelled)?;
                }
            }
            AttemptReport::CertifiedNotStarted {} => {
                self.finish_attempt(
                    &attempt,
                    AttemptDisposition::CertifiedNotStarted,
                    ProviderEvidence::default(),
                    None,
                )?;
                self.resolve_certified_original(
                    report.task_id,
                    attempt.request.id,
                    RetryCause::Lost,
                    actions,
                )?;
            }
            AttemptReport::Cancelled => {
                self.finish_attempt(
                    &attempt,
                    AttemptDisposition::Cancelled,
                    ProviderEvidence {
                        resources,
                        execution,
                        console,
                        console_scopes,
                        delegated_console_root,
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
        let certified_not_started = matches!(report.report, AttemptReport::CertifiedNotStarted {});
        let certified_original = certified_not_started
            && matches!(
                attempt.state,
                AttemptState::Assigned | AttemptState::Starting
            )
            && self
                .snapshot
                .tasks
                .get(&report.task_id)
                .is_some_and(|task| {
                    task.state == TaskState::Indeterminate
                        && task.active_attempt == Some(report.attempt_id)
                })
            && attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|evidence| {
                    evidence.retired_at_driver_millis.is_none()
                        && evidence.resources.is_none()
                        && evidence.execution.is_none()
                        && evidence.console.is_none()
                        && evidence.console_scopes.is_empty()
                        && evidence.delegated_console_root.is_none()
                        && evidence.failure.is_none()
                        && evidence.failure_kind.is_none()
                });
        if report.driver_fence != attempt.request.driver_fence
            || (report.driver_fence != self.snapshot.driver_fence
                && !original_retirement
                && !certified_original)
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
        if certified_not_started
            && (!certified_original
                || report.unsuccessful_resources.is_some()
                || report.execution.is_some()
                || report.console.is_some()
                || !report.console_scopes.is_empty()
                || report.delegated_console_root.is_some())
        {
            return Some("never-started certification differs from unresolved original assignment");
        }
        if matches!(report.report, AttemptReport::Retired)
            && attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|original| {
                    report.unsuccessful_resources.as_ref().is_some_and(|value| {
                        original
                            .resources
                            .as_ref()
                            .is_some_and(|saved| saved != value)
                    }) || report.execution.as_ref().is_some_and(|value| {
                        original
                            .execution
                            .as_ref()
                            .is_some_and(|saved| saved != value)
                    }) || report.console.as_ref().is_some_and(|value| {
                        original
                            .console
                            .as_ref()
                            .is_some_and(|saved| saved != value)
                    })
                })
        {
            return Some("retirement report differs from original observation tuple");
        }
        if matches!(report.report, AttemptReport::Retired)
            && report.delegated_console_root.is_some()
            && attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|original| {
                    original.delegated_console_root.is_some()
                        && original.delegated_console_root != report.delegated_console_root
                })
        {
            return Some("retirement report replaced original delegated console root");
        }
        if matches!(report.report, AttemptReport::Retired)
            && !report.console_scopes.is_empty()
            && attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|original| {
                    !original.console_scopes.is_empty()
                        && original.console_scopes != report.console_scopes
                })
        {
            return Some("retirement report replaced original scoped console evidence");
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
        if runmat_execution::console::validate_scopes(&report.console_scopes).is_err()
            || (report.delegated_console_root.is_some()
                && matches!(&report.report, AttemptReport::Succeeded { result } if !matches!(result, crate::task::AttemptSuccess::Graph { .. })))
            || runmat_execution::console::validate_delegated_root(
                report.console.as_deref(),
                &report.console_scopes,
                report.delegated_console_root,
                matches!(report.report, AttemptReport::Succeeded { .. }),
            )
            .is_err()
            || (report.delegated_console_root.is_some()
                && matches!(
                    report.report,
                    AttemptReport::Started | AttemptReport::CertifiedNotStarted {}
                ))
            || (!report.console_scopes.is_empty()
                && matches!(
                    report.report,
                    AttemptReport::Started | AttemptReport::CertifiedNotStarted {}
                ))
            || runmat_execution::console::validate_source_controls(
                report.console.as_deref(),
                &report.console_scopes,
                report.delegated_console_root,
                attempt.request.task.console,
                report.attempt_id,
            )
            .is_err()
        {
            return Some("invalid original scoped console evidence");
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
        if let AttemptReport::Succeeded { result } = &report.report {
            if result.resources().is_some_and(|resources| {
                report.execution.as_ref().is_some_and(|execution| {
                    resources.resources.usage.wall_millis != execution.wall_millis
                })
            }) {
                return Some("success accounting differs from observed execution interval");
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
        self.finish_unsuccessful(task_id, cause, None)
    }

    /// Resolve only an exact original reservation whose non-execution or
    /// complete original usage has been certified. Its ordinal is retained.
    fn resolve_certified_original(
        &mut self,
        task_id: TaskId,
        attempt_id: runmat_execution::identity::AttemptId,
        cause: RetryCause,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let now = self.snapshot.now_millis;
        let deadline = self
            .task_record(task_id)?
            .submission
            .request
            .deadline_unix_millis;
        if self.task_record(task_id)?.cancellation.is_some() {
            self.transition_task(task_id, TaskState::Cancelled)?;
        } else if deadline.is_some_and(|at| at <= now) {
            self.emit(DriverEventKind::DeadlineExpired { task_id });
            self.task_mut(task_id)?.cancellation = Some(crate::TaskCancellation {
                cause: crate::TaskCancellationCause::Deadline,
                requested_at_driver_millis: now,
            });
            self.transition_task(task_id, TaskState::Cancelled)?;
        } else {
            self.finish_unsuccessful(task_id, cause, Some(attempt_id))?;
            if self.task_record(task_id)?.state == TaskState::Ready {
                self.snapshot.deadlines.insert(task_id, deadline);
            }
        }
        if self.task_record(task_id)?.state != TaskState::Indeterminate {
            self.reopen_certified_dependency_descendants(task_id, actions)?;
        }
        Ok(())
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
            console_scopes: evidence.console_scopes,
            delegated_console_root: evidence.delegated_console_root,
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

    fn finish_unsuccessful(
        &mut self,
        task_id: TaskId,
        cause: RetryCause,
        certified_original: Option<runmat_execution::identity::AttemptId>,
    ) -> RunnerResult<()> {
        let recovering_certified = certified_original.is_some_and(|attempt_id| {
            self.snapshot.tasks.get(&task_id).is_some_and(|task| {
                task.state == TaskState::Indeterminate && task.active_attempt == Some(attempt_id)
            }) && self
                .snapshot
                .attempts
                .get(&attempt_id)
                .is_some_and(|attempt| {
                    attempt.terminal.as_ref().is_some_and(|terminal| {
                        terminal.disposition == AttemptDisposition::CertifiedNotStarted
                            || (terminal.resources.is_some()
                                && matches!(
                                    terminal.disposition,
                                    AttemptDisposition::Lost | AttemptDisposition::Failed { .. }
                                ))
                    })
                })
        });
        if self.task_is_terminal(task_id)? && !recovering_certified {
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
