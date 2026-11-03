use std::sync::atomic::Ordering;
use std::sync::Arc;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution_runner::driver::DriverEventKind;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::task::{AttemptRetirementUnverified, TaskRecord};
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, Driver, DriverAction, DriverCommand,
};

use super::RemotePoolDriver;
use crate::NativeProgramFailure;

type PendingTaskResolution = (
    tokio::sync::oneshot::Sender<(u64, super::CompletionResult)>,
    u64,
    super::CompletionResult,
);

#[derive(Clone, Copy)]
enum ReportAuthority<'a> {
    Unauthenticated,
    PhysicalRetirement(&'a runmat_execution_runner::AttemptRequest),
    CertifiedNeverAdmitted(&'a runmat_execution_runner::AttemptRequest),
}

impl RemotePoolDriver {
    pub(super) fn apply_authenticated_retirement(
        self: &Arc<Self>,
        request: &runmat_execution_runner::AttemptRequest,
        evidence: super::super::AuthenticatedPhysicalRetirement,
        resources: Option<runmat_execution_runner::task::AttemptResourceAccounting>,
    ) -> bool {
        if !evidence.matches(request) {
            return false;
        }
        let resources = match evidence.resources() {
            Ok(Some(observed)) => {
                if resources
                    .as_ref()
                    .is_some_and(|supplied| supplied != &observed)
                {
                    return false;
                }
                Some(observed)
            }
            Ok(None) => resources,
            Err(_) => return false,
        };
        let mut report = BackendReport::for_request(request, AttemptReport::Retired);
        report.unsuccessful_resources = resources.map(Box::new);
        // The exact original-request check and terminal report are one driver
        // lock transaction in apply_report_inner. Cancellation cannot race a
        // pre-check into an unaccounted terminal state.
        self.apply_report_inner(report, ReportAuthority::PhysicalRetirement(request));
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .attempts
            .get(&request.id)
            .is_some_and(|attempt| {
                &attempt.request == request
                    && (attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|record| record.retired_at_driver_millis.is_some())
                        || (attempt.state == AttemptState::Cancelled
                            && attempt.cancellation_requested_at.is_some()))
            })
    }

    pub(super) fn apply_certified_never_admitted(
        self: &Arc<Self>,
        witness: super::recovery::CertifiedNeverAdmitted,
    ) -> bool {
        let request = witness.request();
        if !witness.matches(request) {
            return false;
        }
        let report = BackendReport::for_request(request, AttemptReport::CertifiedNotStarted {});
        self.apply_report_inner(report, ReportAuthority::CertifiedNeverAdmitted(request));
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .attempts
            .get(&request.id)
            .is_some_and(|attempt| {
                &attempt.request == request
                    && attempt.terminal.as_ref().is_some_and(|terminal| {
                        terminal.disposition
                        == runmat_execution_runner::task::AttemptDisposition::CertifiedNotStarted
                    })
            })
    }

    pub(super) fn apply_report(self: &Arc<Self>, report: BackendReport) {
        self.apply_report_inner(report, ReportAuthority::Unauthenticated);
    }

    pub(super) fn apply_report_with_authenticated_evidence(
        self: &Arc<Self>,
        report: BackendReport,
        request: &runmat_execution_runner::AttemptRequest,
        evidence: super::super::AuthenticatedPhysicalRetirement,
    ) -> bool {
        if !evidence.matches(request) {
            self.apply_report(report);
            return false;
        }
        self.apply_report_inner(report, ReportAuthority::PhysicalRetirement(request));
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .attempts
            .get(&request.id)
            .is_some_and(|attempt| {
                &attempt.request == request
                    && (attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|record| record.retired_at_driver_millis.is_some())
                        || (attempt.state == AttemptState::Cancelled
                            && attempt.cancellation_requested_at.is_some()))
            })
    }

    fn apply_report_inner(
        self: &Arc<Self>,
        mut report: BackendReport,
        authority: ReportAuthority<'_>,
    ) {
        if matches!(authority, ReportAuthority::Unauthenticated)
            && matches!(
                &report.report,
                AttemptReport::Retired | AttemptReport::CertifiedNotStarted {}
            )
        {
            // Only the local process owner can issue this certificate. A
            // remote worker cannot prove parent-owned process-tree cleanup.
            report.report = AttemptReport::RetirementUnverified {
                message: "remote retirement certificate has no local process owner".into(),
            };
        }
        let task_id = report.task_id;
        let spmd_task = self.spmd_task_context(task_id);
        let (actions, outcome, committed_watermark) = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let verified_request = match authority {
                ReportAuthority::Unauthenticated => None,
                ReportAuthority::PhysicalRetirement(request)
                | ReportAuthority::CertifiedNeverAdmitted(request) => Some(request),
            };
            let exact_original =
                verified_request.is_some_and(|request| {
                    driver
                        .snapshot_ref()
                        .attempts
                        .get(&request.id)
                        .is_some_and(|attempt| {
                            &attempt.request == request
                                && report.attempt_id == request.id
                                && report.worker_id == request.worker_id
                                && report.task_id == request.task_id
                                && report.driver_fence == request.driver_fence
                                && (attempt.cancellation_requested_at.is_some()
                                    || attempt.unverified_retirement.as_ref().is_some_and(
                                        |record| record.retired_at_driver_millis.is_none(),
                                    ))
                        })
                });
            if exact_original {
                // A lost/cancelled owner's successful late result supplies its
                // original accounting even though its values cannot be committed.
                // Move the existing box before replacing the result disposition.
                if matches!(authority, ReportAuthority::PhysicalRetirement(_))
                    && report.unsuccessful_resources.is_none()
                {
                    if let AttemptReport::Succeeded { result } = &mut report.report {
                        report.unsuccessful_resources = match result {
                            runmat_execution_runner::AttemptSuccess::Values {
                                resources, ..
                            }
                            | runmat_execution_runner::AttemptSuccess::Graph {
                                resources, ..
                            } => resources.take(),
                            runmat_execution_runner::AttemptSuccess::Spmd { .. } => None,
                        };
                    }
                }
                report.report = match authority {
                    ReportAuthority::CertifiedNeverAdmitted(_) => {
                        AttemptReport::CertifiedNotStarted {}
                    }
                    ReportAuthority::PhysicalRetirement(_) => AttemptReport::Retired,
                    ReportAuthority::Unauthenticated => unreachable!(),
                };
            } else if matches!(
                &report.report,
                AttemptReport::Retired | AttemptReport::CertifiedNotStarted {}
            ) {
                report.report = AttemptReport::RetirementUnverified {
                    message:
                        "authenticated retirement no longer matches the original fenced attempt"
                            .into(),
                };
            }
            let terminal_report = report.report.clone();
            let reported_attempt_id = report.attempt_id;
            let retired_active_original = exact_original
                && matches!(&terminal_report, AttemptReport::Retired)
                && driver
                    .task_record(task_id)
                    .ok()
                    .is_some_and(|task| task.active_attempt == Some(reported_attempt_id));
            let previous_events = driver.events().len();
            if let Err(error) = self.validate_console_report_reservation(&report) {
                tracing::error!(%task_id, %error, "remote console report rejected before retention");
                return;
            }
            let actions = match driver.handle(DriverCommand::BackendReport(report)) {
                Ok(actions) => actions,
                Err(error) => {
                    tracing::error!(%task_id, %error, "remote scheduler rejected backend report");
                    return;
                }
            };
            if let Some(attempt) = driver.snapshot_ref().attempts.get(&reported_attempt_id) {
                let (console, scopes) =
                    if let Some(record) = attempt.terminal.as_ref().filter(|record| {
                        record.console.is_some() || !record.console_scopes.is_empty()
                    }) {
                        (record.console.as_deref(), record.console_scopes.as_slice())
                    } else if let Some(record) = attempt.unverified_retirement.as_ref() {
                        (record.console.as_deref(), record.console_scopes.as_slice())
                    } else {
                        (
                            None,
                            &[] as &[runmat_execution::console::ScopedConsoleTranscript],
                        )
                    };
                if let Err(error) = self.settle_console_attempt(&attempt.request, console, scopes) {
                    // A valid accepted transcript cannot exceed its original
                    // pre-dispatch maxima. Never turn an accounting fault into
                    // silently dropped console or a successful completion.
                    tracing::error!(%task_id, %error, "accepted remote console retention accounting failed");
                    return;
                }
            }
            let outcome = driver.task_record(task_id).ok().and_then(|task| {
                if task.state == TaskState::Succeeded {
                    Some(match &task.committed {
                        Some(commit) => self
                            .execution_objects
                            .commit_results(commit.result.result_objects())
                            .map(|()| commit.result.clone())
                            .map_err(|error| {
                                NativeProgramFailure::Infrastructure(error.to_string())
                            }),
                        None => Err(NativeProgramFailure::Infrastructure(
                            "succeeded remote task has no committed result".into(),
                        )),
                    })
                } else if self.direct_bundle.is_none()
                    && task.state == TaskState::Indeterminate
                    && matches!(&terminal_report, AttemptReport::Succeeded { .. })
                {
                    Some(Err(NativeProgramFailure::WorkerLost(
                        "Graph attempt reported success without verified worker retirement".into(),
                    )))
                } else {
                    let original = if retired_active_original && task.state == TaskState::Failed {
                        driver
                            .snapshot_ref()
                            .attempts
                            .get(&reported_attempt_id)
                            .filter(|attempt| attempt.request.task_id == task_id)
                            .filter(|attempt| {
                                attempt.terminal.as_ref().is_some_and(|terminal| {
                                    matches!(
                                        terminal.disposition,
                                        runmat_execution_runner::task::AttemptDisposition::Failed { .. }
                                    )
                                })
                            })
                            .and_then(|attempt| attempt.unverified_retirement.as_deref())
                            .filter(|original| original.retired_at_driver_millis.is_some())
                    } else {
                        None
                    };
                    self.terminal_failure(&driver, task).map(|message| {
                        Err(report_failure(
                            &terminal_report,
                            task.state,
                            message,
                            original,
                        ))
                    })
                }
            });
            // Serialize committed-object protection and stale-result disposal with
            // scheduler admission. Otherwise a discarded overlapping result can
            // remove an accepted object's catalog entry before it is protected.
            // Lock order is driver -> object catalog; neither operation calls back
            // into the driver. Dispatch/worker calls remain outside both locks.
            let actions = actions
                .into_iter()
                .filter_map(|action| match action {
                    DriverAction::GarbageCollectResults { objects, .. } => {
                        self.execution_objects.discard_results(&objects);
                        None
                    }
                    action => Some(action),
                })
                .collect();
            let committed_watermark = driver.events()[previous_events..].iter().find_map(|event| {
                matches!(&event.kind, DriverEventKind::ResultCommitted { task_id: committed, .. }
                    if *committed == task_id)
                .then_some(event.sequence + 1)
            });
            if committed_watermark.is_some() {
                let mut intervals = self
                    .task_wall_intervals
                    .lock()
                    .expect("remote task wall intervals poisoned");
                match intervals.get_mut(&task_id) {
                    Some(super::timing::TaskWallInterval::ObservedPending(owner)) => {
                        if self
                            .task_clocks
                            .lock()
                            .expect("task clock bindings poisoned")
                            .get(&task_id)
                            != Some(owner)
                        {
                            tracing::error!(%task_id, "committed task lost its original observer binding");
                        }
                    }
                    Some(interval) => {
                        if let Err(error) = interval.finish() {
                            tracing::error!(%task_id, %error, "original task wall interval invalid");
                        }
                    }
                    None => {
                        tracing::error!(%task_id, "committed task lacks original wall interval")
                    }
                }
            }
            (actions, outcome, committed_watermark)
        };
        self.dispatch(actions);
        if let Some(watermark) = committed_watermark {
            self.committed_event_watermark
                .fetch_max(watermark, Ordering::AcqRel);
            self.committed_task.notify_one();
        }
        if let Some(outcome) = outcome {
            self.publish_ordered(task_id, spmd_task, outcome);
        }
        self.cancellation_settled.notify_waiters();
    }

    fn terminal_failure(&self, driver: &Driver, task: &TaskRecord) -> Option<String> {
        // Only a retained, authenticated original worker owner makes the
        // old-fence retirement pending. A direct cold restore without that
        // owner cannot obtain the certificate and must report WorkerLost.
        let has_original_owner = task.state == TaskState::Indeterminate
            && task.active_attempt.is_some_and(|attempt_id| {
                driver
                    .snapshot_ref()
                    .attempts
                    .get(&attempt_id)
                    .is_some_and(|attempt| {
                        self.original_owners
                            .lock()
                            .expect("original owners poisoned")
                            .get(&attempt.request.worker_id)
                            .is_some_and(|owner| {
                                owner.driver_fencing_token == attempt.request.driver_fence
                            })
                    })
            });
        terminal_failure(driver, task, has_original_owner)
    }

    pub(super) fn resolve_non_success_terminals(&self) {
        let terminal = {
            let driver = self.driver.lock().expect("remote driver poisoned");
            driver
                .task_records()
                .filter_map(|task| {
                    self.terminal_failure(&driver, task)
                        .map(|message| (task.id(), state_failure(task.state, message)))
                })
                .collect::<Vec<_>>()
        };
        for (task_id, failure) in terminal {
            let spmd_task = self.spmd_task_context(task_id);
            self.publish_ordered(task_id, spmd_task, Err(failure));
        }
    }

    /// Reserve the originating task's deterministic completion order, then
    /// fence its SPMD peers before making that completion observable.
    /// Otherwise the gang waiter can cancel a blocked peer before the
    /// collective broker has released it.
    fn publish_ordered(
        &self,
        task_id: runmat_execution::TaskId,
        spmd_task: Option<runmat_execution::SpmdTaskContext>,
        outcome: super::CompletionResult,
    ) {
        let failed = outcome.is_err();
        let resolution = self.take_task_resolution(task_id, outcome);
        match spmd_task {
            Some(task) => {
                if failed {
                    // Publish the originating result before releasing peers
                    // blocked in this gang's collective rounds.
                    self.collectives
                        .fail_gang(&task.gang, "an SPMD peer terminated");
                }
                Self::publish_task_resolution(resolution);
                self.collectives.rank_terminated(&task.gang, task.rank);
            }
            None => Self::publish_task_resolution(resolution),
        }
    }

    fn spmd_task_context(
        &self,
        task_id: runmat_execution::TaskId,
    ) -> Option<runmat_execution::SpmdTaskContext> {
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .get(&task_id)
            .and_then(|program| match program.as_ref() {
                runmat_execution_artifact::WorkerProgramRequest::Direct(direct) => {
                    match &direct.context {
                        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
                            Some(task.clone())
                        }
                        _ => None,
                    }
                }
                runmat_execution_artifact::WorkerProgramRequest::Graph(_) => None,
            })
    }

    fn take_task_resolution(
        &self,
        task_id: runmat_execution::TaskId,
        outcome: super::CompletionResult,
    ) -> Option<PendingTaskResolution> {
        let resolution = self
            .completions
            .lock()
            .expect("remote completion registry poisoned")
            .remove(&task_id)
            .map(|sender| {
                let order = self.completion_sequence.fetch_add(1, Ordering::AcqRel);
                (sender, order, outcome)
            });
        // Exact admitted programs and Graph transfer authority have the same
        // lifetime as scheduler task records: completed-task replay and durable
        // recovery still need them.
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .remove(&task_id);
        resolution
    }

    fn publish_task_resolution(resolution: Option<PendingTaskResolution>) {
        if let Some((sender, order, outcome)) = resolution {
            let _ = sender.send((order, outcome));
        }
    }
}

/// Typed failure for the report that made a task terminal; durable diagnostics
/// from the scheduler record remain the message authority.
fn report_failure(
    report: &AttemptReport,
    state: TaskState,
    message: String,
    original: Option<&AttemptRetirementUnverified>,
) -> NativeProgramFailure {
    match report {
        AttemptReport::RuntimeFailed { failure } => NativeProgramFailure::Runtime(failure.clone()),
        AttemptReport::Failed {
            kind: AttemptFailureKind::Infrastructure,
            ..
        } => NativeProgramFailure::Infrastructure(message),
        AttemptReport::Lost { .. } | AttemptReport::RetirementUnverified { .. } => {
            NativeProgramFailure::WorkerLost(message)
        }
        AttemptReport::Cancelled => NativeProgramFailure::Cancelled,
        // These reports prove physical disposition, not the reason the task
        // became terminal. Retry exhaustion and original cancellation remain
        // decisions of the scheduler's accepted task state.
        AttemptReport::Retired if state == TaskState::Failed => match original {
            Some(AttemptRetirementUnverified {
                failure: Some(failure),
                ..
            }) => NativeProgramFailure::Runtime(failure.as_ref().clone()),
            Some(AttemptRetirementUnverified {
                failure_kind: Some(AttemptFailureKind::Infrastructure),
                ..
            }) => NativeProgramFailure::Infrastructure(message),
            _ => state_failure(state, message),
        },
        AttemptReport::Retired | AttemptReport::CertifiedNotStarted {} => {
            state_failure(state, message)
        }
        _ => state_failure(state, message),
    }
}

fn state_failure(state: TaskState, message: String) -> NativeProgramFailure {
    match state {
        TaskState::Cancelled => NativeProgramFailure::Cancelled,
        TaskState::Indeterminate => NativeProgramFailure::WorkerLost(message),
        _ => NativeProgramFailure::Execution(message),
    }
}

fn terminal_failure(
    driver: &Driver,
    task: &TaskRecord,
    has_original_owner: bool,
) -> Option<String> {
    let task_id = task.id();
    // Reconciliation records an old-fence attempt as provisionally
    // Indeterminate before the replacement driver checks its original worker
    // retirement certificate. Do not consume the restored completion waiter
    // at the restore-time tick; the certificate must first resolve that exact
    // attempt to a retry, cancellation, failure, or genuinely unknown result.
    let snapshot = driver.snapshot_ref();
    if has_original_owner
        && task.state == TaskState::Indeterminate
        && task.active_attempt.is_some_and(|attempt_id| {
            snapshot.attempts.get(&attempt_id).is_some_and(|attempt| {
                attempt.request.task_id == task_id
                    && attempt.request.driver_fence < snapshot.driver_fence
                    && attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|record| record.retired_at_driver_millis.is_none())
            })
        })
    {
        return None;
    }
    let message = match task.state {
        TaskState::Failed => "remote task failed",
        TaskState::Indeterminate => "remote worker was lost",
        TaskState::Cancelled
            if !driver
                .task_attempts(task_id)
                .expect("task is borrowed from this driver")
                .any(|attempt| {
                    matches!(
                        attempt.state,
                        AttemptState::Assigned | AttemptState::Starting | AttemptState::Running
                    )
                }) =>
        {
            "remote task was cancelled"
        }
        _ => return None,
    };
    Some(
        driver
            .task_attempts(task_id)
            .expect("task is borrowed from this driver")
            .max_by_key(|attempt| attempt.request.ordinal)
            .and_then(|attempt| {
                attempt
                    .terminal
                    .as_ref()
                    .and_then(|terminal| terminal.diagnostic.clone())
                    .or_else(|| {
                        attempt
                            .unverified_retirement
                            .as_ref()
                            .map(|evidence| evidence.diagnostic.clone())
                    })
            })
            .unwrap_or_else(|| message.into()),
    )
}

#[cfg(test)]
#[path = "../pool_tests/completion_authority.rs"]
mod tests;
