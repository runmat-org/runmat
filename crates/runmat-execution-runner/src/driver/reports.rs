use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::TaskId;

use crate::port::BackendReport;
use crate::scheduler;
use crate::task::{
    retry_decision, AttemptFailureKind, AttemptRecord, AttemptReport, CommitDecision, ResultCommit,
    RetryCause, RetryDecision,
};
use crate::{RunnerError, RunnerResult};

use super::state::attempt_state_is_terminal;
use super::{Driver, DriverAction, DriverEventKind};

impl Driver {
    pub(super) fn apply_backend_report(
        &mut self,
        report: BackendReport,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let Some(attempt) = self.snapshot.attempts.get(&report.attempt_id).cloned() else {
            self.discard_backend_report(report, "unknown attempt", actions);
            return Ok(());
        };
        if report.driver_fence != self.snapshot.driver_fence
            || report.driver_fence != attempt.request.driver_fence
        {
            self.discard_backend_report(report, "stale driver fence", actions);
            return Ok(());
        }
        if report.task_id != attempt.request.task_id
            || report.worker_id != attempt.request.worker_id
        {
            self.discard_backend_report(report, "attempt identity mismatch", actions);
            return Ok(());
        }
        if attempt_state_is_terminal(attempt.state) {
            self.discard_backend_report(report, "attempt is already terminal", actions);
            return Ok(());
        }
        match report.report {
            AttemptReport::Started => self.start_attempt(&report)?,
            AttemptReport::Succeeded { result } => {
                let uncommitted_objects = result.result_objects.clone();
                self.finish_attempt(&attempt, AttemptState::Completed)?;
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
            AttemptReport::Failed { kind, .. } => {
                self.fail_attempt(&attempt, report.task_id, report.attempt_id, kind)?;
            }
            AttemptReport::Lost { .. } => {
                self.finish_attempt(&attempt, AttemptState::Lost)?;
                self.emit(DriverEventKind::AttemptLost {
                    task_id: report.task_id,
                    attempt_id: report.attempt_id,
                });
                self.finish_unsuccessful(report.task_id, RetryCause::Lost)?;
            }
            AttemptReport::Cancelled => {
                self.finish_attempt(&attempt, AttemptState::Cancelled)?;
                self.emit(DriverEventKind::AttemptCancelled {
                    task_id: report.task_id,
                    attempt_id: report.attempt_id,
                });
                if !self.task_is_terminal(report.task_id)? {
                    self.transition_task(report.task_id, TaskState::Cancelled)?;
                }
            }
        }
        Ok(())
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
        task_id: TaskId,
        attempt_id: runmat_execution::identity::AttemptId,
        kind: AttemptFailureKind,
    ) -> RunnerResult<()> {
        let state = if kind == AttemptFailureKind::Rejected {
            AttemptState::Rejected
        } else {
            AttemptState::Completed
        };
        self.finish_attempt(attempt, state)?;
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
        state: AttemptState,
    ) -> RunnerResult<()> {
        self.snapshot
            .attempts
            .get_mut(&attempt.request.id)
            .expect("attempt was validated")
            .state = state;
        let pool = self.pool_mut(attempt.request.task.pool_id)?;
        let worker = pool
            .workers
            .get_mut(&attempt.request.worker_id)
            .ok_or(RunnerError::UnknownWorker(attempt.request.worker_id))?;
        scheduler::release(&mut worker.allocated, &attempt.request.task.resources);
        worker.active_attempts = worker.active_attempts.saturating_sub(1);
        scheduler::release(&mut pool.allocated, &attempt.request.task.resources);
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

    fn discard_backend_report(
        &mut self,
        report: BackendReport,
        reason: &str,
        actions: &mut Vec<DriverAction>,
    ) {
        if let AttemptReport::Succeeded { result } = report.report {
            actions.push(DriverAction::GarbageCollectResults {
                task_id: report.task_id,
                objects: result.result_objects,
            });
        }
        self.emit(DriverEventKind::ReportDiscarded {
            task_id: report.task_id,
            attempt_id: report.attempt_id,
            reason: reason.into(),
        });
    }
}
