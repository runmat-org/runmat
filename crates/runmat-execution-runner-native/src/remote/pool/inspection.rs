//! Read original scheduler metadata under its owner lock; no whole-pool copies.
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::TaskId;
use std::collections::BTreeSet;

fn retained_console(
    attempt: &runmat_execution_runner::task::AttemptRecord,
) -> (
    Option<&runmat_execution::console::ConsoleTranscript>,
    &[runmat_execution::console::ScopedConsoleTranscript],
    Option<runmat_execution::console::ConsoleSourceIdentity>,
) {
    if let Some(terminal) = attempt.terminal.as_ref().filter(|terminal| {
        terminal.console.is_some()
            || !terminal.console_scopes.is_empty()
            || terminal.delegated_console_root.is_some()
    }) {
        (
            terminal.console.as_deref(),
            &terminal.console_scopes,
            terminal.delegated_console_root,
        )
    } else if let Some(retired) = &attempt.unverified_retirement {
        (
            retired.console.as_deref(),
            &retired.console_scopes,
            retired.delegated_console_root,
        )
    } else {
        (None, &[], None)
    }
}

impl RemotePoolDriver {
    pub(crate) fn task_console_requested(&self, id: TaskId) -> bool {
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .tasks
            .get(&id)
            .is_some_and(|task| task.submission.request.console.is_some())
    }

    pub(crate) fn cancellation_state_flags(&self) -> (bool, bool) {
        use runmat_execution::state::TaskState;
        let driver = self.driver.lock().expect("remote driver poisoned");
        let mut flags = (false, false);
        for task in driver.snapshot_ref().tasks.values() {
            flags.0 |= task.state == TaskState::Indeterminate;
            flags.1 |= task.state == TaskState::Cancelled;
        }
        flags
    }

    pub(crate) fn has_task_state(&self, state: runmat_execution::state::TaskState) -> bool {
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .tasks
            .values()
            .any(|task| task.state == state)
    }

    pub(crate) fn task_state(&self, id: TaskId) -> Option<runmat_execution::state::TaskState> {
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .tasks
            .get(&id)
            .map(|task| task.state)
    }

    pub(crate) fn encode_terminal_console(
        &self,
        task_id: TaskId,
        outputs: &runmat_execution_artifact::ProgramExecutionOutputs,
    ) -> NativeExecutionResult<Vec<u8>> {
        let admitted_budget = self.console_retention_limits();
        let driver = self.driver.lock().expect("remote driver poisoned");
        let task = driver
            .task_record(task_id)
            .map_err(NativeExecutionError::Driver)?;
        let attempt = if let Some(committed) = &task.committed {
            driver.snapshot_ref().attempts.get(&committed.attempt_id)
        } else {
            driver
                .snapshot_ref()
                .attempts
                .values()
                .filter(|attempt| attempt.request.task_id == task_id)
                .max_by_key(|attempt| attempt.request.ordinal)
        };
        // Root, source inventory and delegation marker share one retained
        // observation, including an interrupted prefix from retirement.
        let (console, _, delegated_root) =
            attempt.map(retained_console).unwrap_or((None, &[], None));
        if let Some(console) = console {
            if Some(console.limits()) != task.submission.request.console {
                return Err(NativeExecutionError::Protocol(
                    "retained console differs from admitted task controls".into(),
                ));
            }
        }
        let root_attempt = attempt.map(|attempt| attempt.request.id);
        let mut scopes = Vec::new();
        for attempt in driver.snapshot_ref().attempts.values() {
            let (transcript, nested, _) = retained_console(attempt);
            for scope in nested {
                if scopes.len() >= admitted_budget.map_or(0, |budget| budget.max_scopes as usize) {
                    return Err(NativeExecutionError::Protocol(
                        "retained console source inventory exceeds admission".into(),
                    ));
                }
                scopes.push(runmat_execution_artifact::ProgramExecutionConsoleScopeRef {
                    scope_id: scope.scope_id,
                    pool_id: scope.pool_id,
                    task_id: scope.task_id,
                    attempt_id: scope.attempt_id,
                    transcript: &scope.transcript,
                });
            }
            if Some(attempt.request.id) == root_attempt {
                continue;
            }
            let Some(transcript) = transcript else {
                continue;
            };
            if Some(transcript.limits()) != attempt.request.task.console {
                return Err(NativeExecutionError::Protocol(
                    "retained child console differs from admitted attempt controls".into(),
                ));
            }
            if scopes.len() >= admitted_budget.map_or(0, |budget| budget.max_scopes as usize) {
                return Err(NativeExecutionError::Protocol(
                    "retained console source inventory exceeds admission".into(),
                ));
            }
            scopes.push(runmat_execution_artifact::ProgramExecutionConsoleScopeRef {
                scope_id: attempt.request.scope_id,
                pool_id: attempt.request.task.pool_id,
                task_id: attempt.request.task_id,
                attempt_id: attempt.request.id,
                transcript,
            });
        }
        // Identity order is deterministic inventory order, never a claim of
        // chronology between independent attempt transcripts.
        scopes.sort_by_key(|scope| {
            (
                scope.scope_id,
                scope.pool_id,
                scope.task_id,
                scope.attempt_id,
            )
        });
        let budget = if console.is_some() || !scopes.is_empty() {
            admitted_budget
        } else {
            None
        };
        outputs
            .encode_with_retained_console(console, &scopes, budget, delegated_root)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
    }

    /// Query only authenticated retained scheduler metadata under its owner lock.
    /// This is generic task state, independent of any FEA producer identity.
    pub(crate) fn retained_committed_task_ids(
        &self,
        candidates: &BTreeSet<TaskId>,
    ) -> BTreeSet<TaskId> {
        let driver = self.driver.lock().expect("remote driver poisoned");
        candidates
            .iter()
            .copied()
            .filter(|id| {
                driver.task_record(*id).is_ok_and(|task| {
                    task.submission.request.scope_id == self.scope_id
                        && task.submission.request.pool_id == self.pool_id
                        && task.state == runmat_execution::state::TaskState::Succeeded
                        && task.committed.is_some()
                })
            })
            .collect()
    }

    /// Original admitted task/history captured under the driver lock, without a pool copy.
    pub fn task_execution_evidence(
        &self,
        task_id: TaskId,
    ) -> NativeExecutionResult<runmat_execution_runner::TaskExecutionEvidence> {
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .task_execution_evidence(task_id)
            .map_err(NativeExecutionError::Driver)
    }

    pub fn snapshot(&self) -> runmat_execution_runner::DriverSnapshot {
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot()
    }

    /// Advisory inspection only; actual launches remain admitted by the shared driver.
    pub(in crate::remote) async fn worker_is_ready_and_routed(
        &self,
        worker_id: runmat_execution::identity::WorkerId,
    ) -> bool {
        if !self.channels.read().await.contains_key(&worker_id) {
            return false;
        }
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .pool_record(self.pool_id)
            .is_ok_and(|pool| {
                pool.workers.get(&worker_id).is_some_and(|worker| {
                    worker.lifecycle == runmat_execution_runner::pool::WorkerLifecycle::Ready
                })
            })
    }
}
