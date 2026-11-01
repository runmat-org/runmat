//! Read original scheduler metadata under its owner lock; no whole-pool copies.
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::TaskId;
use std::collections::BTreeSet;

impl RemotePoolDriver {
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
