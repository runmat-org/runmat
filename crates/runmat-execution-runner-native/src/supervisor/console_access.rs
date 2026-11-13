//! Bounded original-attempt reads; never materialize a whole attachment or scheduler history.
use super::{console_journal, LocalSupervisor};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{console::*, identity::AttemptId, JobHandle};

impl LocalSupervisor {
    pub async fn console_page(
        &self,
        handle: JobHandle,
        attempt_id: AttemptId,
        cursor: ConsoleCursor,
        budget: ConsoleReadBudget,
    ) -> NativeExecutionResult<Option<ConsolePage>> {
        budget
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        // No reconcile/attachment here: querying a running prefix does not clone
        // complete final fields, transcripts, argument inventories or histories.
        let plan = {
            let store = self.store.lock().await;
            let record = store.record(handle.id)?;
            if record.handle != handle {
                return Err(NativeExecutionError::Protocol(
                    "console handle differs from original job".into(),
                ));
            }
            let scheduler = self.scheduler.lock().await;
            let attempt = scheduler
                .driver
                .snapshot_ref()
                .attempts
                .get(&attempt_id)
                .ok_or_else(|| {
                    NativeExecutionError::Protocol("console attempt is absent".into())
                })?;
            if scheduler.route(attempt.request.task_id)? != handle.id
                || attempt.request.task.console.is_none()
            {
                return Err(NativeExecutionError::Protocol(
                    "console attempt differs from original job".into(),
                ));
            }
            console_journal::ConsoleReadPlan::bind(&store.job_dir(handle.id), &attempt.request)?
        };
        // Source reads and chain validation must not hold scheduler/store authority.
        plan.read_page(cursor, budget)
    }
}
