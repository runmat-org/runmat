//! Complete scheduler-owned task history. Never aggregate only the winning retry.
use runmat_execution::TaskId;

use crate::task::{AttemptRecord, AttemptResourceAggregate, TaskExecutionEvidence, TaskRecord};
use crate::{RunnerError, RunnerResult};

use super::{Driver, DriverSnapshot};

impl DriverSnapshot {
    /// Original attempts in admitted ordinal order, including live and unsuccessful
    /// attempts. Validation prevents gaps or substituted request/fence identities.
    pub fn task_attempt_history(&self, task_id: TaskId) -> RunnerResult<Vec<&AttemptRecord>> {
        self.validate()?;
        if !self.tasks.contains_key(&task_id) {
            return Err(RunnerError::UnknownTask(task_id));
        }
        let mut history = self
            .attempts
            .values()
            .filter(|attempt| attempt.request.task_id == task_id)
            .collect::<Vec<_>>();
        history.sort_by_key(|attempt| attempt.request.ordinal);
        Ok(history)
    }

    /// Original reported accounting for every executed attempt. A certified
    /// never-started ordinal is retained in history but has no usage to charge;
    /// unknown consumption and active attempts still block this claim.
    /// The enclosing host supplies its observed graph wall interval separately.
    pub fn task_resource_aggregate(
        &self,
        task_id: TaskId,
        graph_wall_millis: u64,
    ) -> RunnerResult<AttemptResourceAggregate> {
        self.task_execution_evidence(task_id)?
            .resource_aggregate(graph_wall_millis)
    }

    /// Extract original admitted state without copying the enclosing pool inventory.
    pub fn task_execution_evidence(&self, task_id: TaskId) -> RunnerResult<TaskExecutionEvidence> {
        let attempts = self
            .task_attempt_history(task_id)?
            .into_iter()
            .cloned()
            .collect();
        Ok(TaskExecutionEvidence {
            task: self.tasks[&task_id].clone(),
            attempts,
            driver_fence: self.driver_fence,
            observed_at_driver_millis: self.now_millis,
        })
    }
}

impl Driver {
    /// Borrow live scheduler-owned metadata. This is inspection under the caller's
    /// driver authority, not validation or attestation of an imported carrier.
    pub fn task_record(&self, task_id: TaskId) -> RunnerResult<&TaskRecord> {
        self.snapshot
            .tasks
            .get(&task_id)
            .ok_or(RunnerError::UnknownTask(task_id))
    }

    pub fn task_records(&self) -> impl Iterator<Item = &TaskRecord> {
        self.snapshot.tasks.values()
    }

    /// Borrow all original attempts for a task; ordinal order is not implied.
    pub fn task_attempts(
        &self,
        task_id: TaskId,
    ) -> RunnerResult<impl Iterator<Item = &AttemptRecord>> {
        self.task_record(task_id)?;
        Ok(self
            .snapshot
            .attempts
            .values()
            .filter(move |attempt| attempt.request.task_id == task_id))
    }

    /// Extract one original task without cloning the enclosing live pool inventory.
    pub fn task_execution_evidence(&self, task_id: TaskId) -> RunnerResult<TaskExecutionEvidence> {
        self.snapshot.task_execution_evidence(task_id)
    }
}
