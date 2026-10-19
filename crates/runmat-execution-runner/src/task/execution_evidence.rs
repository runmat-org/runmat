//! Portable original task history, independent of a pool's delivery channel.
//! Validation checks internal lineage; it does not measure missing consumption or
//! attest that an imported worker was authorized by a trusted pool.
use serde::{Deserialize, Serialize};

use super::{AttemptRecord, AttemptResourceAggregate, TaskRecord};
use crate::{RunnerError, RunnerResult};

mod validation;
pub(crate) use validation::validate_history;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskExecutionEvidence {
    pub task: TaskRecord,
    pub attempts: Vec<AttemptRecord>,
    pub driver_fence: u64,
    pub observed_at_driver_millis: u64,
}

impl TaskExecutionEvidence {
    /// Validate the complete ordinal inventory and the original scheduler commit.
    pub fn validate(&self) -> RunnerResult<()> {
        let history = self.attempts.iter().collect::<Vec<_>>();
        validate_history(
            &self.task,
            &history,
            self.driver_fence,
            self.observed_at_driver_millis,
        )
    }

    /// Charge every original attempt. Unknown or live consumption blocks the claim.
    pub fn resource_aggregate(
        &self,
        graph_wall_millis: u64,
    ) -> RunnerResult<AttemptResourceAggregate> {
        self.validate()?;
        let resources = self
            .attempts
            .iter()
            .map(|attempt| {
                attempt
                    .terminal
                    .as_ref()
                    .and_then(|terminal| terminal.resources.as_deref())
                    .cloned()
                    .ok_or_else(|| {
                        RunnerError::Invalid(format!(
                            "task attempt {} has no terminal resource accounting",
                            attempt.request.ordinal
                        ))
                    })
            })
            .collect::<RunnerResult<Vec<_>>>()?;
        AttemptResourceAggregate::new(resources, graph_wall_millis).map_err(RunnerError::Invalid)
    }
}
