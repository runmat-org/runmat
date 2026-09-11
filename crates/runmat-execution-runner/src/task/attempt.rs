use std::collections::BTreeSet;

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::resource::{ResourceAccounting, ResourceAccountingSummary};
use runmat_execution::state::AttemptState;
use runmat_execution::task::TaskRequest;
use runmat_execution::value::{ValuePayload, ValueRef};
use runmat_execution::{ExecutionScopeId, TaskId};
use serde::{Deserialize, Serialize};

use crate::cancellation::CancellationEscalation;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptRequest {
    pub id: AttemptId,
    pub task_id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub worker_id: WorkerId,
    pub ordinal: u16,
    pub driver_fence: u64,
    pub task: TaskRequest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptSuccess {
    pub outputs: Vec<ValuePayload>,
    pub result_objects: Vec<ValueRef>,
    /// Present on scheduler attempts. Serial domain executors may prepare an unmetered success,
    /// but the generic driver rejects it as an attempt report until a worker binds accounting.
    pub resources: Option<Box<AttemptResourceAccounting>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptResourceAccounting {
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    pub resources: ResourceAccounting,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptResourceAggregate {
    pub attempts: Vec<AttemptResourceAccounting>,
    pub summary: ResourceAccountingSummary,
}

impl AttemptResourceAggregate {
    pub fn new(
        mut attempts: Vec<AttemptResourceAccounting>,
        graph_wall_millis: u64,
    ) -> Result<Self, String> {
        if attempts.is_empty() {
            return Err("resource aggregate requires at least one attempt".into());
        }
        for attempt in &attempts {
            attempt.validate()?;
        }
        attempts.sort_by_key(|attempt| attempt.attempt_id);
        if attempts
            .iter()
            .map(|attempt| attempt.attempt_id)
            .collect::<BTreeSet<_>>()
            .len()
            != attempts.len()
        {
            return Err("resource aggregate contains duplicate attempts".into());
        }
        let intervals = attempts
            .iter()
            .map(|attempt| attempt.resources.clone())
            .collect::<Vec<_>>();
        let summary = ResourceAccountingSummary::aggregate(&intervals, graph_wall_millis)
            .map_err(|error| error.to_string())?;
        Ok(Self { attempts, summary })
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = Self::new(self.attempts.clone(), self.summary.usage.wall_millis)?;
        if self != &expected {
            return Err("attempt resource aggregate is internally inconsistent".into());
        }
        Ok(())
    }
}

impl AttemptResourceAccounting {
    pub fn new(
        attempt_id: AttemptId,
        worker_id: WorkerId,
        resources: ResourceAccounting,
    ) -> Result<Self, String> {
        resources.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            attempt_id,
            worker_id,
            resources,
        })
    }

    pub fn for_attempt(
        request: &AttemptRequest,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        Self::new(
            request.id,
            request.worker_id,
            ResourceAccounting::for_reserved_interval(
                request.task.resources.clone(),
                wall_millis,
                artifact_bytes,
                egress_bytes,
                relay_bytes,
            )
            .map_err(|error| error.to_string())?,
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        self.resources.validate().map_err(|error| error.to_string())
    }

    pub fn validate_against(&self, request: &AttemptRequest) -> Result<(), String> {
        if self.attempt_id != request.id
            || self.worker_id != request.worker_id
            || self.resources.allocation != request.task.resources
        {
            return Err(
                "attempt resource accounting differs from its allocation or identity".into(),
            );
        }
        self.validate()
    }
}

impl AttemptSuccess {
    pub fn new(
        request: &AttemptRequest,
        outputs: Vec<ValuePayload>,
        result_objects: Vec<ValueRef>,
        wall_millis: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        let artifact_bytes = result_objects.iter().try_fold(0_u64, |total, reference| {
            total
                .checked_add(reference.encoded_length)
                .ok_or_else(|| "attempt result artifact accounting overflowed".to_string())
        })?;
        Ok(Self {
            outputs,
            result_objects,
            resources: Some(Box::new(AttemptResourceAccounting::for_attempt(
                request,
                wall_millis,
                artifact_bytes,
                egress_bytes,
                relay_bytes,
            )?)),
        })
    }

    pub fn validate_against(&self, request: &AttemptRequest) -> Result<(), String> {
        self.validate_accounting()?;
        self.resources
            .as_ref()
            .ok_or_else(|| "attempt success has no worker resource accounting".to_string())?
            .validate_against(request)
    }

    pub fn validate_accounting(&self) -> Result<(), String> {
        let artifact_bytes = self
            .result_objects
            .iter()
            .try_fold(0_u64, |total, reference| {
                total
                    .checked_add(reference.encoded_length)
                    .ok_or_else(|| "attempt result artifact accounting overflowed".to_string())
            })?;
        let resources = self
            .resources
            .as_ref()
            .ok_or_else(|| "attempt success has no worker resource accounting".to_string())?;
        if artifact_bytes != resources.resources.artifact_bytes {
            return Err("attempt resource accounting differs from its result objects".into());
        }
        resources.validate()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureKind {
    Infrastructure,
    Execution,
    Rejected,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum AttemptReport {
    Started,
    Succeeded {
        result: AttemptSuccess,
    },
    Failed {
        kind: AttemptFailureKind,
        message: String,
    },
    Lost {
        message: String,
    },
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub request: AttemptRequest,
    pub state: AttemptState,
    pub assigned_at_millis: u64,
    pub cancellation_requested_at: Option<u64>,
    pub cancellation_escalation: Option<CancellationEscalation>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::resource::{ResourceAccounting, ResourceRequest};

    fn accounting(name: &[u8], wall_millis: u64, artifact_bytes: u64) -> AttemptResourceAccounting {
        AttemptResourceAccounting::new(
            AttemptId::derive(&[name, b"attempt"]),
            WorkerId::derive(&[name, b"worker"]),
            ResourceAccounting::for_reserved_interval(
                ResourceRequest {
                    cpu_millicores: 1_000,
                    memory_bytes: 2_048,
                    scratch_bytes: 1_024,
                    max_wall_millis: 1_000,
                    max_artifact_bytes: 4_096,
                    max_egress_bytes: 100,
                    max_relay_bytes: 100,
                    accelerators: Vec::new(),
                    required_capabilities: BTreeSet::new(),
                },
                wall_millis,
                artifact_bytes,
                0,
                0,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn attempt_aggregate_sums_work_and_preserves_graph_wall_time() {
        let first = accounting(b"first", 10, 20);
        let second = accounting(b"second", 15, 30);
        let aggregate =
            AttemptResourceAggregate::new(vec![second.clone(), first.clone()], 18).unwrap();
        assert_eq!(aggregate.summary.usage.cpu_core_millis, 25);
        assert_eq!(aggregate.summary.usage.memory_byte_millis, 51_200);
        assert_eq!(aggregate.summary.usage.wall_millis, 18);
        assert_eq!(aggregate.summary.artifact_bytes, 50);
        let mut expected_attempts = vec![first, second];
        expected_attempts.sort_by_key(|attempt| attempt.attempt_id);
        assert_eq!(aggregate.attempts, expected_attempts);
        aggregate.validate().unwrap();
    }

    #[test]
    fn attempt_aggregate_rejects_duplicates_short_wall_time_and_substitution() {
        let attempt = accounting(b"only", 10, 20);
        assert!(AttemptResourceAggregate::new(vec![attempt.clone(), attempt.clone()], 10).is_err());
        assert!(AttemptResourceAggregate::new(vec![attempt.clone()], 9).is_err());
        let mut aggregate = AttemptResourceAggregate::new(vec![attempt], 10).unwrap();
        aggregate.summary.artifact_bytes += 1;
        assert!(aggregate.validate().is_err());
    }
}
