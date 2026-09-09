use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::resource::{AcceleratorUsage, ResourceRequest, ResourceUsage};
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
    pub allocation: ResourceRequest,
    /// Quota consumption over the measured worker interval. CPU, memory, and accelerator time
    /// represent the resources reserved for the attempt, not sampled process utilization.
    pub usage: ResourceUsage,
    pub artifact_bytes: u64,
}

impl AttemptResourceAccounting {
    pub fn for_attempt(
        request: &AttemptRequest,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        Self::build(
            request.id,
            request.worker_id,
            request.task.resources.clone(),
            wall_millis,
            artifact_bytes,
            egress_bytes,
            relay_bytes,
        )
    }

    fn build(
        attempt_id: AttemptId,
        worker_id: WorkerId,
        allocation: ResourceRequest,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        let cpu_core_millis = u64::from(allocation.cpu_millicores)
            .checked_mul(wall_millis)
            .ok_or_else(|| "attempt CPU accounting overflowed".to_string())?
            .div_ceil(1_000);
        let memory_byte_millis = u128::from(allocation.memory_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| "attempt memory accounting overflowed".to_string())?;
        let retained_byte_millis = u128::from(artifact_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| "attempt artifact accounting overflowed".to_string())?;
        let accelerator_millis = allocation
            .accelerators
            .iter()
            .map(|accelerator| {
                u64::from(accelerator.count)
                    .checked_mul(wall_millis)
                    .map(|device_millis| AcceleratorUsage {
                        class: accelerator.class.clone(),
                        device_millis,
                    })
                    .ok_or_else(|| "attempt accelerator accounting overflowed".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            attempt_id,
            worker_id,
            allocation,
            usage: ResourceUsage {
                cpu_core_millis,
                memory_byte_millis,
                accelerator_millis,
                wall_millis,
                retained_byte_millis,
                egress_bytes,
                relay_bytes,
            },
            artifact_bytes,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = Self::build(
            self.attempt_id,
            self.worker_id,
            self.allocation.clone(),
            self.usage.wall_millis,
            self.artifact_bytes,
            self.usage.egress_bytes,
            self.usage.relay_bytes,
        )?;
        if self != &expected {
            return Err("attempt resource accounting is internally inconsistent".into());
        }
        Ok(())
    }

    pub fn validate_against(&self, request: &AttemptRequest) -> Result<(), String> {
        if self.attempt_id != request.id
            || self.worker_id != request.worker_id
            || self.allocation != request.task.resources
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
        if artifact_bytes != resources.artifact_bytes {
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
