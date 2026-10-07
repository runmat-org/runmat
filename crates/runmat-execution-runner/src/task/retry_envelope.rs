//! Count-only cumulative reservation projection for the admitted retry policy.
//! This is requested work, not measured utilization or proof of lost-worker exit.
use super::TaskSubmission;
use crate::{RunnerError, RunnerResult};
use runmat_execution::resource::{ResourceAccounting, ResourceUsage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryResourceEnvelope {
    pub maximum_attempts: u16,
    pub maximum_reserved_usage: ResourceUsage,
    pub maximum_artifact_bytes: u64,
}

impl TaskSubmission {
    pub fn retry_resource_envelope(&self) -> RunnerResult<RetryResourceEnvelope> {
        self.request
            .validate_environment_reservation()
            .map_err(|error| RunnerError::Invalid(error.to_string()))?;
        let count = super::maximum_attempts(self.request.retry);
        if count == 0 {
            return Err(RunnerError::Invalid(
                "retry policy admits zero attempts".into(),
            ));
        }
        let request = &self.request.resources;
        let base = ResourceAccounting::for_reserved_interval(
            request.clone(),
            request.max_wall_millis,
            request.max_artifact_bytes,
            request.max_egress_bytes,
            request.max_relay_bytes,
        )
        .map_err(|error| RunnerError::Invalid(error.to_string()))?;
        let times = |value: u64| {
            value.checked_mul(u64::from(count)).ok_or_else(|| {
                RunnerError::Invalid("cumulative retry reservation overflowed".into())
            })
        };
        let times128 = |value: u128| {
            value.checked_mul(u128::from(count)).ok_or_else(|| {
                RunnerError::Invalid("cumulative retry reservation overflowed".into())
            })
        };
        let mut usage = base.usage;
        // Round each attempt's CPU charge using the shared owner before multiplying.
        usage.cpu_core_millis = times(usage.cpu_core_millis)?;
        usage.memory_byte_millis = times128(usage.memory_byte_millis)?;
        usage.retained_byte_millis = times128(usage.retained_byte_millis)?;
        usage.wall_millis = times(usage.wall_millis)?;
        usage.egress_bytes = times(usage.egress_bytes)?;
        usage.relay_bytes = times(usage.relay_bytes)?;
        for accelerator in &mut usage.accelerator_millis {
            accelerator.device_millis = times(accelerator.device_millis)?;
        }
        Ok(RetryResourceEnvelope {
            maximum_attempts: count,
            maximum_reserved_usage: usage,
            maximum_artifact_bytes: times(base.artifact_bytes)?,
        })
    }
}
