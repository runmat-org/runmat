//! Count-only cumulative reservation projection for the admitted retry policy.
//! This is requested work, not measured utilization or proof of lost-worker exit.
use super::TaskSubmission;
use crate::{RunnerError, RunnerResult};
use runmat_execution::resource::RequestedResourceUsage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryResourceEnvelope {
    pub maximum_attempts: u16,
    pub maximum_reserved_usage: RequestedResourceUsage,
    pub maximum_artifact_bytes: u64,
}

impl TaskSubmission {
    pub fn retry_resource_envelope(&self) -> RunnerResult<RetryResourceEnvelope> {
        self.request
            .validate_invocation_reservation()
            .map_err(|error| RunnerError::Invalid(error.to_string()))?;
        let count = super::maximum_attempts(self.request.retry);
        if count == 0 {
            return Err(RunnerError::Invalid(
                "retry policy admits zero attempts".into(),
            ));
        }
        let request = &self.request.resources;
        let usage = RequestedResourceUsage::for_maximum_intervals(request, count)
            .map_err(|error| RunnerError::Invalid(error.to_string()))?;
        let times = |value: u64| {
            value.checked_mul(u64::from(count)).ok_or_else(|| {
                RunnerError::Invalid("cumulative retry reservation overflowed".into())
            })
        };
        Ok(RetryResourceEnvelope {
            maximum_attempts: count,
            maximum_reserved_usage: usage,
            maximum_artifact_bytes: times(request.max_artifact_bytes)?,
        })
    }
}
