use serde::{Deserialize, Serialize};

use super::MeshingContractError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationPolicyV2 {
    pub maximum_checkpoint_latency_ms: u64,
    pub maximum_work_units_between_checks: u64,
}

impl CancellationPolicyV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.maximum_checkpoint_latency_ms == 0 || self.maximum_work_units_between_checks == 0 {
            return Err(MeshingContractError::invalid(
                "cancellation policy",
                "latency and work interval must be non-zero",
            ));
        }
        Ok(())
    }
}

pub trait MeshingCancellationSignal: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NeverCancelled;

impl MeshingCancellationSignal for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}
