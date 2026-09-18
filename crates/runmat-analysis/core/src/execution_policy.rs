use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredExecutionPolicyRef {
    pub module_path: String,
    pub policy_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredExecutionPolicy {
    pub module_path: String,
    pub policy_id: String,
    pub placement: PlacementPreference,
    pub backend: BackendPreference,
    pub resources: ExecutionResourceLimits,
    pub fanout: FanoutPolicy,
    pub cooperative: CooperativePolicy,
    pub resilience: ResiliencePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PlacementPreference {
    Auto,
    Local,
    Cluster { profile: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    Auto,
    Cpu,
    Accelerator,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionResourceLimits {
    pub workers: Option<InclusiveCountRange>,
    pub maximum_memory_bytes_per_worker: Option<u64>,
    pub accelerators: AcceleratorPreference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InclusiveCountRange {
    pub minimum: u32,
    pub maximum: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorPreference {
    Disabled,
    Preferred,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FanoutPolicy {
    pub sweep_cases: ParallelismLimit,
    pub frequency_samples: ParallelismLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "maximum")]
pub enum ParallelismLimit {
    Auto,
    Fixed(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooperativePolicy {
    pub operator_partitions: ParallelismLimit,
    pub partitioner: OperatorPartitioner,
    pub balance: PartitionBalance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorPartitioner {
    Auto,
    Graph,
    Geometric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionBalance {
    Nonzeros,
    Dofs,
    Elements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResiliencePolicy {
    pub retries: u16,
    pub checkpoint: CheckpointPreference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointPreference {
    Disabled,
    Automatic,
    Required,
}

impl AuthoredExecutionPolicy {
    pub fn validate(&self) -> Result<(), AuthoredExecutionPolicyError> {
        if self.module_path.trim().is_empty() || self.policy_id.trim().is_empty() {
            return Err(AuthoredExecutionPolicyError::EmptyIdentity);
        }
        if let Some(workers) = self.resources.workers {
            if workers.minimum == 0 || workers.minimum > workers.maximum {
                return Err(AuthoredExecutionPolicyError::InvalidWorkerRange);
            }
        }
        if self.resources.maximum_memory_bytes_per_worker == Some(0) {
            return Err(AuthoredExecutionPolicyError::InvalidMemoryLimit);
        }
        for limit in [
            self.fanout.sweep_cases,
            self.fanout.frequency_samples,
            self.cooperative.operator_partitions,
        ] {
            if matches!(limit, ParallelismLimit::Fixed(0)) {
                return Err(AuthoredExecutionPolicyError::InvalidParallelism);
            }
        }
        if matches!(
            (self.backend, self.resources.accelerators),
            (
                BackendPreference::Cpu,
                AcceleratorPreference::Preferred | AcceleratorPreference::Required
            ) | (
                BackendPreference::Accelerator,
                AcceleratorPreference::Disabled
            )
        ) {
            return Err(AuthoredExecutionPolicyError::ContradictoryAcceleratorPolicy);
        }
        Ok(())
    }
}

impl Default for ExecutionResourceLimits {
    fn default() -> Self {
        Self {
            workers: None,
            maximum_memory_bytes_per_worker: None,
            accelerators: AcceleratorPreference::Disabled,
        }
    }
}

impl Default for FanoutPolicy {
    fn default() -> Self {
        Self {
            sweep_cases: ParallelismLimit::Auto,
            frequency_samples: ParallelismLimit::Auto,
        }
    }
}

impl Default for CooperativePolicy {
    fn default() -> Self {
        Self {
            operator_partitions: ParallelismLimit::Auto,
            partitioner: OperatorPartitioner::Auto,
            balance: PartitionBalance::Nonzeros,
        }
    }
}

impl Default for ResiliencePolicy {
    fn default() -> Self {
        Self {
            retries: 0,
            checkpoint: CheckpointPreference::Automatic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredExecutionPolicyError {
    #[error("execution-policy identity must not be empty")]
    EmptyIdentity,
    #[error("worker range must be positive and ordered")]
    InvalidWorkerRange,
    #[error("memory-per-worker limit must be positive")]
    InvalidMemoryLimit,
    #[error("parallelism limits must be positive or automatic")]
    InvalidParallelism,
    #[error("backend and accelerator-resource preferences contradict each other")]
    ContradictoryAcceleratorPolicy,
}
