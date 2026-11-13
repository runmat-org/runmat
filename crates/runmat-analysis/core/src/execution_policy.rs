use serde::{Deserialize, Serialize};

pub const MAX_CPU_THREADS_PER_WORKER: u32 = 1_024;

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
    /// Exact shared-memory CPU concurrency available inside each admitted task worker.
    ///
    /// This is distinct from `workers` (independent task slots) and cooperative operator
    /// partitions (distributed/domain decomposition). Numerical kernels must consume the
    /// corresponding admitted CPU resource rather than discovering host parallelism.
    /// `None` (the default) admits every core of the worker; the host resolves it once.
    #[serde(default)]
    pub cpu_threads_per_worker: Option<u32>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
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

/// What execution does when a study requests accelerator placement and the accelerator path
/// cannot run the solve (no device, or no device kernel for that physics family).
///
/// One policy for every physics family (owner decision 2026-10-28): by default the solve runs on
/// the CPU reference path, publishes `cpu_reference` as its execution backend, and emits the
/// `FEA_ACCELERATOR_ADMISSION_FALLBACK` diagnostic. `Required` fails closed instead. Authored
/// sources spell the strict option `accelerators = required` in an execution policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorAdmissionPolicy {
    #[default]
    FallbackToCpu,
    Required,
}

impl AcceleratorAdmissionPolicy {
    /// The default policy: a requested accelerator that cannot run falls back to `cpu_reference`.
    pub fn is_fallback(&self) -> bool {
        *self == Self::FallbackToCpu
    }
}

impl ExecutionResourceLimits {
    /// The admission policy the authored accelerator preference selects.
    pub fn accelerator_admission(&self) -> AcceleratorAdmissionPolicy {
        match self.accelerators {
            AcceleratorPreference::Required => AcceleratorAdmissionPolicy::Required,
            AcceleratorPreference::Disabled | AcceleratorPreference::Preferred => {
                AcceleratorAdmissionPolicy::FallbackToCpu
            }
        }
    }
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
        if self
            .resources
            .cpu_threads_per_worker
            .is_some_and(|threads| threads == 0 || threads > MAX_CPU_THREADS_PER_WORKER)
        {
            return Err(AuthoredExecutionPolicyError::InvalidCpuThreadsPerWorker);
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
            cpu_threads_per_worker: None,
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
    #[error("CPU threads per worker must be in 1..=1024")]
    InvalidCpuThreadsPerWorker,
    #[error("memory-per-worker limit must be positive")]
    InvalidMemoryLimit,
    #[error("parallelism limits must be positive or automatic")]
    InvalidParallelism,
    #[error("backend and accelerator-resource preferences contradict each other")]
    ContradictoryAcceleratorPolicy,
}
