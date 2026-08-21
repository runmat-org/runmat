use std::collections::BTreeSet;

use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_test_runner::worker::{BackendError, BackendErrorKind};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBackendConfig {
    pub max_workers: usize,
    pub worker_resources: ResourceInventory,
    pub attempt_resources: ResourceRequest,
}

impl ExecutionBackendConfig {
    pub fn local(max_workers: usize) -> Self {
        Self {
            max_workers,
            worker_resources: default_inventory(),
            attempt_resources: default_request(),
        }
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        if self.max_workers == 0 {
            return Err(rejected("execution-backed test capacity must be non-zero"));
        }
        self.attempt_resources
            .validate()
            .map_err(|error| rejected(error.to_string()))?;
        if self.worker_resources.cpu_millicores == 0
            || self.worker_resources.memory_bytes == 0
            || self.worker_resources.cpu_millicores < self.attempt_resources.cpu_millicores
            || self.worker_resources.memory_bytes < self.attempt_resources.memory_bytes
            || self.worker_resources.scratch_bytes < self.attempt_resources.scratch_bytes
            || !self
                .attempt_resources
                .required_capabilities
                .is_subset(&self.worker_resources.capabilities)
        {
            return Err(rejected(
                "execution-backed test attempt exceeds its worker resource envelope",
            ));
        }
        Ok(())
    }
}

fn default_inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 1024 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: BTreeSet::new(),
    }
}

fn default_request() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 1024 * 1024 * 1024,
        max_wall_millis: 24 * 60 * 60 * 1_000,
        max_artifact_bytes: 1024 * 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::new(),
    }
}

fn rejected(message: impl Into<String>) -> BackendError {
    BackendError::new(BackendErrorKind::Rejected, message)
}
