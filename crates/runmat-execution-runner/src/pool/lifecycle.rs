use std::collections::BTreeMap;

use runmat_execution::identity::WorkerId;
use runmat_execution::state::PoolState;
use serde::{Deserialize, Serialize};

use crate::scheduler::ResourceAllocation;

use super::{PoolSpec, WorkerSpec};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycle {
    Ready,
    Draining,
    Stopped,
    Lost,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerRecord {
    pub spec: WorkerSpec,
    pub lifecycle: WorkerLifecycle,
    pub allocated: ResourceAllocation,
    pub active_attempts: u32,
}

impl WorkerRecord {
    pub fn new(spec: WorkerSpec) -> Self {
        Self {
            spec,
            lifecycle: WorkerLifecycle::Ready,
            allocated: ResourceAllocation::default(),
            active_attempts: 0,
        }
    }

    pub fn accepts_work(&self) -> bool {
        self.lifecycle == WorkerLifecycle::Ready
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PoolRecord {
    pub spec: PoolSpec,
    pub state: PoolState,
    pub workers: BTreeMap<WorkerId, WorkerRecord>,
    pub active_attempts: u32,
    pub allocated: ResourceAllocation,
}

impl PoolRecord {
    pub fn new(spec: PoolSpec) -> Self {
        Self {
            spec,
            state: PoolState::Creating,
            workers: BTreeMap::new(),
            active_attempts: 0,
            allocated: ResourceAllocation::default(),
        }
    }

    pub fn accepts_work(&self) -> bool {
        self.state == PoolState::Ready && self.active_attempts < self.spec.max_in_flight
    }

    pub fn fits(&self, request: &runmat_execution::resource::ResourceRequest) -> bool {
        crate::scheduler::fits(&self.spec.resource_limit, &self.allocated, request)
    }
}
