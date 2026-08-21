use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

use super::PoolSpec;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResizeRequest {
    pub desired_workers: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResizeDecision {
    GrowBy(u32),
    DrainBy(u32),
    Unchanged,
}

impl ResizeRequest {
    pub fn decide(self, spec: &PoolSpec, current_workers: u32) -> RunnerResult<ResizeDecision> {
        if self.desired_workers < spec.min_workers || self.desired_workers > spec.max_workers {
            return Err(RunnerError::Invalid(format!(
                "desired worker count {} is outside pool bounds {}..={}",
                self.desired_workers, spec.min_workers, spec.max_workers
            )));
        }
        Ok(match self.desired_workers.cmp(&current_workers) {
            std::cmp::Ordering::Greater => {
                ResizeDecision::GrowBy(self.desired_workers - current_workers)
            }
            std::cmp::Ordering::Less => {
                ResizeDecision::DrainBy(current_workers - self.desired_workers)
            }
            std::cmp::Ordering::Equal => ResizeDecision::Unchanged,
        })
    }
}
