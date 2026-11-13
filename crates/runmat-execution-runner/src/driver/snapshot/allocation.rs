//! Reconstruct only reservation metadata; no executable or output is cloned.
use std::collections::BTreeMap;

use crate::scheduler::{self, ResourceAllocation};
use crate::{RunnerError, RunnerResult};

use super::DriverSnapshot;

impl DriverSnapshot {
    pub(super) fn validate_allocations(&self) -> RunnerResult<()> {
        let mut pools = self
            .pools
            .keys()
            .map(|id| (*id, (ResourceAllocation::default(), 0u32)))
            .collect::<BTreeMap<_, _>>();
        let mut workers = self
            .pools
            .values()
            .flat_map(|pool| pool.workers.keys())
            .map(|id| (*id, (ResourceAllocation::default(), 0u32, false)))
            .collect::<BTreeMap<_, _>>();
        let worker_count = self.pools.values().try_fold(0usize, |count, pool| {
            count
                .checked_add(pool.workers.len())
                .ok_or_else(|| invalid("worker inventory count overflow"))
        })?;
        if workers.len() != worker_count {
            return Err(invalid("worker identity appears in multiple pools"));
        }
        for attempt in self
            .attempts
            .values()
            .filter(|attempt| attempt.terminal.is_none())
        {
            let (pool, count) = pools
                .get_mut(&attempt.request.task.pool_id)
                .ok_or_else(|| invalid("live attempt references an unknown pool"))?;
            scheduler::reserve_scalar(pool, &attempt.request.task.resources)?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| invalid("pool attempt count overflow"))?;
            let (worker, count, uncertain) = workers
                .get_mut(&attempt.request.worker_id)
                .ok_or_else(|| invalid("live attempt references an unknown worker"))?;
            scheduler::reserve(
                worker,
                &attempt.request.task.resources,
                &attempt.request.resource_assignment,
            )?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| invalid("worker attempt count overflow"))?;
            *uncertain |= attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none());
        }
        for (id, pool) in &self.pools {
            let occupied = pool
                .workers
                .values()
                .filter(|worker| {
                    worker.active_attempts > 0
                        || !matches!(
                            worker.lifecycle,
                            crate::pool::WorkerLifecycle::Lost
                                | crate::pool::WorkerLifecycle::Stopped
                        )
                })
                .count();
            if occupied > pool.spec.max_workers as usize {
                return Err(invalid(
                    "physical worker occupancy exceeds the original pool bound",
                ));
            }
            let (allocation, count) = &pools[id];
            if allocation != &pool.allocated || *count != pool.active_attempts {
                return Err(invalid(
                    "pool allocation differs from its live original attempts",
                ));
            }
            for (id, worker) in &pool.workers {
                let (allocation, count, uncertain) = &workers[id];
                if allocation != &worker.allocated || *count != worker.active_attempts {
                    return Err(invalid(
                        "worker allocation differs from its live original attempts",
                    ));
                }
                if *uncertain && worker.accepts_work() {
                    return Err(invalid(
                        "worker with unresolved retirement accepts replacement work",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn invalid(message: &str) -> RunnerError {
    RunnerError::Invalid(message.into())
}
