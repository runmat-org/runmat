use super::PipelineOwner;
use anyhow::{ensure, Result};
use hashbrown::HashMap;
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
};
use std::collections::hash_map::RandomState;
use std::sync::{Arc, Mutex};

type Entries<T> = HashMap<u64, PipelineOwner<T>, RandomState>;

// hashbrown retains a structured failed Layout but does not implement Error.
// Keep that original detail instead of replacing it with a generic string.
#[derive(Debug, thiserror::Error)]
#[error("fused pipeline table allocation failed: {0:?}")]
struct TableAllocationError(hashbrown::TryReserveError);

pub(crate) struct PipelineCache<T = wgpu::ComputePipeline> {
    entries: Entries<T>,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    reserved: usize,
    epoch: u64,
    invalid: bool,
}
impl<T> Default for PipelineCache<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::with_hasher(RandomState::new()),
            claim: None,
            reserved: 0,
            epoch: 0,
            invalid: false,
        }
    }
}

/// Field order conserves the table claim until its physical backing is gone.
pub(crate) struct RetiredBacking<T> {
    _entries: Entries<T>,
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

pub(crate) enum CacheLookup<'a, T> {
    Hit(PipelineOwner<T>),
    Miss(CacheReservation<'a, T>),
}

pub(crate) struct CacheReservation<'a, T> {
    cache: &'a Mutex<PipelineCache<T>>,
    key: u64,
    active: bool,
}

#[derive(Clone, Copy)]
struct Plan {
    epoch: u64,
    needed: usize,
    bytes: u64,
    replace: bool,
}

impl<T> PipelineCache<T> {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
    pub(crate) fn contains_key(&self, key: &u64) -> bool {
        self.entries.contains_key(key)
    }
    pub(crate) fn get(&self, key: &u64) -> Option<&PipelineOwner<T>> {
        self.entries.get(key)
    }

    fn plan(&self, managed: bool) -> Result<Plan> {
        ensure!(
            !self.invalid,
            "fused pipeline cache reservation state is invalid"
        );
        self.epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("fused pipeline cache generation overflow"))?;
        let needed = self
            .entries
            .len()
            .checked_add(self.reserved)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("fused pipeline cache capacity overflow"))?;
        let replace = needed > self.entries.capacity();
        let bytes = if replace {
            runmat_runtime::context::host_allocation::hashbrown_table_layout::<u64, PipelineOwner<T>>(needed)?
                .map_or(Ok(0), |layout| u64::try_from(layout.size()))?
        } else if managed && self.claim.is_none() {
            u64::try_from(self.entries.allocation_size())?
        } else {
            0
        };
        Ok(Plan {
            epoch: self.epoch,
            needed,
            bytes,
            replace,
        })
    }

    fn prepare_backing(
        &mut self,
        plan: Plan,
        claim: &mut Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Result<Option<RetiredBacking<T>>> {
        ensure!(
            self.epoch == plan.epoch,
            "fused pipeline cache changed before allocation admission"
        );
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("fused pipeline cache generation overflow"))?;
        let retired = if plan.replace {
            // A clean replacement avoids tombstone-driven reserve growth and
            // holds BOTH old and new claims until old backing deallocation.
            let mut replacement = HashMap::with_hasher(RandomState::new());
            replacement
                .try_reserve(plan.needed)
                .map_err(TableAllocationError)?;
            ensure!(
                u64::try_from(replacement.allocation_size())? == plan.bytes,
                "fused pipeline table differs from canonical requested layout"
            );
            replacement.extend(self.entries.drain());
            let old = std::mem::replace(&mut self.entries, replacement);
            let old_claim = std::mem::replace(&mut self.claim, claim.take());
            Some(RetiredBacking {
                _entries: old,
                _claim: old_claim,
            })
        } else {
            if claim.is_some() {
                self.claim = claim.take();
            }
            None
        };
        self.epoch = next_epoch;
        Ok(retired)
    }

    /// Claim admission runs outside the cache mutex. A concurrent change
    /// retries its exact quote BEFORE driver creation or backing allocation;
    /// obsolete claims are also retired outside the cache mutex.
    pub(crate) fn reserve<'a>(
        cache: &'a Mutex<Self>,
        key: u64,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> Result<CacheLookup<'a, T>> {
        loop {
            let plan = {
                let guard = cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("fused pipeline cache poisoned"))?;
                ensure!(
                    !guard.invalid,
                    "fused pipeline cache reservation state is invalid"
                );
                if let Some(hit) = guard.get(&key) {
                    return Ok(CacheLookup::Hit(hit.clone()));
                }
                guard.plan(metadata.is_some())?
            };
            let mut admitted: std::result::Result<
                Option<Arc<dyn RetainedHostAllocationClaim>>,
                ProviderHostMetadataError,
            > = if plan.bytes != 0 {
                metadata
                    .map(|metadata| metadata.claim_record(plan.bytes))
                    .transpose()
            } else {
                Ok(None)
            };
            // Only a stale capacity quote can be superseded by new table
            // geometry. Cancellation, authority, overflow and owner failures
            // remain the original cause, even if another key won meanwhile.
            if let Err(error) = &admitted {
                if *error != ProviderHostMetadataError::ExceedsCapacity {
                    return Err((*error).into());
                }
            }
            let retired;
            {
                let mut guard = cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("fused pipeline cache poisoned"))?;
                ensure!(
                    !guard.invalid,
                    "fused pipeline cache reservation state is invalid"
                );
                if let Some(hit) = guard.get(&key) {
                    let hit = hit.clone();
                    drop(guard);
                    drop(admitted);
                    return Ok(CacheLookup::Hit(hit));
                }
                if guard.epoch != plan.epoch {
                    drop(guard);
                    drop(admitted);
                    continue;
                }
                if let Err(error) = admitted {
                    drop(guard);
                    return Err(error.into());
                }
                let reserved = guard
                    .reserved
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("fused pipeline reservation overflow"))?;
                retired = guard
                    .prepare_backing(plan, admitted.as_mut().expect("successful admission"))?;
                guard.reserved = reserved;
            }
            drop(retired);
            drop(admitted);
            return Ok(CacheLookup::Miss(CacheReservation {
                cache,
                key,
                active: true,
            }));
        }
    }

    /// Existing disk warmup is explicitly unmanaged. It still respects live
    /// selected reservations and retires any prior charged backing afterward.
    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub(crate) fn insert_unbounded(
        &mut self,
        key: u64,
        candidate: &PipelineOwner<T>,
    ) -> Result<(PipelineOwner<T>, Option<RetiredBacking<T>>)> {
        if let Some(existing) = self.get(&key) {
            return Ok((existing.clone(), None));
        }
        let plan = self.plan(false)?;
        let retired = self.prepare_backing(plan, &mut None)?;
        self.entries.insert(key, candidate.clone());
        Ok((candidate.clone(), retired))
    }
}

impl<T> CacheReservation<'_, T> {
    /// Publish only into the prearmed slot. A concurrent winner survives;
    /// caller receives the loser for destruction outside the cache mutex.
    pub(crate) fn publish(
        mut self,
        candidate: PipelineOwner<T>,
    ) -> Result<(PipelineOwner<T>, Option<PipelineOwner<T>>, bool)> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("fused pipeline cache poisoned"))?;
        ensure!(
            !guard.invalid,
            "fused pipeline cache reservation state is invalid"
        );
        ensure!(
            guard.reserved != 0,
            "fused pipeline reservation disappeared"
        );
        let next_epoch = guard
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("fused pipeline cache generation overflow"))?;
        if let Some(existing) = guard.get(&self.key) {
            let existing = existing.clone();
            guard.reserved -= 1;
            guard.epoch = next_epoch;
            self.active = false;
            return Ok((existing, Some(candidate), false));
        }
        ensure!(
            guard.entries.len() < guard.entries.capacity(),
            "fused pipeline slot was not prearmed"
        );
        guard.reserved -= 1;
        self.active = false;
        guard.entries.insert(self.key, candidate.clone());
        guard.epoch = next_epoch;
        Ok((candidate, None, true))
    }
}

impl<T> Drop for CacheReservation<'_, T> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let retired = if let Ok(mut guard) = self.cache.lock() {
            if guard.reserved == 0 {
                guard.invalid = true;
                log::error!("fused pipeline reservation disappeared during retirement");
                return;
            }
            guard.reserved -= 1;
            if let Some(next) = guard.epoch.checked_add(1) {
                guard.epoch = next;
            } else {
                guard.invalid = true;
            }
            if guard.entries.is_empty() && guard.reserved == 0 {
                let entries =
                    std::mem::replace(&mut guard.entries, HashMap::with_hasher(RandomState::new()));
                let claim = guard.claim.take();
                Some(RetiredBacking {
                    _entries: entries,
                    _claim: claim,
                })
            } else {
                None
            }
        } else {
            None
        };
        drop(retired);
    }
}
