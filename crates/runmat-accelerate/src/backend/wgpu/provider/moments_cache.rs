//! Bounded provider-owned reduction reuse, separate from the adapter buffer pool.
//!
//! Physical retirement removes every view of an allocation through the reverse
//! index; detached strong handles are released after the cache mutex is left.

#[path = "moments_cache/bounded.rs"]
mod bounded;
#[path = "moments_cache/drain.rs"]
mod drain;
#[path = "moments_cache/reuse.rs"]
mod reuse;

use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::{GpuTensorHandle, WeakGpuTensorHandle};

use super::WgpuProvider;
pub(super) use drain::drain_detached;
pub(super) use reuse::ReductionReuseCache;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CacheBudget {
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl CacheBudget {
    fn admits(self, entries: usize, bytes: u64) -> bool {
        self.max_entries > 0
            && self.max_bytes > 0
            && entries <= self.max_entries
            && bytes <= self.max_bytes
    }
}

impl WgpuProvider {
    /// Prune dead exact source views without acquiring an allocation owner
    /// under the cache mutex. A sibling physical view cannot reuse their key.
    fn prune_dead_moment_sources(&self) -> Result<()> {
        let candidates = match self.reduction_reuse.lock() {
            Ok(cache) => cache.source_views(),
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; source pruning unavailable");
                return Ok(());
            }
        };
        let stale: Vec<_> = candidates
            .into_iter()
            .filter_map(|(key, source)| source.acquire().is_err().then_some(key))
            .collect();
        if stale.is_empty() {
            return Ok(());
        }
        let detached = match self.reduction_reuse.lock() {
            Ok(mut cache) => cache.remove_stale_sources(&stale),
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; source pruning unavailable");
                return Ok(());
            }
        };
        drain_detached(detached)
    }

    pub(super) fn cached_square_source(
        &self,
        square: &GpuTensorHandle,
    ) -> Result<Option<WeakGpuTensorHandle>> {
        self.get_entry(square)?;
        match self.reduction_reuse.lock() {
            Ok(mut cache) => Ok(cache.square_source(square.view_identity())),
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; square reuse unavailable");
                Ok(None)
            }
        }
    }

    pub(super) fn remember_square(
        &self,
        square: &GpuTensorHandle,
        base: &GpuTensorHandle,
    ) -> Result<()> {
        self.get_entry(square)?;
        self.get_entry(base)?;
        match self.reduction_reuse.lock() {
            Ok(mut cache) => {
                cache.record_square(square.view_identity(), square.allocation_identity(), base);
            }
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; pow2 provenance not retained")
            }
        }
        Ok(())
    }

    pub(super) fn cached_moment_pair(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
    ) -> Result<Option<(GpuTensorHandle, GpuTensorHandle)>> {
        self.get_entry(source)?;
        let pair = match self.reduction_reuse.lock() {
            Ok(mut cache) => cache.moment_pair(&(source.view_identity(), dims.to_vec())),
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; moment reuse unavailable");
                return Ok(None);
            }
        };
        if let Some((mean, ex2)) = &pair {
            self.get_entry_raw(mean)?;
            self.get_entry_raw(ex2)?;
        }
        Ok(pair)
    }

    pub(super) fn remember_moments(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
        mean: &GpuTensorHandle,
        ex2: &GpuTensorHandle,
    ) -> Result<()> {
        self.get_entry(source)?;
        self.prune_dead_moment_sources()?;
        let mean_bytes = self.get_entry_raw(mean)?.allocated_bytes;
        let ex2_bytes = self.get_entry_raw(ex2)?.allocated_bytes;
        // The cache retains physical allocations, not two view descriptors.
        // Aliased moment outputs consume one physical buffer budget.
        let ex2_increment = if mean.allocation_identity() == ex2.allocation_identity() {
            0
        } else {
            ex2_bytes
        };
        let bytes = mean_bytes
            .checked_add(ex2_increment)
            .ok_or_else(|| anyhow!("WGPU moments cache byte count overflow"))?;
        let detached = match self.reduction_reuse.lock() {
            Ok(mut cache) => cache.record_moments(
                (source.view_identity(), dims.to_vec()),
                source,
                mean.clone(),
                ex2.clone(),
                bytes,
            ),
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; moments not retained");
                return Ok(());
            }
        };
        drain_detached(detached)
    }
}

#[cfg(test)]
#[path = "moments_cache/tests.rs"]
mod tests;
