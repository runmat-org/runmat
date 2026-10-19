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

use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::allocation::{GpuTensorHandle, WeakGpuTensorHandle};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

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
    /// Exact inline/key payload of one selected cold moments entry. Cache
    /// bucket infrastructure is separately bounded at the provider owner.
    pub(in crate::backend::wgpu::provider) fn cold_key_payload_bytes(dims: usize) -> Option<u64> {
        reuse::cold_key_payload_bytes(dims)
    }
    /// Return the exact strong pair already held under this source view and
    /// dimension key. Unlike a warm boolean, the returned allocation leases
    /// remain valid if a later insertion evicts the cache entry before the
    /// selected stage actually consumes its mean.
    pub(super) fn reserve_cached_mean_nd_exec(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
    ) -> Result<Option<runmat_accelerate_api::ProviderPinnedNdMean>> {
        let entry = self.real_floating_entry(source)?;
        if entry.storage != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(source)
        {
            return Ok(None);
        }
        // Selected warm and cold receipts use the same canonical axis order;
        // ordinary direct reductions may still use their original cache key.
        if dims.is_empty()
            || dims.iter().any(|&axis| axis >= entry.shape.len())
            || dims.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Ok(None);
        }
        let Some((mean, ex2)) = self.cached_moment_pair(source, dims)? else {
            return Ok(None);
        };
        let mean_bytes = self.get_entry_raw(&mean)?.buffer.size();
        let ex2_bytes = if mean.allocation_identity() == ex2.allocation_identity() {
            0
        } else {
            self.get_entry_raw(&ex2)?.buffer.size()
        };
        let pair_bytes = mean_bytes
            .checked_add(ex2_bytes)
            .ok_or_else(|| anyhow!("WGPU pinned moments byte count overflow"))?;
        Ok(Some(runmat_accelerate_api::ProviderPinnedNdMean::new(
            source.clone(),
            dims.to_vec(),
            mean,
            ex2,
            pair_bytes,
        )))
    }

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
        let square_entry = self.real_floating_entry(square)?;
        let base_entry = self.real_floating_entry(base)?;
        ensure!(
            square_entry.precision == base_entry.precision
                && square_entry.shape == base_entry.shape,
            "square provenance must preserve its source class and shape"
        );
        if square_entry.precision != self.precision {
            // The alternate F32 shader computed a real square, but the native
            // square-to-ex2 substitution has no cross-precision certificate.
            // Provenance reuse is optional; the produced square remains valid.
            return Ok(());
        }
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
        self.real_floating_entry(source)?;
        let pair = match self.reduction_reuse.lock() {
            Ok(mut cache) => cache.moment_pair(&(source.view_identity(), dims.to_vec())),
            Err(_) => {
                log::error!("WGPU reduction reuse cache poisoned; moment reuse unavailable");
                return Ok(None);
            }
        };
        if let Some((mean, ex2)) = &pair {
            self.validate_nd_moment_pair(source, dims, mean, ex2)?;
        }
        Ok(pair)
    }

    pub(super) fn remember_moments(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
        mean: &GpuTensorHandle,
        ex2: &GpuTensorHandle,
        metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Result<()> {
        self.validate_nd_moment_pair(source, dims, mean, ex2)?;
        // The selected finite route has already admitted its exact cold key.
        // Do not clone the entire provider cache into an unquoted pruning
        // snapshot; stale entries remain subject to the bounded LRU eviction.
        if metadata_claim.is_none() {
            self.prune_dead_moment_sources()?;
        }
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
                metadata_claim,
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
