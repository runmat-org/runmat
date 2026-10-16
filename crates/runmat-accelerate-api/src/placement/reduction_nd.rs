//! Selected one-submission WGPU N-D moments route used by a staged mean.
//! Requested bytes are not measured device consumption or a timing sample.
use crate::{allocation::AllocationAcquisitionError, GpuTensorHandle};

#[path = "reduction_nd/ordered_real_mean.rs"]
mod ordered_real_mean;
pub use ordered_real_mean::{
    ProviderExecutedOrderedRealMean, ProviderOrderedRealMean, ProviderOrderedRealMeanControl,
    ProviderOrderedRealMeanLimits, ProviderOrderedRealMeanRecipe, ProviderOrderedRealMeanResources,
    ProviderOrderedRealMeanSource, ProviderOrderedRealMeanStep, ProviderRealMeanNaNMode,
};

/// An exact live cache hit, pinned by strong allocation leases until the
/// selected mean checkpoint consumes it. This is not a cost estimate for the
/// remainder of the image program. The second-moment lease is deliberately
/// retained even when the caller only requests the mean.
pub struct ProviderPinnedNdMean {
    source: GpuTensorHandle,
    dims: Vec<usize>,
    mean: GpuTensorHandle,
    ex2: GpuTensorHandle,
    retained_pair_bytes: u64,
}

impl ProviderPinnedNdMean {
    pub fn new(
        source: GpuTensorHandle,
        dims: Vec<usize>,
        mean: GpuTensorHandle,
        ex2: GpuTensorHandle,
        retained_pair_bytes: u64,
    ) -> Self {
        Self {
            source,
            dims,
            mean,
            ex2,
            retained_pair_bytes,
        }
    }

    pub fn validate_live_source(
        &self,
        source: &GpuTensorHandle,
    ) -> Result<bool, AllocationAcquisitionError> {
        self.source.allocation_lease().validate_live()?;
        self.mean.allocation_lease().validate_live()?;
        self.ex2.allocation_lease().validate_live()?;
        source.allocation_lease().validate_live()?;
        Ok(self.source.view_identity() == source.view_identity()
            && self.mean.allocation_identity() != self.ex2.allocation_identity()
            && self
                .mean
                .physical()
                .requested_bytes()
                .checked_add(self.ex2.physical().requested_bytes())
                .is_some_and(|requested| requested <= self.retained_pair_bytes))
    }

    pub fn mean(&self) -> &GpuTensorHandle {
        &self.mean
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn ex2(&self) -> &GpuTensorHandle {
        &self.ex2
    }

    pub fn retained_pair_bytes(&self) -> u64 {
        self.retained_pair_bytes
    }

    pub fn into_mean(self) -> GpuTensorHandle {
        self.mean
    }
}

pub struct ProviderColdNdMean {
    source: GpuTensorHandle,
    dims: Vec<usize>,
    output_shape: Vec<usize>,
    requested_output_bytes: u64,
    requested_uniform_bytes: u64,
    requested_cache_metadata_payload_bytes: u64,
    cache_view_grants: Option<[Box<dyn runmat_execution::resource::RetainedHostViewGrant>; 2]>,
    cache_metadata_claim:
        Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}

/// Provider-produced completion facts for the selected cold moments kernel.
/// These are one-hook facts, not a complete staged-image cost observation.
pub struct ProviderExecutedColdNdMean {
    pub mean: GpuTensorHandle,
    /// The hidden second moment stays pinned through consumer admission.
    pub ex2: GpuTensorHandle,
    pub requested_pair_bytes: u64,
    pub observed_pair_bytes: u64,
    pub requested_uniform_bytes: u64,
    /// The selected kernel's fixed submission inventory, not a timing sample.
    pub selected_submissions: u32,
}

impl ProviderColdNdMean {
    /// Describe a provider route; this is not executable authority. The
    /// consuming provider revalidates the source, geometry, byte extents and
    /// current binding limits before submitting work.
    pub fn new(
        source: GpuTensorHandle,
        dims: Vec<usize>,
        output_shape: Vec<usize>,
        requested_output_bytes: u64,
        requested_uniform_bytes: u64,
        requested_cache_metadata_payload_bytes: u64,
    ) -> Self {
        Self {
            source,
            dims,
            output_shape,
            requested_output_bytes,
            requested_uniform_bytes,
            requested_cache_metadata_payload_bytes,
            cache_view_grants: None,
            cache_metadata_claim: None,
        }
    }

    /// Bind two single-use host-view claims before the cold selected kernel
    /// runs. The provider retains these exact views with its cache pair; a
    /// failed or declined execution drops the unconsumed grants.
    pub fn with_cache_view_grants(
        mut self,
        grants: [Box<dyn runmat_execution::resource::RetainedHostViewGrant>; 2],
    ) -> Self {
        self.cache_view_grants = Some(grants);
        self
    }

    pub fn take_cache_view_grants(
        &mut self,
    ) -> Option<[Box<dyn runmat_execution::resource::RetainedHostViewGrant>; 2]> {
        self.cache_view_grants.take()
    }

    pub fn with_cache_metadata_claim(
        mut self,
        claim: std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> Self {
        self.cache_metadata_claim = Some(claim);
        self
    }

    pub fn take_cache_metadata_claim(
        &mut self,
    ) -> Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>> {
        self.cache_metadata_claim.take()
    }

    pub fn requested_cache_metadata_payload_bytes(&self) -> u64 {
        self.requested_cache_metadata_payload_bytes
    }

    pub fn source(&self) -> &GpuTensorHandle {
        &self.source
    }
    pub fn validate_live_source(
        &self,
        source: &GpuTensorHandle,
    ) -> Result<bool, AllocationAcquisitionError> {
        self.source.allocation_lease().validate_live()?;
        source.allocation_lease().validate_live()?;
        Ok(self.source.view_identity() == source.view_identity())
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    /// Both mean and hidden E[x²] allocate independently on the cold route.
    pub fn requested_pair_bytes(&self) -> Option<u64> {
        self.requested_output_bytes.checked_mul(2)
    }
    pub fn requested_uniform_bytes(&self) -> u64 {
        self.requested_uniform_bytes
    }
    /// Incremental requested peak for this one kernel, excluding its borrowed
    /// input and all other live staged values/cache entries. Not a whole-stage
    /// memory bound.
    pub fn requested_incremental_peak_bytes(&self) -> Option<u64> {
        self.requested_pair_bytes()?
            .checked_add(self.requested_uniform_bytes)
    }
}
