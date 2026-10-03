//! Typed physical hook facts shared by filter2 and imfilter. The public
//! identity remains at the runtime boundary; this is the provider's actual
//! two-input filtering hook, not a parser or placement policy.

use serde::{Deserialize, Serialize};

use crate::{
    allocation::{AllocationAcquisitionError, TensorViewIdentity},
    GpuTensorHandle, ImfilterMode, ImfilterOptions, ImfilterPadding, ImfilterShape,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImfilterWorkload {
    pub mode: ImfilterMode,
    pub shape: ImfilterShape,
    pub padding: ImfilterPadding,
    pub constant_value_bits: u64,
    pub explicit_image: bool,
    /// A wire-decoded workload cannot claim process-local input ownership.
    pub input_backings: Option<ProviderImfilterInputBackings>,
}

impl ProviderImfilterWorkload {
    pub fn from_options(
        options: &ImfilterOptions,
        explicit_image: bool,
        input_backings: Option<ProviderImfilterInputBackings>,
    ) -> Self {
        Self {
            mode: options.mode,
            shape: options.shape,
            padding: options.padding,
            constant_value_bits: options.constant_value.to_bits(),
            explicit_image,
            input_backings,
        }
    }

    pub fn matches_options(self, options: &ImfilterOptions, explicit_image: bool) -> bool {
        self.mode == options.mode
            && self.shape == options.shape
            && self.padding == options.padding
            && self.constant_value_bits == options.constant_value.to_bits()
            && self.explicit_image == explicit_image
    }
}

/// Exact physical allocation and view identity of two borrowed hook inputs.
/// Logical shapes/classes live in ProviderFeasibilityQuery. The live receipt
/// is deliberately skipped in serialized cache facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImfilterInputBackings {
    requested_bytes: [u64; 2],
    #[serde(skip)]
    live_views: Option<[TensorViewIdentity; 2]>,
}

impl ProviderImfilterInputBackings {
    pub fn from_live_handles(
        handles: [&GpuTensorHandle; 2],
    ) -> Result<Self, AllocationAcquisitionError> {
        for handle in handles {
            handle.allocation_lease().validate_live()?;
            handle
                .view()
                .validate(handle.physical())
                .map_err(AllocationAcquisitionError::InvalidView)?;
        }
        Ok(Self {
            requested_bytes: handles.map(|handle| handle.physical().requested_bytes()),
            live_views: Some(handles.map(GpuTensorHandle::view_identity)),
        })
    }

    pub fn matches_live_handles(
        self,
        handles: [&GpuTensorHandle; 2],
    ) -> Result<bool, AllocationAcquisitionError> {
        let Some(views) = self.live_views else {
            return Ok(false);
        };
        let current = Self::from_live_handles(handles)?;
        Ok(current.live_views == Some(views) && current.requested_bytes == self.requested_bytes)
    }

    pub fn unique_live_bytes(
        self,
        handles: [&GpuTensorHandle; 2],
    ) -> Result<Option<u64>, AllocationAcquisitionError> {
        if !self.matches_live_handles(handles)? {
            return Ok(None);
        }
        if handles[0].allocation_identity() == handles[1].allocation_identity() {
            Ok(Some(self.requested_bytes[0].max(self.requested_bytes[1])))
        } else {
            Ok(self.requested_bytes[0].checked_add(self.requested_bytes[1]))
        }
    }

    pub fn live_requested_bytes(self) -> Option<[u64; 2]> {
        self.live_views.map(|_| self.requested_bytes)
    }
}
