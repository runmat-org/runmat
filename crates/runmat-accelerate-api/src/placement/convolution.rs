//! Admitted convolution controls and physical launch limits. Placement consumes
//! these facts; public option decoding and scheduling remain with their owners.

use runmat_types::ConvolutionMode;
use serde::{Deserialize, Serialize};

use crate::{
    allocation::{AllocationAcquisitionError, GpuTensorLayout, TensorViewIdentity},
    GpuTensorHandle, ProviderConvOrientation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderConvolutionForm {
    Vector {
        orientation: ProviderConvOrientation,
    },
    Matrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConvolutionWorkload {
    pub form: ProviderConvolutionForm,
    pub mode: ConvolutionMode,
    /// Stable byte facts derived from the exact borrowed input allocations.
    /// A deserialized workload never carries the process-local live receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_backings: Option<ProviderConvolutionInputBackings>,
}

/// Physical bytes of the two actual borrowed inputs, in call order. View IDs
/// authenticate this receipt only in its originating process and invocation;
/// they are deliberately absent from the serialized workload/cache identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConvolutionInputBackings {
    requested_bytes: [u64; 2],
    #[serde(skip)]
    live_views: Option<[TensorViewIdentity; 2]>,
}

impl ProviderConvolutionInputBackings {
    pub fn from_full_live_handles(
        handles: [&GpuTensorHandle; 2],
    ) -> Result<Option<Self>, AllocationAcquisitionError> {
        for handle in handles {
            handle.allocation_lease().validate_live()?;
            handle
                .view()
                .validate(handle.physical())
                .map_err(AllocationAcquisitionError::InvalidView)?;
            let count = handle.shape().iter().try_fold(1_u64, |n, &extent| {
                n.checked_mul(u64::try_from(extent).ok()?)
            });
            if handle.view().layout() != GpuTensorLayout::Contiguous
                || count != Some(handle.physical().element_capacity())
            {
                // Both hooks consume the full registered buffer, not a
                // strided or smaller view of a larger allocation.
                return Ok(None);
            }
        }
        Ok(Some(Self {
            requested_bytes: handles.map(|handle| handle.physical().requested_bytes()),
            live_views: Some(handles.map(GpuTensorHandle::view_identity)),
        }))
    }

    /// A wire-decoded claim is not a live allocation receipt.
    pub fn live_requested_bytes(self) -> Option<[u64; 2]> {
        self.live_views.map(|_| self.requested_bytes)
    }

    /// Requested bytes already counted in the provider's live allocation
    /// snapshot. Two full views of one allocation contribute that allocation
    /// only once, even though the physical route's conservative peak may
    /// count both hook arguments.
    pub fn unique_live_bytes(
        self,
        handles: [&GpuTensorHandle; 2],
    ) -> Result<Option<u64>, AllocationAcquisitionError> {
        if !self.matches_live_handles(handles)? {
            return Ok(None);
        }
        let bytes = if handles[0].allocation_identity() == handles[1].allocation_identity() {
            self.requested_bytes[0].max(self.requested_bytes[1])
        } else {
            match self.requested_bytes[0].checked_add(self.requested_bytes[1]) {
                Some(bytes) => bytes,
                None => return Ok(None),
            }
        };
        Ok(Some(bytes))
    }

    /// The caller must recheck the exact borrowed pair immediately before the
    /// selected hook, after any suspension or preparation.
    pub fn matches_live_handles(
        self,
        handles: [&GpuTensorHandle; 2],
    ) -> Result<bool, AllocationAcquisitionError> {
        let Some(views) = self.live_views else {
            return Ok(false);
        };
        let Some(current) = Self::from_full_live_handles(handles)? else {
            return Ok(false);
        };
        Ok(current.live_views == Some(views) && current.requested_bytes == self.requested_bytes)
    }
}

/// Actual conv1d shader limits from the selected WGPU adapter and launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderConvolutionGeometry {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    pub max_workgroups_per_dimension: u32,
    pub workgroup_size: u32,
}
