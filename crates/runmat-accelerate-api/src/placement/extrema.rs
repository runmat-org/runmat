//! The language owner has already admitted a dimensional min/max call.
//! These facts distinguish the physical two-output hook from its public
//! requested-output contract; they do not select a device or imply timing.
use serde::{Deserialize, Serialize};
mod global;
pub use global::{
    ProviderCheckedGlobalExtrema, ProviderExecutedGlobalExtrema, ProviderGlobalExtremaObserved,
    ProviderGlobalExtremaOutputRoute, ProviderGlobalExtremaPhysicalProposal,
    ProviderGlobalExtremaRecipe, ProviderGlobalExtremaRequest, ProviderGlobalExtremaScalar,
};

use crate::{
    allocation::{AllocationAcquisitionError, GpuTensorLayout, TensorViewIdentity},
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderElementType,
    ProviderRepresentation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderExtremaKind {
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderExtremaDemand {
    ValuesOnly,
    ValuesAndIndices,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderExtremaWorkload {
    pub kind: ProviderExtremaKind,
    /// Zero-based, in-bounds axis after the builtin's own controls are parsed.
    pub axis: u64,
    /// Language-visible demand. Both outputs are physically allocated by the
    /// current dimensional provider hook even when indices are discarded.
    pub demand: ProviderExtremaDemand,
    /// Optional invocation-local proof of the exact source buffer consumed
    /// without a view materialization. A wire-decoded value loses its live
    /// view receipt and cannot claim this physical route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_backing: Option<ProviderExtremaInputBacking>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderExtremaInputBacking {
    requested_bytes: u64,
    shape: [u64; 2],
    element_type: NumericElementType,
    #[serde(skip)]
    live_view: Option<TensorViewIdentity>,
}

/// Limits of the selected provider's actual two-output dimensional kernel.
/// A missing geometry is unresolved physical work, not a generic reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderExtremaGeometry {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    pub workgroup_size: u32,
    pub max_workgroups_per_dimension: u32,
}

impl ProviderExtremaInputBacking {
    pub fn from_full_live_handle(
        handle: &GpuTensorHandle,
    ) -> Result<Option<Self>, AllocationAcquisitionError> {
        handle.allocation_lease().validate_live()?;
        handle
            .view()
            .validate(handle.physical())
            .map_err(AllocationAcquisitionError::InvalidView)?;
        let count = handle.shape().iter().try_fold(1_u64, |total, &extent| {
            total.checked_mul(u64::try_from(extent).ok()?)
        });
        if handle.view().layout() != GpuTensorLayout::Contiguous
            || handle.view().is_logical()
            || handle.physical().storage() != GpuTensorStorage::Real
            || handle.shape().len() != 2
            || count != Some(handle.physical().element_capacity())
        {
            return Ok(None);
        }
        let shape = [
            u64::try_from(handle.shape()[0]).ok(),
            u64::try_from(handle.shape()[1]).ok(),
        ];
        let [Some(rows), Some(cols)] = shape else {
            return Ok(None);
        };
        Ok(Some(Self {
            requested_bytes: handle.physical().requested_bytes(),
            shape: [rows, cols],
            element_type: handle.physical().element_type(),
            live_view: Some(handle.view_identity()),
        }))
    }

    pub fn live_requested_bytes(self) -> Option<u64> {
        self.live_view.map(|_| self.requested_bytes)
    }

    pub fn matches_representation(self, input: &ProviderRepresentation) -> bool {
        let element_type = match self.element_type {
            NumericElementType::F32 => ProviderElementType::F32,
            NumericElementType::F64 => ProviderElementType::F64,
            NumericElementType::I8 => ProviderElementType::I8,
            NumericElementType::I16 => ProviderElementType::I16,
            NumericElementType::I32 => ProviderElementType::I32,
            NumericElementType::I64 => ProviderElementType::I64,
            NumericElementType::U8 => ProviderElementType::U8,
            NumericElementType::U16 => ProviderElementType::U16,
            NumericElementType::U32 => ProviderElementType::U32,
            NumericElementType::U64 => ProviderElementType::U64,
        };
        self.live_view.is_some() && input.element_type == element_type && input.shape == self.shape
    }

    pub fn matches_live_handle(
        self,
        handle: &GpuTensorHandle,
    ) -> Result<bool, AllocationAcquisitionError> {
        let Some(current) = Self::from_full_live_handle(handle)? else {
            return Ok(false);
        };
        Ok(self.live_view.is_some()
            && self.live_view == current.live_view
            && self.requested_bytes == current.requested_bytes)
    }
}
