//! Non-owning representation facts. Parsing these grants no device access.
use super::AllocationIdentity;
use crate::{GpuHandleProvenance, GpuTensorStorage, NumericElementType, TransposeInfo};
use serde::{Deserialize, Serialize};

/// Immutable facts returned by the allocation record, not by a caller's view.
/// Requested bytes are the backend buffer request, not driver VRAM granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuPhysicalDescriptor {
    element_type: NumericElementType,
    storage: GpuTensorStorage,
    element_capacity: u64,
    payload_bytes: u64,
    requested_bytes: u64,
}

impl GpuPhysicalDescriptor {
    pub fn new(
        element_type: NumericElementType,
        storage: GpuTensorStorage,
        element_capacity: u64,
        payload_bytes: u64,
        requested_bytes: u64,
    ) -> Result<Self, GpuViewError> {
        let lanes = if storage == GpuTensorStorage::ComplexInterleaved {
            2
        } else {
            1
        };
        let minimum = element_capacity
            .checked_mul(element_type.element_size() as u64)
            .and_then(|n| n.checked_mul(lanes))
            .ok_or(GpuViewError::SizeOverflow)?;
        // Some providers encode narrow integer elements in wider device words.
        // Their owning record supplies the actual byte geometry.
        if minimum > payload_bytes || payload_bytes > requested_bytes {
            return Err(GpuViewError::InvalidPhysicalExtent);
        }
        Ok(Self {
            element_type,
            storage,
            element_capacity,
            payload_bytes,
            requested_bytes,
        })
    }
    pub const fn element_type(&self) -> NumericElementType {
        self.element_type
    }
    pub const fn storage(&self) -> GpuTensorStorage {
        self.storage
    }
    pub const fn element_capacity(&self) -> u64 {
        self.element_capacity
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuTensorLayout {
    Contiguous,
    Transpose(TransposeInfo),
}

/// Proposed logical interpretation. It is non-owning until validated against an
/// authentic allocation. Domain reshape rules may be stricter than capacity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTensorView {
    shape: Vec<usize>,
    layout: GpuTensorLayout,
    provenance: Option<GpuHandleProvenance>,
    logical: bool,
}

impl GpuTensorView {
    pub fn contiguous(shape: Vec<usize>) -> Self {
        Self {
            shape,
            layout: GpuTensorLayout::Contiguous,
            provenance: None,
            logical: false,
        }
    }
    pub fn with_layout(mut self, layout: GpuTensorLayout) -> Self {
        self.layout = layout;
        self
    }
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }
    pub fn with_provenance(mut self, provenance: GpuHandleProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }
    /// Propose classification, not evidence of encoded boolean values. Live
    /// admission additionally requires the captured owner's producer facts.
    pub fn with_logical(mut self, logical: bool) -> Self {
        self.logical = logical;
        self
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// The retained allocation may exceed the logical rank for a caller-
    /// supplied view. Resource admission must fund capacity, not just length.
    pub fn shape_capacity(&self) -> usize {
        self.shape.capacity()
    }
    pub const fn layout(&self) -> GpuTensorLayout {
        self.layout
    }
    pub const fn provenance(&self) -> Option<GpuHandleProvenance> {
        self.provenance
    }
    pub const fn is_logical(&self) -> bool {
        self.logical
    }

    /// Logical cardinality, including empty views with otherwise overflowing
    /// axes. Physical admission and readback must use the same empty rule.
    pub fn checked_element_count(&self) -> Result<u64, GpuViewError> {
        if self.shape.contains(&0) {
            return Ok(0);
        }
        self.shape.iter().try_fold(1_u64, |count, &extent| {
            count
                .checked_mul(u64::try_from(extent).map_err(|_| GpuViewError::SizeOverflow)?)
                .ok_or(GpuViewError::SizeOverflow)
        })
    }

    pub fn validate(&self, physical: &GpuPhysicalDescriptor) -> Result<(), GpuViewError> {
        let count = self.checked_element_count()?;
        if count > physical.element_capacity {
            return Err(GpuViewError::OutsideAllocation);
        }
        if let GpuTensorLayout::Transpose(info) = self.layout {
            if self.shape.as_slice() != [info.base_cols, info.base_rows]
                || info.base_rows.checked_mul(info.base_cols).is_none()
                || physical.storage != GpuTensorStorage::Real
            {
                return Err(GpuViewError::InvalidLayout);
            }
        }
        if self.logical
            && (physical.storage != GpuTensorStorage::Real
                || !matches!(
                    physical.element_type,
                    NumericElementType::F32 | NumericElementType::F64 | NumericElementType::U8
                ))
        {
            return Err(GpuViewError::InvalidLogicalClass);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuViewError {
    SizeOverflow,
    InvalidPhysicalExtent,
    OutsideAllocation,
    InvalidLayout,
    InvalidLogicalClass,
    IdentityExhausted,
}
impl std::fmt::Display for GpuViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid GPU tensor view: {self:?}")
    }
}
impl std::error::Error for GpuViewError {}

/// Diagnostic/same-process handoff only. Deserialize never upgrades a lease.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTensorWireDescriptor {
    pub allocation: AllocationIdentity,
    pub physical: GpuPhysicalDescriptor,
    pub view: GpuTensorView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationAcquisitionError {
    WrongOwner,
    StaleGeneration,
    NotLive,
    PhysicalMismatch,
    InvalidView(GpuViewError),
    Unsupported,
    RetainedHostView(runmat_execution::resource::RetainedHostViewError),
}
impl std::fmt::Display for AllocationAcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU allocation acquisition failed: {self:?}")
    }
}
impl std::error::Error for AllocationAcquisitionError {}
