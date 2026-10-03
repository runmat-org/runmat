//! Typed physical hook facts shared by filter2 and imfilter. The public
//! identity remains at the runtime boundary; this is the provider's actual
//! two-input filtering hook, not a parser or placement policy.

use serde::{Deserialize, Serialize};

use crate::{
    allocation::{AllocationAcquisitionError, TensorViewIdentity},
    handle_is_logical, handle_is_transposed, GpuTensorHandle, GpuTensorStorage, ImfilterMode,
    ImfilterOptions, ImfilterPadding, ImfilterShape, NumericElementType, ProviderElementType,
    ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};

/// Device limits of the provider's actual imfilter shader, not a generic GPU
/// tuning table. Providers without this direct recipe return no geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderImfilterGeometry {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    pub max_rank: usize,
    pub max_workgroups_per_dispatch: u32,
    pub workgroup_size: u32,
    pub shader_f64: bool,
}

/// Provider-owned physical stages for one audited direct hook recipe. The
/// returned host bound includes internal native H readback/plan workspace;
/// runtime-owned preparation before the hook is accounted separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderImfilterPhysicalEvidence {
    pub host_workspace: runmat_execution::ResourceByteBound,
    pub input_stages: runmat_execution::ProviderInputStageDemands,
    pub output_queue_slots: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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

    pub fn matches_options(&self, options: &ImfilterOptions, explicit_image: bool) -> bool {
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImfilterInputBackings {
    device_ids: [u32; 2],
    requested_bytes: [u64; 2],
    unique_requested_bytes: Option<u64>,
    representations: [ProviderRepresentation; 2],
    #[serde(skip)]
    live_views: Option<[TensorViewIdentity; 2]>,
}

impl ProviderImfilterInputBackings {
    pub fn from_live_handles(
        handles: [&GpuTensorHandle; 2],
    ) -> Result<Option<Self>, AllocationAcquisitionError> {
        for handle in handles {
            handle.allocation_lease().validate_live()?;
            handle
                .view()
                .validate(handle.physical())
                .map_err(AllocationAcquisitionError::InvalidView)?;
        }
        let representations = handles.map(|handle| {
            if handle.physical().storage() != GpuTensorStorage::Real || handle_is_logical(handle) {
                return None;
            }
            let element_type = match handle.physical().element_type() {
                NumericElementType::F32 => ProviderElementType::F32,
                NumericElementType::F64 => ProviderElementType::F64,
                _ => return None,
            };
            Some(ProviderRepresentation {
                element_type,
                storage: ProviderStorage::DenseReal,
                layout: if handle_is_transposed(handle) {
                    ProviderLayout::Strided
                } else {
                    ProviderLayout::ColumnMajorContiguous
                },
                shape: handle
                    .shape()
                    .iter()
                    .map(|&extent| u64::try_from(extent).ok())
                    .collect::<Option<Vec<_>>>()?,
                residency: ProviderResidency::Device,
            })
        });
        let [Some(image), Some(kernel)] = representations else {
            return Ok(None);
        };
        let requested_bytes = handles.map(|handle| handle.physical().requested_bytes());
        let unique_requested_bytes =
            if handles[0].allocation_identity() == handles[1].allocation_identity() {
                Some(requested_bytes[0].max(requested_bytes[1]))
            } else {
                requested_bytes[0].checked_add(requested_bytes[1])
            };
        Ok(Some(Self {
            device_ids: handles.map(GpuTensorHandle::device_id),
            requested_bytes,
            unique_requested_bytes,
            representations: [image, kernel],
            live_views: Some(handles.map(GpuTensorHandle::view_identity)),
        }))
    }

    pub fn matches_live_handles(
        &self,
        handles: [&GpuTensorHandle; 2],
    ) -> Result<bool, AllocationAcquisitionError> {
        let Some(views) = self.live_views else {
            return Ok(false);
        };
        let Some(current) = Self::from_live_handles(handles)? else {
            return Ok(false);
        };
        Ok(current.live_views == Some(views)
            && current.device_ids == self.device_ids
            && current.requested_bytes == self.requested_bytes
            && current.unique_requested_bytes == self.unique_requested_bytes
            && current.representations == self.representations)
    }

    pub fn unique_live_bytes(
        &self,
        handles: [&GpuTensorHandle; 2],
    ) -> Result<Option<u64>, AllocationAcquisitionError> {
        if !self.matches_live_handles(handles)? {
            return Ok(None);
        }
        Ok(self.unique_requested_bytes)
    }

    pub fn live_requested_bytes(&self) -> Option<[u64; 2]> {
        self.live_views.map(|_| self.requested_bytes)
    }

    pub fn live_device_ids(&self) -> Option<[u32; 2]> {
        self.live_views.map(|_| self.device_ids)
    }

    pub fn live_unique_requested_bytes(&self) -> Option<u64> {
        self.live_views.and(self.unique_requested_bytes)
    }

    pub fn live_representations(&self) -> Option<[ProviderRepresentation; 2]> {
        self.live_views.as_ref()?;
        Some(self.representations.clone())
    }
}
