//! Checked, non-executable projection of ordered typed reduction input preparation.
//! Physical allocation authentication remains with the executing backend.
use crate::{
    GpuPhysicalDescriptor, GpuTensorStorage, NumericElementType, ProviderElementType,
    ProviderLayout, ProviderRealElementwiseOutputRoute, ProviderRealElementwisePhysicalProposal,
    ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderUploadPhysicalEvidence,
};
use anyhow::{ensure, Result};
use runmat_execution::Digest;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum ProviderReductionInputStage {
    BorrowResident {
        device_id: u32,
    },
    UploadExact {
        upload: ProviderUploadPhysicalEvidence,
    },
    HostCastThenUpload {
        upload: ProviderUploadPhysicalEvidence,
        conversion_workspace_bytes: u64,
        conversion_elements: u64,
    },
    ResidentCast {
        device_id: u32,
        recipe_identity: Digest,
        physical: ProviderRealElementwisePhysicalProposal,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProviderReductionInputPreparation {
    pub source: ProviderRepresentation,
    pub prepared: ProviderRepresentation,
    pub source_physical: Option<GpuPhysicalDescriptor>,
    pub stage: ProviderReductionInputStage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProviderReductionInputInventory {
    entries: Vec<ProviderReductionInputPreparation>,
    recipe_identity: Digest,
    identity: Digest,
}

impl ProviderReductionInputInventory {
    pub fn new_checked(
        recipe_identity: Digest,
        entries: Vec<ProviderReductionInputPreparation>,
        selected_inputs: &[ProviderRepresentation],
    ) -> Result<Self> {
        ensure!(
            entries.len() == selected_inputs.len(),
            "typed preparation input count mismatch"
        );
        for (entry, selected) in entries.iter().zip(selected_inputs) {
            ensure!(
                &entry.prepared == selected,
                "typed prepared input differs from selected recipe"
            );
            ensure!(
                entry.source.storage == ProviderStorage::DenseReal
                    && entry.prepared.storage == ProviderStorage::DenseReal
                    && matches!(
                        entry.source.element_type,
                        ProviderElementType::F32 | ProviderElementType::F64
                    )
                    && matches!(
                        entry.prepared.element_type,
                        ProviderElementType::F32 | ProviderElementType::F64
                    )
                    && entry.source.shape == entry.prepared.shape,
                "typed input source or prepared class/shape unavailable"
            );
            let source_device = entry.source.residency == ProviderResidency::Device;
            let source_host = entry.source.residency == ProviderResidency::Host;
            let prepared_device = entry.prepared.residency == ProviderResidency::Device;
            ensure!(
                prepared_device
                    && entry.prepared.checked_byte_len().is_some()
                    && entry.source.checked_byte_len().is_some(),
                "typed input is not a bounded source/device representation"
            );
            if let Some(physical) = entry.source_physical {
                let element_type = match entry.source.element_type {
                    ProviderElementType::F32 => NumericElementType::F32,
                    ProviderElementType::F64 => NumericElementType::F64,
                    _ => unreachable!("checked real element type"),
                };
                ensure!(
                    source_device
                        && physical.element_type() == element_type
                        && physical.storage() == GpuTensorStorage::Real
                        && physical.element_capacity()
                            >= entry.source.checked_element_count().unwrap()
                        && physical.payload_bytes() >= entry.source.checked_byte_len().unwrap()
                        && physical.requested_bytes() >= physical.payload_bytes(),
                    "typed source physical descriptor differs from representation"
                );
            }
            match &entry.stage {
                ProviderReductionInputStage::BorrowResident { .. } => {
                    ensure!(
                        source_device
                            && entry.source == entry.prepared
                            && entry.source_physical.is_some(),
                        "invalid resident borrow stage"
                    );
                }
                ProviderReductionInputStage::UploadExact { upload } => {
                    ensure!(
                        source_host
                            && entry.source_physical.is_none()
                            && entry.source.element_type == entry.prepared.element_type
                            && entry.source.layout == ProviderLayout::ColumnMajorContiguous
                            && entry.prepared.layout == ProviderLayout::ColumnMajorContiguous
                            && upload.output_requested_bytes
                                >= entry.prepared.checked_byte_len().unwrap().max(4)
                            && upload.completion_queue_slots > 0,
                        "invalid exact typed upload stage"
                    );
                }
                ProviderReductionInputStage::HostCastThenUpload {
                    upload,
                    conversion_workspace_bytes,
                    conversion_elements,
                } => {
                    ensure!(
                        source_host
                            && entry.source_physical.is_none()
                            && entry.source.element_type != entry.prepared.element_type
                            && entry.source.layout == ProviderLayout::ColumnMajorContiguous
                            && entry.prepared.layout == ProviderLayout::ColumnMajorContiguous
                            && *conversion_workspace_bytes
                                >= entry.prepared.checked_byte_len().unwrap()
                            && Some(*conversion_elements) == entry.source.checked_element_count()
                            && upload.output_requested_bytes
                                >= entry.prepared.checked_byte_len().unwrap().max(4)
                            && upload.completion_queue_slots > 0,
                        "invalid host cast/upload stage"
                    );
                }
                ProviderReductionInputStage::ResidentCast { physical, .. } => {
                    ensure!(
                        source_device
                            && entry.source_physical.is_some()
                            && entry.source.layout == ProviderLayout::ColumnMajorContiguous
                            && entry.prepared.layout == ProviderLayout::ColumnMajorContiguous
                            && entry.source.element_type != entry.prepared.element_type
                            && physical.output_route
                                == ProviderRealElementwiseOutputRoute::FreshAllocation
                            && physical.requested_incremental_peak_bytes().is_some()
                            && physical.workgroup_size > 0
                            && physical.chunk_capacity > 0
                            && physical.queue_submissions > 0
                            && physical.output_buffer_bytes
                                >= entry.prepared.checked_byte_len().unwrap(),
                        "invalid resident cast stage"
                    );
                }
            }
        }
        let identity = Digest::sha256_serialized(&(
            "typed-reduction-input-preparation-v1",
            recipe_identity,
            &entries,
        ))?;
        Ok(Self {
            entries,
            recipe_identity,
            identity,
        })
    }

    pub fn entries(&self) -> &[ProviderReductionInputPreparation] {
        &self.entries
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }

    pub fn recipe_identity(&self) -> Digest {
        self.recipe_identity
    }
}

#[cfg(test)]
mod tests;
