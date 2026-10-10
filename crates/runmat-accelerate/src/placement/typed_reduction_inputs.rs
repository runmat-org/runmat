//! Canonical resource and route projection of one checked input inventory.
//! This remains non-executable; the request owner and backend reauthenticate
//! actual handles before effects and dispatch.
use runmat_accelerate_api::reduction::{
    ProviderReductionInputInventory, ProviderReductionInputStage,
};
use runmat_accelerate_api::ProviderPreparationRoute;
use runmat_execution::{
    ProviderInputStageDemand, ProviderInputStageDemands, ResourceBoundUnavailable,
    ResourceByteBound,
};

#[derive(Clone, Copy)]
pub(crate) struct TypedReductionInputResources {
    pub(crate) route: ProviderPreparationRoute,
    pub(crate) required_upload_bytes: u64,
    pub(crate) host_workspace: ResourceByteBound,
    pub(crate) stages: ProviderInputStageDemands,
    /// Fresh inputs and provider scratch with no authenticated release before
    /// the selected reduction starts. Borrowed inputs are in the snapshot.
    pub(crate) final_created_live_bytes: ResourceByteBound,
}

fn add(left: ResourceByteBound, right: ResourceByteBound) -> ResourceByteBound {
    match (left, right) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => {
            left.checked_add(right).map_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound),
                ResourceByteBound::Known,
            )
        }
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => ResourceByteBound::Unavailable(reason),
    }
}

fn max(left: ResourceByteBound, right: ResourceByteBound) -> ResourceByteBound {
    match (left, right) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => {
            ResourceByteBound::Known(left.max(right))
        }
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => ResourceByteBound::Unavailable(reason),
    }
}

pub(crate) fn typed_reduction_input_resources(
    device_id: u32,
    inventory: &ProviderReductionInputInventory,
) -> Option<TypedReductionInputResources> {
    let mut required_upload_bytes = 0_u64;
    let mut created_live_bytes = ResourceByteBound::Known(0);
    let mut host_workspace = ResourceByteBound::Known(0);
    let mut stages = Vec::new();
    for entry in inventory.entries() {
        let (stage_device, incremental, workspace, queue_slots, upload_bytes) = match &entry.stage {
            ProviderReductionInputStage::BorrowResident { .. } => continue,
            ProviderReductionInputStage::UploadExact { upload } => (
                device_id,
                add(
                    ResourceByteBound::Known(upload.output_requested_bytes),
                    upload.provider_scratch_bytes,
                ),
                upload.host_workspace_bytes,
                upload.completion_queue_slots,
                entry.prepared.checked_byte_len()?,
            ),
            ProviderReductionInputStage::HostCastThenUpload {
                upload,
                conversion_workspace_bytes,
                ..
            } => (
                device_id,
                add(
                    ResourceByteBound::Known(upload.output_requested_bytes),
                    upload.provider_scratch_bytes,
                ),
                add(
                    ResourceByteBound::Known(*conversion_workspace_bytes),
                    upload.host_workspace_bytes,
                ),
                upload.completion_queue_slots,
                entry.prepared.checked_byte_len()?,
            ),
            ProviderReductionInputStage::ResidentCast {
                device_id: cast_device,
                physical,
                ..
            } => {
                if *cast_device != device_id {
                    return None;
                }
                (
                    *cast_device,
                    ResourceByteBound::Known(physical.requested_incremental_peak_bytes()?),
                    ResourceByteBound::Known(0),
                    physical.queue_submissions,
                    0,
                )
            }
        };
        required_upload_bytes = required_upload_bytes.checked_add(upload_bytes)?;
        host_workspace = max(host_workspace, workspace);
        stages.push(ProviderInputStageDemand {
            device_id: stage_device,
            incremental_peak_bytes: add(created_live_bytes, incremental),
            queue_slots,
        });
        created_live_bytes = add(created_live_bytes, incremental);
    }
    Some(TypedReductionInputResources {
        route: ProviderPreparationRoute::TypedReductionInputs {
            identity: inventory.identity(),
        },
        required_upload_bytes,
        host_workspace,
        stages: ProviderInputStageDemands::sequential(stages)?,
        final_created_live_bytes: created_live_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::reduction::ProviderReductionInputPreparation;
    use runmat_accelerate_api::{
        GpuPhysicalDescriptor, GpuTensorStorage, NumericElementType, ProviderElementType,
        ProviderLayout, ProviderRealElementwiseOutputRoute,
        ProviderRealElementwisePhysicalProposal, ProviderRepresentation, ProviderResidency,
        ProviderStorage, ProviderUploadPhysicalEvidence,
    };
    use runmat_execution::Digest;

    fn representation(
        element_type: ProviderElementType,
        residency: ProviderResidency,
    ) -> ProviderRepresentation {
        ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 1],
            residency,
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn sequential_inputs_retain_uploads_and_cast_uniforms_through_execution() {
        let prepared = representation(ProviderElementType::F32, ProviderResidency::Device);
        let upload = ProviderUploadPhysicalEvidence {
            output_requested_bytes: 8,
            host_workspace_bytes: ResourceByteBound::Known(12),
            provider_scratch_bytes: ResourceByteBound::Known(4),
            completion_queue_slots: 1,
        };
        let host = || ProviderReductionInputPreparation {
            source: representation(ProviderElementType::F32, ProviderResidency::Host),
            prepared: prepared.clone(),
            source_physical: None,
            stage: ProviderReductionInputStage::UploadExact { upload },
        };
        let cast = ProviderReductionInputPreparation {
            source: representation(ProviderElementType::F64, ProviderResidency::Device),
            prepared: prepared.clone(),
            source_physical: Some(
                GpuPhysicalDescriptor::new(
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                    2,
                    16,
                    16,
                )
                .unwrap(),
            ),
            stage: ProviderReductionInputStage::ResidentCast {
                device_id: 7,
                recipe_identity: Digest::sha256(b"cast"),
                physical: ProviderRealElementwisePhysicalProposal {
                    output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
                    output_buffer_bytes: 8,
                    workgroup_size: 64,
                    chunk_capacity: 2,
                    uniform_allocations: 2,
                    uniform_buffer_bytes: 16,
                    dispatches: 2,
                    queue_submissions: 2,
                    pipeline_key: Some(1),
                },
            },
        };
        let inventory = ProviderReductionInputInventory::new_checked(
            Digest::sha256(b"reduction"),
            vec![host(), host(), cast],
            &[prepared.clone(), prepared.clone(), prepared.clone()],
        )
        .unwrap();
        let resources = typed_reduction_input_resources(7, &inventory).unwrap();
        assert_eq!(resources.required_upload_bytes, 16);
        assert_eq!(resources.host_workspace, ResourceByteBound::Known(12));
        assert_eq!(
            resources.final_created_live_bytes,
            ResourceByteBound::Known(48)
        );
        let stages: Vec<_> = resources.stages.iter().collect();
        assert_eq!(stages.len(), 1);
        assert_eq!(
            stages[0].incremental_peak_bytes,
            ResourceByteBound::Known(48)
        );
        assert_eq!(stages[0].queue_slots, 2);

        let with_scratch = |scratch| {
            let mut entry = host();
            if let ProviderReductionInputStage::UploadExact { upload } = &mut entry.stage {
                upload.provider_scratch_bytes = scratch;
            }
            ProviderReductionInputInventory::new_checked(
                Digest::sha256(b"reduction"),
                vec![entry],
                &[representation(
                    ProviderElementType::F32,
                    ProviderResidency::Device,
                )],
            )
            .unwrap()
        };
        let unknown = typed_reduction_input_resources(
            7,
            &with_scratch(ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoScratchBound,
            )),
        )
        .unwrap();
        assert_eq!(
            unknown.final_created_live_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
        );
        let overflow =
            typed_reduction_input_resources(7, &with_scratch(ResourceByteBound::Known(u64::MAX)))
                .unwrap();
        assert_eq!(
            overflow.final_created_live_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
        );
    }
}
