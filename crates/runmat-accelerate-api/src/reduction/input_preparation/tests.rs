use super::*;
use crate::ProviderUploadPhysicalEvidence;
use runmat_execution::ResourceByteBound;

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

fn upload() -> ProviderUploadPhysicalEvidence {
    ProviderUploadPhysicalEvidence {
        output_requested_bytes: 8,
        host_workspace_bytes: ResourceByteBound::Known(0),
        provider_scratch_bytes: ResourceByteBound::Known(4),
        completion_queue_slots: 1,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_and_route_separate_inventory_identity() {
    let selected = representation(ProviderElementType::F32, ProviderResidency::Device);
    let exact = ProviderReductionInputPreparation {
        source: representation(ProviderElementType::F32, ProviderResidency::Host),
        prepared: selected.clone(),
        source_physical: None,
        stage: ProviderReductionInputStage::UploadExact { upload: upload() },
    };
    let cast = ProviderReductionInputPreparation {
        source: representation(ProviderElementType::F64, ProviderResidency::Host),
        prepared: selected.clone(),
        source_physical: None,
        stage: ProviderReductionInputStage::HostCastThenUpload {
            upload: upload(),
            conversion_workspace_bytes: 8,
            conversion_elements: 2,
        },
    };
    let recipe = Digest::sha256(b"selected recipe");
    let a = ProviderReductionInputInventory::new_checked(
        recipe,
        vec![exact.clone()],
        std::slice::from_ref(&selected),
    )
    .unwrap();
    let b = ProviderReductionInputInventory::new_checked(
        recipe,
        vec![cast],
        std::slice::from_ref(&selected),
    )
    .unwrap();
    assert_ne!(a.identity(), b.identity());
    assert_eq!(a.recipe_identity(), recipe);
    assert_eq!(b.recipe_identity(), recipe);
    assert!(ProviderReductionInputInventory::new_checked(recipe, vec![exact], &[]).is_err());
    assert!(ProviderReductionInputInventory::new_checked(
        recipe,
        vec![],
        std::slice::from_ref(&selected)
    )
    .is_err());

    let unknown_source = ProviderReductionInputPreparation {
        source: representation(ProviderElementType::F32, ProviderResidency::Unknown),
        prepared: selected.clone(),
        source_physical: None,
        stage: ProviderReductionInputStage::UploadExact { upload: upload() },
    };
    assert!(ProviderReductionInputInventory::new_checked(
        recipe,
        vec![unknown_source],
        std::slice::from_ref(&selected),
    )
    .is_err());

    let mut no_queue = upload();
    no_queue.completion_queue_slots = 0;
    let zero_queue = ProviderReductionInputPreparation {
        source: representation(ProviderElementType::F32, ProviderResidency::Host),
        prepared: selected.clone(),
        source_physical: None,
        stage: ProviderReductionInputStage::UploadExact { upload: no_queue },
    };
    assert!(
        ProviderReductionInputInventory::new_checked(recipe, vec![zero_queue], &[selected],)
            .is_err()
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_descriptor_must_match_source_class_and_extent() {
    let selected = representation(ProviderElementType::F32, ProviderResidency::Device);
    let physical =
        GpuPhysicalDescriptor::new(NumericElementType::F64, GpuTensorStorage::Real, 2, 16, 16)
            .unwrap();
    let entry = ProviderReductionInputPreparation {
        source: selected.clone(),
        prepared: selected.clone(),
        source_physical: Some(physical),
        stage: ProviderReductionInputStage::BorrowResident { device_id: 7 },
    };
    assert!(ProviderReductionInputInventory::new_checked(
        Digest::sha256(b"typed recipe"),
        vec![entry],
        &[selected],
    )
    .is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn class_changing_cast_needs_fresh_queued_proposal() {
    let source = representation(ProviderElementType::F64, ProviderResidency::Device);
    let prepared = representation(ProviderElementType::F32, ProviderResidency::Device);
    let physical =
        GpuPhysicalDescriptor::new(NumericElementType::F64, GpuTensorStorage::Real, 2, 16, 16)
            .unwrap();
    let proposal = ProviderRealElementwisePhysicalProposal {
        output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
        output_buffer_bytes: 8,
        workgroup_size: 8,
        chunk_capacity: 8,
        uniform_allocations: 0,
        uniform_buffer_bytes: 0,
        dispatches: 1,
        queue_submissions: 0,
        pipeline_key: None,
    };
    let entry = ProviderReductionInputPreparation {
        source,
        prepared: prepared.clone(),
        source_physical: Some(physical),
        stage: ProviderReductionInputStage::ResidentCast {
            device_id: 7,
            recipe_identity: Digest::sha256(b"cast"),
            physical: proposal,
        },
    };
    assert!(ProviderReductionInputInventory::new_checked(
        Digest::sha256(b"recipe"),
        vec![entry],
        &[prepared],
    )
    .is_err());
}
