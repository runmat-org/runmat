//! Private selected-stage witnesses; actual caller/placement tests are separate.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::reduction::{
    ProviderReductionInputPreparation, ProviderReductionInputStage,
};
use runmat_accelerate_api::{
    test_support::TestAllocation, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    NumericElementType, ProviderElementType, ProviderLayout, ProviderRealElementwiseOperation,
    ProviderRealElementwiseRecipe, ProviderRealElementwiseRequest, ProviderRepresentation,
    ProviderResidency, ThreadProviderGuard,
};
use runmat_execution::Digest;
use runmat_value::Value;
use std::sync::atomic::Ordering;

#[path = "tests/fixture.rs"]
mod fixture;
use fixture::Owner;

fn rep(
    element_type: ProviderElementType,
    residency: ProviderResidency,
    shape: Vec<u64>,
) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage: runmat_accelerate_api::ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency,
    }
}

fn host_entry(owner: &Owner) -> ProviderReductionInputPreparation {
    ProviderReductionInputPreparation {
        source: rep(
            ProviderElementType::F64,
            ProviderResidency::Host,
            vec![1, 1],
        ),
        prepared: rep(
            ProviderElementType::F64,
            ProviderResidency::Device,
            vec![1, 1],
        ),
        source_physical: None,
        stage: ProviderReductionInputStage::UploadExact {
            upload: owner.evidence(),
        },
    }
}

fn resident_entry(
    handle: &GpuTensorHandle,
    element_type: ProviderElementType,
) -> ProviderReductionInputPreparation {
    let shape = handle
        .shape()
        .iter()
        .map(|&extent| u64::try_from(extent).unwrap())
        .collect();
    let source = rep(element_type, ProviderResidency::Device, shape);
    ProviderReductionInputPreparation {
        source: source.clone(),
        prepared: source,
        source_physical: Some(*handle.physical()),
        stage: ProviderReductionInputStage::BorrowResident {
            device_id: handle.device_id(),
        },
    }
}

fn plan<'a>(
    owner: &'static Owner,
    entries: Vec<ProviderReductionInputPreparation>,
    actions: Vec<InputAction<'a>>,
    casts: Vec<ProviderRealElementwiseRecipe>,
    binding_cap: u64,
) -> TypedInputPlan<'a> {
    let prepared = entries
        .iter()
        .map(|entry| entry.prepared.clone())
        .collect::<Vec<_>>();
    let inventory = ProviderReductionInputInventory::new_checked(
        Digest::sha256(b"private typed input fixture"),
        entries,
        &prepared,
    )
    .unwrap();
    assert_eq!(actions.len(), inventory.entries().len());
    TypedInputPlan {
        inventory,
        owner,
        binding_cap,
        actions,
        casts,
    }
}

fn allocation(owner: &Owner, shape: Vec<usize>, element: NumericElementType) -> TestAllocation {
    owner
        .allocations
        .allocate_zeroed(shape, element, GpuTensorStorage::Real)
        .unwrap()
}

mod effects;
mod source_guards;
