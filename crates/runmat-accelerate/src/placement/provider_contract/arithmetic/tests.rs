use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderWorkload,
};

fn representation(shape: &[u64]) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Host,
    }
}
fn query(operation: Operation) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(identity(operation)),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![representation(&[1, 2]); 2],
        outputs: vec![],
        workload: ProviderWorkload::default(),
    }
}
fn snapshot(provider: &InProcessProvider) -> ProviderCapabilitySnapshot {
    let mut snapshot = provider.capability_snapshot();
    if !snapshot.supports_operation(
        &ProviderOperationIdentity::new(identity(Operation::Add)),
        ProviderOperationFamily::Elementwise,
    ) {
        append(&mut snapshot);
    }
    snapshot
}
fn upload(provider: &InProcessProvider, data: &[f64]) -> runmat_accelerate_api::GpuTensorHandle {
    provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(data),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

#[test]
fn declarations_match_real_in_process_arithmetic_hooks() {
    let provider = InProcessProvider::new();
    let snapshot = snapshot(&provider);
    let left = upload(&provider, &[2.0, 4.0]);
    let right = upload(&provider, &[1.0, 2.0]);
    for (&operation, expected) in
        OPERATIONS
            .iter()
            .zip([[3.0, 6.0], [1.0, 2.0], [2.0, 8.0], [2.0, 2.0], [0.5, 0.5]])
    {
        assert!(feasibility(
            HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            &query(operation)
        )
        .unwrap()
        .is_supported());
        let output = block_on(match operation {
            Operation::Add => provider.elem_add(&left, &right),
            Operation::Subtract => provider.elem_sub(&left, &right),
            Operation::Multiply => provider.elem_mul(&left, &right),
            Operation::RightDivide => provider.elem_div(&left, &right),
            Operation::LeftDivide => provider.elem_div(&right, &left),
            Operation::Power => unreachable!(),
        })
        .unwrap();
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        provider.free(&output).unwrap();
    }
    provider.free(&left).unwrap();
    provider.free(&right).unwrap();
    assert!(feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &query(Operation::Power)
    )
    .is_none());
}

#[test]
fn broadcast_resources_include_expanded_buffers_and_missing_output_description() {
    let provider = InProcessProvider::new();
    let snapshot = snapshot(&provider);
    let mut request = query(Operation::Add);
    request.inputs = vec![representation(&[2, 1]), representation(&[1, 3])];
    let ProviderFeasibility::Supported { estimate } = feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &request,
    )
    .unwrap() else {
        panic!("broadcast support")
    };
    assert_eq!(estimate.transient_bytes, Some(184));
    assert_eq!(estimate.output_bytes, Some(48));
    assert_eq!(estimate.dispatches, Some(3));
    request.inputs[0].shape = vec![0, 1];
    assert!(feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &request
    )
    .unwrap()
    .is_supported());
    request.inputs[0].shape = vec![u64::MAX, 2];
    request.inputs[1].shape = vec![u64::MAX, 2];
    assert!(!feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &request
    )
    .unwrap()
    .is_supported());
}

#[test]
fn representation_and_shape_rejections_do_not_claim_hook_support() {
    let provider = InProcessProvider::new();
    let snapshot = snapshot(&provider);
    for element_type in [
        ProviderElementType::F32,
        ProviderElementType::U64,
        ProviderElementType::ComplexF64,
    ] {
        let mut request = query(Operation::Add);
        request.inputs[0].element_type = element_type;
        assert!(!feasibility(
            HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            &request
        )
        .unwrap()
        .is_supported());
    }
    let mut request = query(Operation::Add);
    request.inputs[1].shape = vec![1, 3];
    assert!(!feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &request
    )
    .unwrap()
    .is_supported());
    request = query(Operation::Add);
    request.outputs = vec![representation(&[2, 1])];
    assert!(!feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &request
    )
    .unwrap()
    .is_supported());
}
