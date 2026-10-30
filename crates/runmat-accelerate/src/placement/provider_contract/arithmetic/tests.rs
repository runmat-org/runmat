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
        let request = query(operation);
        let mut receipt = runmat_accelerate_api::observation::ProviderOperationReceipt::new();
        receipt.require_provider_execution(provider.device_id(), request.operation);
        let output = runmat_accelerate_api::observation::with_provider_observation(
            Some(receipt.recorder()),
            || {
                block_on(match operation {
                    Operation::Add => provider.elem_add(&left, &right),
                    Operation::Subtract => provider.elem_sub(&left, &right),
                    Operation::Multiply => provider.elem_mul(&left, &right),
                    Operation::RightDivide => provider.elem_div(&left, &right),
                    Operation::LeftDivide => provider.elem_div(&right, &left),
                    Operation::Power => unreachable!(),
                })
            },
        )
        .unwrap();
        receipt.publish().unwrap();
        assert!(matches!(
            receipt.status(),
            runmat_accelerate_api::observation::ProviderObservationStatus::Completed {
                submissions: 0,
                ..
            }
        ));
        assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
        output.release().unwrap();
    }
    left.release().unwrap();
    right.release().unwrap();
    assert!(feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &query(Operation::Power)
    )
    .is_none());
}

#[test]
fn broadcast_resources_match_direct_native_hooks_and_missing_output_description() {
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
    assert_eq!(estimate.transient_bytes, Some(88));
    assert_eq!(estimate.output_bytes, Some(48));
    assert_eq!(estimate.dispatches, Some(1));
    for operation in [Operation::RightDivide, Operation::LeftDivide] {
        request.operation = ProviderOperationIdentity::new(identity(operation));
        let ProviderFeasibility::Supported { estimate } = feasibility(
            HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            &request,
        )
        .unwrap() else {
            panic!("direct broadcast division")
        };
        assert_eq!(estimate.transient_bytes, Some(88));
        assert_eq!(estimate.output_bytes, Some(48));
        assert_eq!(estimate.dispatches, Some(1));
    }
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
fn division_hook_broadcasts_without_replicating_inputs() {
    let provider = InProcessProvider::new();
    let left = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[2.0, 4.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let right = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 2.0, 4.0]),
            shape: &[1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let result = block_on(provider.elem_div(&left, &right)).unwrap();
    let downloaded = block_on(provider.download(&result)).unwrap();
    assert_eq!(downloaded.shape, vec![2, 3]);
    assert_eq!(downloaded.data, vec![2.0, 4.0, 1.0, 2.0, 0.5, 1.0]);
    for handle in [result, left, right] {
        handle.release().unwrap();
    }
}

#[test]
fn representation_and_shape_rejections_do_not_claim_hook_support() {
    let provider = InProcessProvider::new();
    let snapshot = snapshot(&provider);
    // Original mixed representations are admitted in either public operand
    // order, including native broadcast; publication is Single for all five
    // operations (the Double intermediate is not an input pre-cast).
    for &operation in OPERATIONS {
        for single_index in 0..2 {
            let mut request = query(operation);
            request.inputs[0].shape = vec![2, 1];
            request.inputs[1].shape = vec![1, 3];
            for input in &mut request.inputs {
                input.residency = ProviderResidency::Device;
            }
            request.inputs[single_index].element_type = ProviderElementType::F32;
            let mut output = representation(&[2, 3]);
            output.residency = ProviderResidency::Device;
            request.outputs = vec![output];
            assert!(!provider.query_feasibility(&request).is_supported());
            request.outputs[0].element_type = ProviderElementType::F32;
            assert!(provider.query_feasibility(&request).is_supported());
            request.inputs[single_index].layout = ProviderLayout::Opaque;
            assert!(!provider.query_feasibility(&request).is_supported());
        }
    }
    for element_type in [ProviderElementType::U64, ProviderElementType::ComplexF64] {
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
    let mut mixed = query(Operation::Add);
    mixed.inputs[0].element_type = ProviderElementType::F32;
    assert!(feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &mixed
    )
    .unwrap()
    .is_supported());
    let mut wrong_output = representation(&[1, 2]);
    wrong_output.residency = ProviderResidency::Device;
    mixed.outputs.push(wrong_output);
    assert!(
        !feasibility(
            HookProfile::InProcess,
            provider.precision(),
            &snapshot,
            &mixed
        )
        .unwrap()
        .is_supported(),
        "a mixed native result is Single, not Double"
    );
    mixed.outputs[0].element_type = ProviderElementType::F32;
    assert!(feasibility(
        HookProfile::InProcess,
        provider.precision(),
        &snapshot,
        &mixed
    )
    .unwrap()
    .is_supported());
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
