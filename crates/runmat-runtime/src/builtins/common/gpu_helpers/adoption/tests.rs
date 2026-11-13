use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, NumericElementType, ProviderPrecision,
};

#[test]
fn checked_adoption_preserves_borrowed_view_and_assigns_automatic_intent() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let original = allocation.handle.clone();
    let value = resident_gpu_value(allocation.handle).unwrap();
    let Value::GpuTensor(adopted) = &value else {
        unreachable!()
    };
    assert!(original.shares_allocation_with(adopted));
    assert_ne!(original.view_identity(), adopted.view_identity());
    assert_eq!(runmat_accelerate_api::handle_provenance(&original), None);
    assert_eq!(
        runmat_accelerate_api::handle_provenance(adopted),
        Some(GpuHandleProvenance::Automatic)
    );
    drop(value);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(original);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn checked_adoption_rejects_relabeling_and_preserves_other_owners() {
    let owner = TestAllocationOwner::new();
    for complex in [false, true] {
        let allocation = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let borrowed = allocation.handle.clone();
        let result = if complex {
            complex_gpu_value(allocation.handle)
        } else {
            logical_gpu_value(allocation.handle)
        };
        assert_eq!(
            result.unwrap_err().gpu_gather_retry,
            crate::GpuGatherRetry::Never
        );
        assert_eq!(allocation.probe.retirements(), 0);
        assert!(!runmat_accelerate_api::handle_is_logical(&borrowed));
        assert_eq!(borrowed.physical().storage(), GpuTensorStorage::Real);
        drop(borrowed);
        assert_eq!(allocation.probe.retirements(), 1);
    }
    let logical = owner
        .allocate_logical_zeroed(vec![0, 2, 3], ProviderPrecision::F32)
        .unwrap();
    drop(logical_gpu_value(logical.handle).unwrap());
    assert_eq!(logical.probe.retirements(), 1);
    let complex = owner
        .allocate_zeroed(
            vec![2, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    drop(complex_gpu_value(complex.handle).unwrap());
    assert_eq!(complex.probe.retirements(), 1);

    let adopters: [fn(GpuTensorHandle) -> BuiltinResult<Value>; 4] = [
        resident_gpu_value,
        logical_gpu_value,
        complex_gpu_value,
        |output| propagate_owned_output_provenance(output, []).map(Value::GpuTensor),
    ];
    for prelabelled in [false, true] {
        for adopt in adopters {
            let allocation = owner
                .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
                .unwrap();
            let handle = if prelabelled {
                allocation
                    .handle
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap()
            } else {
                allocation.handle.clone()
            };
            drop(allocation.handle);
            allocation.probe.mark_unavailable();
            allocation.probe.fail_retirement("retained adoption owner");
            let error = adopt(handle).unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.message().contains("retained adoption owner"));
            assert!(std::error::Error::source(&error).is_some());
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }
}

#[test]
fn explicit_output_intent_is_a_distinct_view_not_input_mutation() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let explicit = allocation
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let mut output =
        propagate_owned_output_provenance(allocation.handle.clone(), [&explicit]).unwrap();
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(
        runmat_accelerate_api::handle_provenance(&allocation.handle),
        None
    );
    assert!(output.shares_allocation_with(&allocation.handle));
    let before = output.view_identity();
    propagate_output_provenance(&mut output, [&explicit]).unwrap();
    assert_eq!(output.view_identity(), before);
    drop(output);
    drop(explicit);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}
