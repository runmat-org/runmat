use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage, NumericElementType,
};

#[test]
fn owned_object_array_projection_preserves_class_shape_and_source_aliases() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut object = runmat_value::ObjectInstance::new("ProvenanceValue");
    object
        .properties
        .insert("data".into(), Value::GpuTensor(allocation.handle.clone()));
    let array =
        runmat_value::ObjectArray::from_objects("ProvenanceValue", vec![object], vec![1, 1, 1])
            .unwrap();
    let source = Value::ObjectArray(array);
    let mut result = source.clone();
    let explicit = allocation
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    propagate_gpu_provenance(&[Value::GpuTensor(explicit)], &mut result).unwrap();
    let Value::ObjectArray(projected) = &result else {
        panic!("object array")
    };
    assert_eq!(projected.shape(), &[1, 1, 1]);
    assert_eq!(projected.class_name().display_name(), "ProvenanceValue");
    let mut results = Vec::new();
    try_visit_gpu_handles(&result, &mut |handle| results.push(handle.clone())).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].shares_allocation_with(&allocation.handle));
    assert_eq!(
        results[0].view().provenance(),
        Some(GpuHandleProvenance::Explicit)
    );
    try_visit_gpu_handles(&source, &mut |handle| {
        assert_eq!(handle.view().provenance(), None)
    })
    .unwrap();
    assert_eq!(allocation.handle.view().provenance(), None);
}

#[test]
fn managed_results_preserve_stored_intent_without_borrowing_or_mutating_referents() {
    runmat_gc::gc_test_context(|| {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let stored = allocation
            .handle
            .with_provenance(GpuHandleProvenance::Automatic)
            .unwrap();
        let target = runmat_gc::gc_allocate(Value::GpuTensor(stored)).unwrap();
        let handle = runmat_value::HandleRef {
            class_name: "ProvenanceManaged".into(),
            target,
            valid: true,
        };
        let listener = runmat_value::Listener {
            id: 1,
            target,
            target_class_name: "ProvenanceManaged".into(),
            event_name: "changed".into(),
            callback: target,
            enabled: true,
            valid: true,
        };
        let mut results = [
            Value::HandleObject(handle.clone()),
            Value::ObjectArray(
                runmat_value::ObjectArray::from_handles(
                    "ProvenanceManaged",
                    vec![handle.clone(), handle],
                    vec![1, 2],
                )
                .unwrap(),
            ),
            Value::Listener(listener),
        ];
        let explicit = allocation
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        runmat_gc::gc_with_value_mut(&target, |stored| {
            for result in &mut results {
                propagate_gpu_provenance(&[Value::GpuTensor(explicit.clone())], result).unwrap();
            }
            let Value::GpuTensor(stored) = stored else {
                panic!("stored GPU value")
            };
            assert_eq!(
                stored.view().provenance(),
                Some(GpuHandleProvenance::Automatic)
            );
            assert!(stored.shares_allocation_with(&allocation.handle));
        })
        .unwrap();
        for result in &results {
            try_visit_gpu_handles(result, &mut |handle| {
                assert_eq!(
                    handle.view().provenance(),
                    Some(GpuHandleProvenance::Automatic)
                );
            })
            .unwrap();
        }
    });
}

#[test]
fn result_projection_preserves_source_view_and_allocation_lifetime() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::U64, GpuTensorStorage::Real)
        .unwrap();
    let source = allocation
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let mut result = Value::GpuTensor(allocation.handle.clone());
    propagate_gpu_provenance(&[Value::GpuTensor(source)], &mut result).unwrap();
    let Value::GpuTensor(projected) = result else {
        unreachable!()
    };
    assert!(projected.shares_allocation_with(&allocation.handle));
    assert_eq!(
        projected.view().provenance(),
        Some(GpuHandleProvenance::Explicit)
    );
    assert_eq!(allocation.handle.view().provenance(), None);
    assert_eq!(projected.shape(), &[2, 1]);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 0);
    projected.allocation_lease().validate_live().unwrap();
    drop(projected);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn explicit_producer_intent_survives_without_resident_arguments() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![0, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut result = Value::GpuTensor(
        allocation
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    propagate_gpu_provenance(&[], &mut result).unwrap();
    let Value::GpuTensor(projected) = result else {
        unreachable!()
    };
    assert_eq!(
        projected.view().provenance(),
        Some(GpuHandleProvenance::Explicit)
    );
    assert_eq!(projected.shape(), &[0, 3]);
}
