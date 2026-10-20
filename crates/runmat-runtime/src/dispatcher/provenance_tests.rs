use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage, NumericElementType,
};

#[test]
fn managed_shared_cell_provenance_admits_outer_and_text_without_copying_numeric_payload() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
    use runmat_value::{CellArray, IntegerStorage, SparseTensor, Tensor};
    use std::rc::Rc;

    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
    let _scope = context.enter();
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let tensor = Tensor::new(vec![9.0], vec![1, 1]).unwrap();
    let numeric_ptr = tensor.as_f64_slice().unwrap().as_ptr();
    let sparse = SparseTensor::new_integer(
        1,
        1,
        vec![0, 1],
        vec![0],
        IntegerStorage::U64(vec![u64::MAX]),
    )
    .unwrap();
    let sparse_indices = sparse.row_indices.as_ptr();
    let sparse_values = match sparse.numeric_host_buffer().unwrap().integer_storage() {
        Some(IntegerStorage::U64(values)) => values.as_ptr(),
        _ => panic!("uint64 sparse storage"),
    };
    let source = Value::Cell(
        CellArray::new_with_shape(
            vec![
                Value::string("owned text"),
                Value::Tensor(tensor),
                Value::Cell(
                    CellArray::new(vec![Value::GpuTensor(allocation.handle.clone())], 1, 1)
                        .unwrap(),
                ),
                Value::SparseTensor(sparse),
            ],
            vec![1, 4, 1],
        )
        .unwrap(),
    );
    let mut projected = source.clone();
    provenance::project(&mut projected, true).unwrap();
    let Value::Cell(cell) = &projected else {
        panic!("projected Cell");
    };
    assert!(cell.data_owner().is_exact());
    assert!(cell.shape.is_admitted());
    assert!(matches!(&cell.data()[0], Value::String(text) if text.has_retained_claim()));
    assert!(
        matches!(&cell.data()[1], Value::Tensor(value) if value.as_f64_slice().unwrap().as_ptr() == numeric_ptr)
    );
    assert!(matches!(&cell.data()[2], Value::Cell(nested) if nested.data_owner().is_exact()));
    let Value::SparseTensor(sparse) = &cell.data()[3] else {
        panic!("shared sparse child");
    };
    assert_eq!(sparse.row_indices.as_ptr(), sparse_indices);
    assert!(matches!(
        sparse.numeric_host_buffer().unwrap().integer_storage(),
        Some(IntegerStorage::U64(values)) if values.as_ptr() == sparse_values
    ));
    assert_eq!(
        sparse.numeric_host_buffer().unwrap().value_at(0),
        Some(runmat_value::NumericScalar::U64(u64::MAX))
    );
    try_visit_gpu_handles(&projected, &mut |handle| {
        assert_eq!(
            handle.view().provenance(),
            Some(GpuHandleProvenance::Explicit)
        );
    })
    .unwrap();
    try_visit_gpu_handles(&source, &mut |handle| {
        assert_eq!(handle.view().provenance(), None);
    })
    .unwrap();
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(projected);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn owned_object_array_projection_preserves_class_shape_and_source_aliases() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut object = runmat_value::ObjectInstance::new("ProvenanceValue");
    object.replace_property_unbounded("data".into(), Value::GpuTensor(allocation.handle.clone()));
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

    // Same-named old/new class products are distinct owners even after all
    // elements have been deleted. A no-work gather must retain both exact Rc
    // identities and the independent unselected-origin marker.
    use runmat_value::{
        BoundFunctionBinding, FunctionDefinitionOwner, LocalExport, ObjectArray,
        SelectedObjectClassFacts, SelectedReceiverClassDeclaration,
        SelectedReceiverDeclarationSeal, SelectedReceiverDeclarations,
    };
    use std::{collections::HashMap, rc::Rc, sync::Arc};
    let class = ClassIdentity::new("ProvenanceValue").unwrap();
    let issuer = Arc::new(());
    let selected = || {
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let declarations = Arc::new(
            SelectedReceiverDeclarations::new(
                class.clone(),
                false,
                vec![class.clone()],
                HashMap::from([(
                    class.clone(),
                    SelectedReceiverClassDeclaration {
                        parent: None,
                        properties: HashMap::new(),
                        events: HashMap::new(),
                    },
                )]),
                HashMap::new(),
            )
            .unwrap(),
        );
        let (seal, key) = SelectedReceiverDeclarationSeal::issue();
        assert!(definition.admit_selected_receiver_seal(Arc::clone(&seal)));
        assert!(seal.fill(&key, Arc::clone(&declarations)));
        Rc::new(
            SelectedObjectClassFacts::new(
                declarations,
                BoundFunctionBinding::local(
                    1,
                    Arc::downgrade(&issuer),
                    FunctionDefinitionOwner::issue(&definition, 1),
                    Vec::new(),
                    LocalExport::Ephemeral,
                ),
                HashMap::new(),
            )
            .unwrap(),
        )
    };
    let old = selected();
    let new = selected();
    assert!(!Rc::ptr_eq(&old, &new));
    for unselected in [false, true] {
        let mut empty = ObjectArray::empty(class.clone(), vec![0, 2, 3])
            .unwrap()
            .with_selected_provenance_unbounded([old.clone(), new.clone()])
            .unwrap();
        if unselected {
            empty = empty.with_unselected_provenance();
        }
        let source = Value::ObjectArray(empty);
        let mut projected = source.clone();
        propagate_gpu_provenance(&[], &mut projected).unwrap();
        for value in [
            projected,
            futures::executor::block_on(gather_if_needed_async(&source)).unwrap(),
        ] {
            let Value::ObjectArray(array) = value else {
                panic!("empty object array")
            };
            assert_eq!(array.class_name(), &class);
            assert_eq!(array.shape(), &[0, 2, 3]);
            assert_eq!(array.has_unselected_provenance(), unselected);
            assert_eq!(array.selected_provenance().len(), 2);
            assert!(Rc::ptr_eq(&array.selected_provenance()[0], &old));
            assert!(Rc::ptr_eq(&array.selected_provenance()[1], &new));
        }
    }
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
        let closure = Value::Closure(runmat_value::Closure::new(
            "managed_capture".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![Value::HandleObject(handle.clone())],
            None,
        ));
        let closure_instance = closure.callable_instance_key();
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
            closure,
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
        assert_eq!(results[3].callable_instance_key(), closure_instance);
        let gathered = futures::executor::block_on(gather_if_needed_async(&results[3]))
            .expect("managed referent is not an owned capture to gather");
        assert_eq!(gathered.callable_instance_key(), closure_instance);
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
