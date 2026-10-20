use super::*;

#[test]
fn gather_traversal_matches_recursive_materialization() {
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_value::{CellArray, Closure, IntegerStorage, ObjectInstance, StructArray, Tensor};
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![2, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let resident = Value::GpuTensor(handle.clone());
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("data", resident.clone());
        let mut object = ObjectInstance::new("GatherFixture");
        object.replace_property_unbounded("data".into(), resident.clone());
        let structure = structure.finish();
        let inputs = vec![
            Value::Cell(CellArray::new_with_shape(vec![resident.clone()], vec![1, 1]).unwrap()),
            Value::Struct(structure.clone()),
            Value::StructArray(
                StructArray::new(vec![structure.clone(), structure], vec![1, 2]).unwrap(),
            ),
            Value::Object(object),
            Value::Closure(Closure::new(
                "fixture".into(),
                None,
                runmat_types::FunctionSourceRole::Anonymous,
                None,
                vec![resident],
                None,
            )),
        ];
        for (input, count) in inputs.into_iter().zip([1, 1, 2, 1, 1]) {
            let mut handles = Vec::new();
            try_visit_gathered_gpu_handles(&input, &mut |h| handles.push(h.buffer_id())).unwrap();
            assert_eq!(handles, vec![handle.buffer_id(); count]);
            let output = futures::executor::block_on(gather_if_needed_async(&input)).unwrap();
            let mut remaining = 0;
            try_visit_gathered_gpu_handles(&output, &mut |_| remaining += 1).unwrap();
            assert_eq!(remaining, 0);
            assert_eq!(
                futures::executor::block_on(gather_if_needed_async(&output)).unwrap(),
                output
            );
        }
        handle.release().unwrap();
    });
}

#[test]
fn gather_traversal_preserves_passthrough_identity_and_rejects_transient_sequences() {
    runmat_gc::gc_test_context(|| {
        let allocation_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocation = allocation_owner
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine resident allocation");
        let target = runmat_gc::gc_allocate(Value::GpuTensor(allocation.handle)).unwrap();
        let handle = Value::HandleObject(runmat_value::HandleRef {
            class_name: "TraversalFixture".into(),
            target,
            valid: true,
        });
        let array = Value::ObjectArray(
            runmat_value::ObjectArray::new("TraversalFixture", vec![handle.clone()], vec![1, 1])
                .unwrap(),
        );
        runmat_gc::gc_with_value_mut(&target, |_| {
            for input in [&handle, &array] {
                let mut visited = 0;
                try_visit_gathered_gpu_handles(input, &mut |_| visited += 1).unwrap();
                assert_eq!(visited, 0);
                assert_eq!(
                    futures::executor::block_on(gather_if_needed_async(input)).unwrap(),
                    *input
                );
            }
        })
        .unwrap();
        let transient = Value::output_list(vec![Value::Num(1.0)]);
        let admission = try_visit_gathered_gpu_handles(&transient, &mut |_| {}).unwrap_err();
        let runtime = futures::executor::block_on(gather_if_needed_async(&transient)).unwrap_err();
        assert_eq!(admission.identifier(), runtime.identifier());
    });
}

#[test]
fn live_borrow_conflict_is_not_absence_of_device_aliases() {
    runmat_gc::gc_test_context(|| {
        let target = runmat_gc::gc_allocate(Value::Num(1.)).unwrap();
        let value = Value::HandleObject(runmat_value::HandleRef {
            class_name: "TraversalFixture".into(),
            target,
            valid: true,
        });
        runmat_gc::gc_with_value_mut(&target, |_| {
            assert!(try_visit_gpu_handles(&value, &mut |_| {}).is_err());
            assert!(value_contains_gpu(&value));
        })
        .unwrap();
        assert!(try_visit_gpu_handles(&value, &mut |_| {}).is_ok());
    });
}

#[test]
fn visits_object_arrays_and_cyclic_handle_properties_once() {
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let expected_id = allocation.handle.buffer_id();
        let device = Value::GpuTensor(allocation.handle);
        let mut object = runmat_value::ObjectInstance::new("TraversalFixture");
        object.replace_property_unbounded("data".into(), device);
        let target = runmat_gc::gc_allocate(Value::Object(object)).unwrap();
        let handle = runmat_value::HandleRef {
            class_name: "TraversalFixture".into(),
            target,
            valid: true,
        };
        runmat_gc::gc_with_value_mut(&target, |value| {
            let Value::Object(object) = value else {
                panic!("object")
            };
            object.replace_property_unbounded("self".into(), Value::HandleObject(handle.clone()));
        })
        .unwrap();
        let value = Value::ObjectArray(
            runmat_value::ObjectArray::new(
                "TraversalFixture",
                vec![
                    Value::HandleObject(handle.clone()),
                    Value::HandleObject(handle),
                ],
                vec![1, 2],
            )
            .unwrap(),
        );
        let mut visited = Vec::new();
        visit_gpu_handles(&value, &mut |handle| visited.push(handle.buffer_id()));
        assert_eq!(visited, vec![expected_id]);
        assert!(value_contains_gpu(&value));
    });
}
