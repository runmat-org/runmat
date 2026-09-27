use super::*;

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
        object.properties.insert("data".into(), device);
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
            object
                .properties
                .insert("self".into(), Value::HandleObject(handle.clone()));
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
