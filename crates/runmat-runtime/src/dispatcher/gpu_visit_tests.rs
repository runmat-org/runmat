use super::*;

#[test]
fn visits_object_arrays_and_cyclic_handle_properties_once() {
    runmat_gc::gc_test_context(|| {
        let device = Value::GpuTensor(GpuTensorHandle::new(vec![1, 1], 72, 8));
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
        visit_gpu_handles(&value, &mut |handle| visited.push(handle.buffer_id));
        assert_eq!(visited, vec![8]);
        assert!(value_contains_gpu(&value));
    });
}
