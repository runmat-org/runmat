use super::{value_dimensions, value_numel};
use futures::executor::block_on;
use runmat_types::{standard, StaticClassIdentity};
use runmat_value::{ObjectInstance, Tensor, Value};

fn wrapper(class: StaticClassIdentity, property: &str, payload: Value) -> Value {
    let mut object = ObjectInstance::new(class);
    object.replace_property_unbounded(property.to_owned(), payload);
    Value::Object(object)
}

#[test]
fn declared_array_wrappers_expose_payload_geometry_without_class_registration() {
    for (class, property) in [
        (standard::DURATION, runmat_builtins::DURATION_DAYS_PROPERTY),
        (
            standard::DATETIME,
            runmat_builtins::DATETIME_SERIAL_PROPERTY,
        ),
        (standard::DL_ARRAY, runmat_builtins::DLARRAY_DATA_PROPERTY),
    ] {
        for shape in [vec![1, 2], vec![2, 3, 4], vec![0, 3], vec![1, 1]] {
            let count = shape.iter().product();
            let tensor = Tensor::new(vec![0.0; count], shape.clone()).unwrap();
            let object = wrapper(class, property, Value::Tensor(tensor));
            assert_eq!(block_on(value_dimensions(&object)).unwrap(), shape);
            assert_eq!(block_on(value_numel(&object)).unwrap(), count);
        }
    }
}

#[test]
fn unrelated_scalar_objects_do_not_inherit_payload_geometry() {
    let object = wrapper(
        StaticClassIdentity::new("example.Scalar"),
        runmat_builtins::DURATION_DAYS_PROPERTY,
        Value::Tensor(Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap()),
    );
    assert_eq!(block_on(value_dimensions(&object)).unwrap(), vec![1, 1]);
    assert_eq!(block_on(value_numel(&object)).unwrap(), 1);
}

#[test]
fn declared_payload_cannot_silently_default_when_missing() {
    let value = Value::Object(ObjectInstance::new(standard::DURATION));
    let error = block_on(value_dimensions(&value)).unwrap_err();
    assert!(error
        .message()
        .contains("array payload property __days is missing"));
}

#[test]
fn malformed_wrapper_storage_is_not_a_scalar_array() {
    for (class, property, payload) in [
        (
            standard::DATETIME,
            runmat_builtins::DATETIME_SERIAL_PROPERTY,
            Value::String("invalid".into()),
        ),
        (
            standard::DURATION,
            runmat_builtins::DURATION_DAYS_PROPERTY,
            Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        ),
        (
            standard::DL_ARRAY,
            runmat_builtins::DLARRAY_DATA_PROPERTY,
            Value::Object(ObjectInstance::new(StaticClassIdentity::new(
                "example.Scalar",
            ))),
        ),
    ] {
        let value = wrapper(class, property, payload);
        assert!(block_on(value_dimensions(&value))
            .unwrap_err()
            .message()
            .contains("invalid storage"));
    }
}

#[test]
fn parallel_payload_dimensions_must_agree() {
    let mut object = ObjectInstance::new(standard::CALENDAR_DURATION);
    object.replace_property_unbounded(
        runmat_builtins::CALENDAR_MONTHS_PROPERTY.into(),
        Value::Tensor(Tensor::new(vec![0.0; 2], vec![1, 2]).unwrap()),
    );
    object.replace_property_unbounded(
        runmat_builtins::CALENDAR_DAYS_PROPERTY.into(),
        Value::Tensor(Tensor::new(vec![0.0; 2], vec![1, 2]).unwrap()),
    );
    assert_eq!(
        block_on(value_dimensions(&Value::Object(object.clone()))).unwrap(),
        vec![1, 2]
    );
    object.replace_property_unbounded(
        runmat_builtins::CALENDAR_DAYS_PROPERTY.into(),
        Value::Num(0.0),
    );
    let error = block_on(value_dimensions(&Value::Object(object))).unwrap_err();
    assert!(error.message().contains("inconsistent dimensions"));
}

#[test]
fn resident_payload_shape_query_borrows_allocation_metadata() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(
            vec![2, 3, 4],
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let probe = allocation.probe;
    let value = wrapper(
        standard::DL_ARRAY,
        runmat_builtins::DLARRAY_DATA_PROPERTY,
        Value::GpuTensor(allocation.handle),
    );
    assert_eq!(block_on(value_dimensions(&value)).unwrap(), vec![2, 3, 4]);
    assert_eq!(block_on(value_numel(&value)).unwrap(), 24);
    assert_eq!(probe.retirements(), 0);
    drop(value);
    assert_eq!(probe.retirements(), 1);
}
