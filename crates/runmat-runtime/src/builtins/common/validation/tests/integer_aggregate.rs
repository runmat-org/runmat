use super::*;

fn structure_array(child: Value) -> Value {
    let mut element = StructValue::new();
    element.insert("Payload", child);
    let mut other = StructValue::new();
    other.insert("Payload", Value::Num(0.0));
    Value::StructArray(StructArray::new(vec![element, other], vec![1, 2]).unwrap())
}

#[test]
fn structure_array_classification_borrows_nested_resident_values() {
    use runmat_accelerate_api::{GpuHandleProvenance, NumericElementType};
    let allocation = resident_fixture(vec![1, 1], NumericElementType::U64);
    let handle = allocation
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    drop(allocation.handle);
    let input = structure_array(Value::Cell(
        CellArray::new(vec![Value::GpuTensor(handle)], 1, 1).unwrap(),
    ));
    assert!(value_contains_native_integer_class(&input));
    assert!(value_contains_explicit_gpu(&input));
    assert!(!native_integer_value_is_exact_f64(&input));
    assert_eq!(allocation.probe.retirements(), 0);
    drop(input);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn aggregate_exactness_retains_wide_integer_distinctions() {
    for (integer, exact) in [((1_u64 << 53) + 1, false), (1_u64 << 54, true)] {
        let input = structure_array(Value::Int(IntValue::U64(integer)));
        assert!(value_contains_native_integer_class(&input));
        assert!(!value_contains_explicit_gpu(&input));
        assert_eq!(native_integer_value_is_exact_f64(&input), exact);
        let mut object = runmat_value::ObjectInstance::new("ExactnessFixture".to_string());
        object.properties.insert("Payload".into(), input);
        assert_eq!(
            native_integer_value_is_exact_f64(&Value::Object(object)),
            exact
        );
    }
}
