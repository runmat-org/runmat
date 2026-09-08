use super::*;
use runmat_value::{IntValue, IntegerStorage, Tensor};

#[test]
fn default_and_all_integer_dimension_classes_are_accepted() {
    let names = || Value::Cell(CellArray::new(vec![Value::from("value")], 1, 1).unwrap());
    let cell = || CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
    assert!(matches!(
        call(cell(), names(), None).unwrap(),
        Value::Struct(_)
    ));
    for dimension in [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(1),
    ] {
        assert!(matches!(
            call(cell(), names(), Some(Value::Int(dimension))).unwrap(),
            Value::Struct(_)
        ));
    }
}

#[test]
fn typed_scalar_tensor_dimension_is_accepted() {
    let cells = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
    let names = Value::from("value");
    let dimension = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();

    assert!(matches!(
        call(cells, names, Some(Value::Tensor(dimension))).unwrap(),
        Value::Struct(_)
    ));
}

#[test]
fn invalid_dimensions_and_shape_mismatch_are_structured() {
    let cells = || CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
    let names = || Value::Cell(CellArray::new(vec![Value::from("value")], 1, 1).unwrap());
    for dimension in [
        Value::Num(0.0),
        Value::Num(1.5),
        Value::Int(IntValue::U64(u64::MAX)),
    ] {
        assert_eq!(
            call(cells(), names(), Some(dimension))
                .unwrap_err()
                .identifier(),
            Some("RunMat:cell2struct:InvalidInput")
        );
    }
    let two_names =
        Value::Cell(CellArray::new(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap());
    assert_eq!(
        call(cells(), two_names, None).unwrap_err().identifier(),
        Some("RunMat:cell2struct:ShapeMismatch")
    );
}
