use super::*;
use runmat_value::CellArray;

#[test]
fn represented_array_preserves_shape_and_reorders_every_element() {
    let values = vec![
        Value::Struct(structure(&[("b", 1.0.into()), ("a", 2.0.into())])),
        Value::Struct(structure(&[("a", 3.0.into()), ("b", 4.0.into())])),
    ];
    let input = CellArray::new_with_shape(values, vec![1, 1, 2]).unwrap();
    let Value::Cell(output) = call(Value::Cell(input), Vec::new()).unwrap() else {
        panic!("expected represented array")
    };
    assert_eq!(output.shape, [1, 1, 2]);
    for value in output.data {
        let Value::Struct(structure) = value else {
            panic!("expected structure element")
        };
        assert_eq!(field_order(&structure), ["a", "b"]);
    }
}

#[test]
fn represented_array_requires_one_field_schema() {
    let input = CellArray::new(
        vec![
            Value::Struct(structure(&[("a", 1.0.into()), ("b", 2.0.into())])),
            Value::Struct(structure(&[("a", 3.0.into()), ("c", 4.0.into())])),
        ],
        1,
        2,
    )
    .unwrap();
    let error = call(Value::Cell(input), Vec::new()).unwrap_err();
    assert_eq!(error_identifier(error), "orderfields:FieldMismatch");
}

#[test]
fn empty_represented_array_accepts_only_empty_order() {
    let empty = || Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap());
    assert!(call(empty(), Vec::new()).is_ok());
    assert!(call(
        empty(),
        vec![Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())]
    )
    .is_ok());
    let reference = structure(&[("field", 1.0.into())]);
    assert_eq!(
        error_identifier(call(empty(), vec![Value::Struct(reference)]).unwrap_err()),
        "orderfields:EmptyStructArray"
    );
}
