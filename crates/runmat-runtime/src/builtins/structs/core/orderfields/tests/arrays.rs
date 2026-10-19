use super::*;
use runmat_value::{CellArray, StructArray};

#[test]
fn typed_array_preserves_shape_and_reorders_every_element() {
    let values = vec![
        structure(&[("b", 1.0.into()), ("a", 2.0.into())]),
        structure(&[("b", 3.0.into()), ("a", 4.0.into())]),
    ];
    let input = StructArray::new(values, vec![1, 1, 2]).unwrap();
    let Value::StructArray(output) = call(Value::StructArray(input), Vec::new()).unwrap() else {
        panic!("expected structure array")
    };
    assert_eq!(output.shape(), [1, 1, 2]);
    for structure in output.elements() {
        assert_eq!(
            structure
                .fields
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    }
}

#[test]
fn ordinary_cell_of_structs_is_rejected() {
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
    assert_eq!(error_identifier(error), "orderfields:InvalidInput");
}

#[test]
fn empty_typed_array_uses_explicit_schema() {
    let empty = || Value::StructArray(StructArray::empty(Vec::new(), vec![0, 0]).unwrap());
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
