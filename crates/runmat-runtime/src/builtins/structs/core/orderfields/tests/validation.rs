use super::*;
use runmat_value::{CellArray, Tensor};

fn source() -> StructValue {
    structure(&[("a", 1.0.into()), ("b", 2.0.into())])
}

#[test]
fn rejects_unknown_and_duplicate_names() {
    let unknown = CellArray::new(vec![Value::from("a"), Value::from("c")], 1, 2).unwrap();
    assert_eq!(
        error_identifier(call(Value::Struct(source()), vec![Value::Cell(unknown)]).unwrap_err()),
        "orderfields:UnknownField"
    );
    let duplicate = CellArray::new(vec![Value::from("a"), Value::from("a")], 1, 2).unwrap();
    assert_eq!(
        error_identifier(call(Value::Struct(source()), vec![Value::Cell(duplicate)]).unwrap_err()),
        "orderfields:DuplicateField"
    );
}

#[test]
fn rejects_invalid_numeric_permutations() {
    let cases = [
        (vec![1.0, 1.5], "orderfields:IndexNotInteger"),
        (vec![1.0, 3.0], "orderfields:IndexOutOfRange"),
        (vec![1.0, 1.0], "orderfields:DuplicateIndex"),
    ];
    for (values, identifier) in cases {
        let order = Tensor::new(values, vec![1, 2]).unwrap();
        let error = call(Value::Struct(source()), vec![Value::Tensor(order)]).unwrap_err();
        assert_eq!(error_identifier(error), identifier);
    }
    let higher_rank = Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap();
    assert_eq!(
        error_identifier(
            call(Value::Struct(source()), vec![Value::Tensor(higher_rank)]).unwrap_err()
        ),
        "orderfields:InvalidPermutation"
    );
}

#[test]
fn rejects_invalid_target_order_and_extra_input() {
    assert_eq!(
        error_identifier(call(Value::Bool(true), Vec::new()).unwrap_err()),
        "orderfields:InvalidInput"
    );
    assert_eq!(
        error_identifier(call(Value::Struct(source()), vec![Value::Bool(true)]).unwrap_err()),
        "orderfields:InvalidOrderArgument"
    );
    assert_eq!(
        error_identifier(
            call(
                Value::Struct(source()),
                vec![Value::from("a"), Value::from("b")]
            )
            .unwrap_err()
        ),
        "orderfields:TooManyInputs"
    );
}
