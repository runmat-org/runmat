use super::run;
use runmat_value::{CellArray, CharArray, StructValue, Value};

#[test]
fn invalid_top_level_name_type_uses_the_owned_error() {
    let error = run(Value::Struct(StructValue::new()), Value::Num(1.0)).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::ISFIELD_ERROR_FIELD_NAME_TYPE.identifier
    );
}

#[test]
fn character_matrix_is_not_a_field_name() {
    let matrix = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
    let error = run(Value::Struct(StructValue::new()), Value::CharArray(matrix)).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::ISFIELD_ERROR_FIELD_NAME_TYPE.identifier
    );
}

#[test]
fn invalid_cell_element_uses_the_cell_error() {
    let names = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
    let error = run(Value::Struct(StructValue::new()), Value::Cell(names)).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::ISFIELD_ERROR_CELL_ELEMENT_TYPE.identifier
    );
}
