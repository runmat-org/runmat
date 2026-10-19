use super::*;
use crate::builtins::table::table_from_columns;
use runmat_value::Tensor;

#[test]
fn matrix_columns_and_table_selectors_are_explicit_extensions() {
    let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap());
    let table = table_from_columns(
        vec!["A".into()],
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )],
    )
    .unwrap();
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let matrix_error = call(matrix, Vec::new()).unwrap_err();
    assert_eq!(
        matrix_error.identifier(),
        runmat_builtins::FINDGROUPS_MATRIX_COLUMNS_EXTENSION.error_identifier
    );
    let selector_error = call(table, vec![Value::from("A")]).unwrap_err();
    assert_eq!(
        selector_error.identifier(),
        runmat_builtins::FINDGROUPS_TABLE_SELECTOR_EXTENSION.error_identifier
    );
}
