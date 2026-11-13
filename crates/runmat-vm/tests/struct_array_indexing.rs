#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

fn structure_array<'a>(vars: &'a [Value], shape: &[usize]) -> &'a runmat_value::StructArray {
    vars.iter()
        .find_map(|value| match value {
            Value::StructArray(array) if array.shape() == shape => Some(array),
            _ => None,
        })
        .expect("expected structure array with requested shape")
}

#[test]
fn compiled_scalar_and_end_plus_one_assignment_grow_structure_arrays() {
    let vars =
        execute_source("s = struct('value', {1, 2}); s(end + 1) = struct('value', 3); out = s;")
            .unwrap();
    let array = structure_array(&vars, &[1, 3]);
    assert_eq!(
        array.get_linear(2).unwrap().fields.get("value"),
        Some(&Value::Num(3.0))
    );
}

#[test]
fn compiled_subscript_assignment_grows_and_preserves_column_major_coordinates() {
    let vars = execute_source(
        "s = reshape(struct('value', {1, 2, 3, 4}), 2, 2); s(3, 2) = struct('value', 9); out = s;",
    )
    .unwrap();
    let array = structure_array(&vars, &[3, 2]);
    assert_eq!(
        array.get_linear(2).unwrap().fields.get("value"),
        Some(&Value::Tensor(
            runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap()
        ))
    );
    assert_eq!(
        array.get_linear(5).unwrap().fields.get("value"),
        Some(&Value::Num(9.0))
    );
}

#[test]
fn compiled_indexing_collapses_trailing_dimensions() {
    let vars = execute_source(
        "s = reshape(struct('value', {1,2,3,4,5,6,7,8}), [2,2,2]); one = s(2,4); out = getfield(one, 'value');",
    )
    .unwrap();
    assert!(vars.iter().any(|value| value == &Value::Num(8.0)));
}

#[test]
fn compiled_row_and_linear_deletion_preserve_structure_array_shape() {
    let row_vars =
        execute_source("s = reshape(struct('value', {1,2,3,4,5,6}), 2, 3); s(1,:) = []; out = s;")
            .unwrap();
    structure_array(&row_vars, &[1, 3]);

    let linear_vars =
        execute_source("s = struct('value', {1,2,3,4}); s([2,4]) = []; out = s;").unwrap();
    structure_array(&linear_vars, &[1, 2]);
}
