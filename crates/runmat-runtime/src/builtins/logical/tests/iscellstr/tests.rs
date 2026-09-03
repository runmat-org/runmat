use super::*;
use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, Tensor};

fn run(value: Value) -> bool {
    match block_on(iscellstr_builtin(value)).expect("iscellstr") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn accepts_empty_cells_and_character_arrays_of_any_size() {
    assert!(run(Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())));
    let cell = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("red")),
            Value::CharArray(CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap()),
        ],
        1,
        2,
    )
    .unwrap();
    assert!(run(Value::Cell(cell)));
}

#[test]
fn rejects_noncells_and_cells_with_noncharacter_members() {
    assert!(!run(Value::String("red".into())));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap()
    )));
    let mixed = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("red")),
            Value::String("blue".into()),
        ],
        1,
        2,
    )
    .unwrap();
    assert!(!run(Value::Cell(mixed)));
}

#[test]
fn resident_numeric_values_are_not_cell_strings() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        assert!(!run(Value::GpuTensor(handle)));
    });
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(iscellstr_builtin(Value::Bool(true))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:iscellstr:TooManyOutputs"));
}
