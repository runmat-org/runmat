use super::*;
use runmat_value::{CharArray, SymbolicArray, SymbolicExpr};

#[test]
fn cell_input_moves_character_values_and_converts_string_scalars() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let source = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("left ")),
            Value::String("right".into()),
        ],
        1,
        2,
    )
    .unwrap();
    let output = cell(call(Value::Cell(source)).unwrap());
    assert_eq!(strings(&output), vec!["left ", "right"]);
}

#[test]
fn cell_and_symbolic_inputs_obey_compatibility_mode() {
    let cell_input = Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap());
    let error = call(cell_input).expect_err("cell extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:CellstrCellInputExtension")
    );
    let error = call(Value::Symbolic(SymbolicExpr::variable("x"))).expect_err("symbolic extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:CellstrSymbolicInputExtension")
    );
}

#[test]
fn symbolic_array_preserves_shape_and_order_in_runmat_mode() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = SymbolicArray::new(
        vec![
            SymbolicExpr::variable("a"),
            SymbolicExpr::variable("c"),
            SymbolicExpr::variable("b"),
            SymbolicExpr::variable("d"),
        ],
        vec![2, 2],
    )
    .unwrap();
    let output = cell(call(Value::SymbolicArray(input)).unwrap());
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(strings(&output), vec!["a", "b", "c", "d"]);
}
