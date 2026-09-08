use super::run;
use runmat_value::{
    CellArray, IntValue, IntegerStorage, LogicalArray, ObjectArray, ObjectInstance, SparseTensor,
    StringArray, SymbolicArray, SymbolicExpr, Value,
};

#[test]
fn cell_input_is_nested_and_payloads_keep_order() {
    let input = CellArray::from_column_major(
        vec![Value::String("a".into()), Value::String("b".into())],
        vec![1, 2],
    )
    .unwrap();
    let Value::Cell(output) = run(Value::Cell(input), Vec::new()) else {
        panic!("expected cell");
    };
    let Value::Cell(first) = output.get(0, 0).unwrap() else {
        panic!("cell elements must remain nested cells");
    };
    assert_eq!(first.get(0, 0).unwrap(), Value::String("a".into()));
}

#[test]
fn string_symbolic_and_logical_arrays_preserve_element_kind() {
    let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
    let Value::Cell(strings) = run(Value::StringArray(strings), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(strings.get(0, 1).unwrap(), Value::String("b".into()));

    let symbolic = SymbolicArray::new(
        vec![SymbolicExpr::constant(1.0), SymbolicExpr::constant(2.0)],
        vec![1, 2],
    )
    .unwrap();
    let Value::Cell(symbolic) = run(Value::SymbolicArray(symbolic), Vec::new()) else {
        panic!("expected cell");
    };
    assert!(matches!(symbolic.get(0, 0).unwrap(), Value::Symbolic(_)));

    let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
    let Value::Cell(logical) = run(Value::LogicalArray(logical), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(logical.get(0, 1).unwrap(), Value::Bool(false));
}

#[test]
fn sparse_input_materializes_dense_cells_without_changing_class() {
    let sparse = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        IntegerStorage::U16(vec![7, 9]),
    )
    .unwrap();
    let Value::Cell(output) = run(Value::SparseTensor(sparse), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(output.get(0, 0).unwrap(), Value::Int(IntValue::U16(7)));
    assert_eq!(output.get(1, 0).unwrap(), Value::Int(IntValue::U16(0)));
}

#[test]
fn grouped_object_arrays_retain_class_identity() {
    let class = "tests.Widget";
    let input = ObjectArray::from_objects(
        class,
        vec![ObjectInstance::new(class), ObjectInstance::new(class)],
        vec![1, 2],
    )
    .unwrap();
    let Value::Cell(output) = run(Value::ObjectArray(input), vec![Value::Num(2.0)]) else {
        panic!("expected cell");
    };
    let Value::ObjectArray(block) = output.get(0, 0).unwrap() else {
        panic!("expected grouped object array");
    };
    assert_eq!(block.class_name().display_name(), class);
    assert_eq!(block.shape(), &[1, 2]);
}
