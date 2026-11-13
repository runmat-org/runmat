use super::run;
use futures::executor::block_on;
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_zero_value_arrays_have_one_cell_but_an_empty_cell_array_stays_empty() {
    let Value::Cell(logical) = run(
        Value::LogicalArray(LogicalArray::new(vec![1], vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("rank-zero logical output expected")
    };
    assert_eq!(logical.shape, vec![1, 1]);
    assert_eq!(logical.data(), vec![Value::Bool(true)]);

    let Value::Cell(strings) = run(
        Value::StringArray(StringArray::new(vec!["only".into()], vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("rank-zero string output expected")
    };
    assert_eq!(strings.shape, vec![1, 1]);
    assert_eq!(strings.data(), vec![Value::String("only".into())]);

    let symbol = SymbolicExpr::constant(7.0);
    let Value::Cell(symbolic) = run(
        Value::SymbolicArray(SymbolicArray::new(vec![symbol.clone()], vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("rank-zero symbolic output expected")
    };
    assert_eq!(symbolic.shape, vec![1, 1]);
    assert_eq!(symbolic.data(), vec![Value::Symbolic(symbol)]);

    let empty = CellArray::new_with_shape(Vec::new(), vec![]).unwrap();
    let Value::Cell(output) = run(Value::Cell(empty), Vec::new()) else {
        panic!("empty cell output expected")
    };
    assert!(output.shape.is_empty());
    assert!(output.data().is_empty());
    assert_eq!((output.rows, output.cols), (0, 0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn empty_cell_shape_revalidates_payload_cardinality_and_cached_geometry() {
    let mut malformed =
        CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("valid original payload");
    malformed.shape = vec![].into();
    malformed.rows = 0;
    malformed.cols = 0;
    let error = block_on(super::super::num2cell_builtin(
        None,
        Value::Cell(malformed),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("malformed empty cell must not become one scalar cell");
    assert_eq!(error.identifier(), Some("RunMat:num2cell:Internal"));
    assert!(error.to_string().contains("Cell data length 1"));

    let mut stale_cache =
        CellArray::new_with_shape(Vec::new(), Vec::new()).expect("valid empty payload");
    stale_cache.rows = 7;
    stale_cache.cols = 8;
    let Value::Cell(output) = run(Value::Cell(stale_cache), Vec::new()) else {
        panic!("valid empty cell payload expected")
    };
    assert!(output.shape.is_empty());
    assert!(output.data().is_empty());
    assert_eq!((output.rows, output.cols), (0, 0));
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
