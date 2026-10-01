use super::*;

#[test]
fn tabulate_expands_positive_integer_levels() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 3.0], vec![1, 3]).unwrap());
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![3, 3]);
    assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&data[3..6], &[1.0, 0.0, 2.0]);
    assert_close(data[6], 100.0 / 3.0);
    assert_close(data[7], 0.0);
    assert_close(data[8], 200.0 / 3.0);
}

#[test]
fn tabulate_numeric_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = int_tensor(IntegerStorage::U8(vec![1, 3, 3]), vec![1, 3]);
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected exact integer cell table");
    };
    assert_eq!((cell.rows, cell.cols), (3, 3));
    assert_eq!(cell.get(0, 0).unwrap(), Value::Int(IntValue::U8(1)));
    assert_eq!(cell.get(1, 0).unwrap(), Value::Int(IntValue::U8(2)));
    assert_eq!(cell.get(2, 0).unwrap(), Value::Int(IntValue::U8(3)));
    assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
    assert_eq!(cell.get(1, 1).unwrap(), Value::Num(0.0));
    assert_eq!(cell.get(2, 1).unwrap(), Value::Num(2.0));
}

#[test]
fn tabulate_integer_extension_preserves_wide_distinct_values() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = 1_u64 << 53;
    let x = int_tensor(IntegerStorage::U64(vec![base, base + 1, base]), vec![1, 3]);
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected exact integer cell table");
    };
    assert_eq!((cell.rows, cell.cols), (2, 3));
    assert_eq!(cell.get(0, 0).unwrap(), Value::Int(IntValue::U64(base)));
    assert_eq!(cell.get(1, 0).unwrap(), Value::Int(IntValue::U64(base + 1)));
    assert_eq!(cell.get(0, 1).unwrap(), Value::Num(2.0));
    assert_eq!(cell.get(1, 1).unwrap(), Value::Num(1.0));
}

#[test]
fn tabulate_numeric_omits_nans_from_counts_and_percent() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 3.0, f64::NAN], vec![1, 4]).unwrap());
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![3, 3]);
    assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&data[3..6], &[1.0, 0.0, 2.0]);
    assert_close(data[6], 100.0 / 3.0);
    assert_close(data[7], 0.0);
    assert_close(data[8], 200.0 / 3.0);
}

#[test]
fn tabulate_logical_returns_cell_table() {
    let x =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap());
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    match out {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 0).unwrap(), Value::Bool(false));
            assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Bool(true));
            assert_eq!(cell.get(1, 1).unwrap(), Value::Num(2.0));
        }
        other => panic!("expected cell table, got {other:?}"),
    }
}

#[test]
fn tabulate_string_array_returns_cell_table() {
    let x = Value::StringArray(
        StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![1, 3]).unwrap(),
    );
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    match out {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(1, 1).unwrap(), Value::Num(2.0));
        }
        other => panic!("expected cell table, got {other:?}"),
    }
}
