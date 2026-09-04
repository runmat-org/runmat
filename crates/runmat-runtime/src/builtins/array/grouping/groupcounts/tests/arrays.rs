use super::*;
use runmat_value::{CellArray, IntegerStorage, LogicalArray, StringArray, Tensor};

#[test]
fn returns_counts_labels_and_percentages() {
    let input = Value::Tensor(Tensor::new(vec![2.0, 1.0, 2.0, 2.0], vec![4, 1]).unwrap());
    let _outputs = crate::output_count::push_output_count(Some(3));
    let outputs = output_list(call(input, Vec::new()).unwrap());
    assert!(matches!(&outputs[0], Value::Tensor(value) if value.materialize_f64() == [1.0, 3.0]));
    assert!(matches!(&outputs[1], Value::Tensor(value) if value.materialize_f64() == [1.0, 2.0]));
    assert!(matches!(&outputs[2], Value::Tensor(value) if value.materialize_f64() == [25.0, 75.0]));
}

#[test]
fn includes_missing_group_by_default() {
    let input = Value::Tensor(Tensor::new(vec![2.0, f64::NAN, 2.0], vec![3, 1]).unwrap());
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(call(input, Vec::new()).unwrap());
    assert!(matches!(&outputs[0], Value::Tensor(value) if value.materialize_f64() == [2.0, 1.0]));
    let Value::Tensor(labels) = &outputs[1] else {
        panic!("expected numeric labels");
    };
    assert_eq!(labels.materialize_f64()[0], 2.0);
    assert!(labels.materialize_f64()[1].is_nan());
}

#[test]
fn preserves_exact_wide_integer_labels() {
    let base = 1_u64 << 53;
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![base, base + 1, base]), vec![3, 1]).unwrap(),
    );
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(call(input, Vec::new()).unwrap());
    assert!(
        matches!(&outputs[1], Value::Tensor(value) if value.integer_storage() == Some(&IntegerStorage::U64(vec![base, base + 1])))
    );
}

#[test]
fn multiple_grouping_vectors_return_one_label_column_each() {
    let groups = Value::Cell(
        CellArray::new(
            vec![
                Value::StringArray(
                    StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
                ),
                Value::LogicalArray(LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(call(groups, Vec::new()).unwrap());
    let Value::Cell(labels) = &outputs[1] else {
        panic!("expected one label vector per grouping role");
    };
    assert_eq!((labels.rows, labels.cols), (1, 2));
    assert!(matches!(&labels.data[0], Value::StringArray(values) if values.data == ["a", "b"]));
    assert!(
        matches!(&labels.data[1], Value::LogicalArray(values) if values.data.as_slice() == [0, 1])
    );
}

#[test]
fn logical_empty_groups_use_closed_value_domain() {
    let input = Value::LogicalArray(LogicalArray::new(vec![1, 1], vec![2, 1]).unwrap());
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(
        call(
            input,
            vec![Value::from("IncludeEmptyGroups"), Value::Bool(true)],
        )
        .unwrap(),
    );
    assert!(matches!(&outputs[0], Value::Tensor(value) if value.materialize_f64() == [0.0, 2.0]));
    assert!(matches!(&outputs[1], Value::LogicalArray(value) if value.data.as_slice() == [0, 1]));
}
