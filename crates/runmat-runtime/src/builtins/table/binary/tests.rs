use runmat_value::{IntegerStorage, StringArray, Tensor, Value};

use crate::builtins::common::binary::BinaryInputPlan;

fn table(names: &[&str], columns: &[&[f64]]) -> Value {
    super::super::table_from_columns(
        names.iter().map(|name| (*name).to_string()).collect(),
        columns
            .iter()
            .map(|values| {
                Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
            })
            .collect(),
    )
    .unwrap()
}

fn set_row_names(value: &mut Value, names: &[&str]) {
    let Value::Object(object) = value else {
        panic!("expected table")
    };
    let mut properties = super::super::table_public_properties(object).unwrap();
    properties.insert(
        super::super::ROW_NAMES,
        Value::StringArray(
            StringArray::new(
                names.iter().map(|name| (*name).to_string()).collect(),
                vec![names.len(), 1],
            )
            .unwrap(),
        ),
    );
    super::super::sync_table_properties(object, properties);
}

fn timetable(values: &[f64], row_times: Vec<u64>) -> Value {
    let mut value = table(&["A"], &[values]);
    let Value::Object(object) = &mut value else {
        panic!("expected table")
    };
    object.class_name = super::super::TIMETABLE_CLASS.owned();
    let times = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(row_times), vec![values.len(), 1]).unwrap(),
    );
    super::super::set_timetable_row_times(object, Some(times)).unwrap();
    value
}

#[test]
fn aligns_variable_and_named_row_order_to_the_left_operand() {
    let mut left = table(&["A", "B"], &[&[1.0, 2.0], &[3.0, 4.0]]);
    let mut right = table(&["B", "A"], &[&[40.0, 30.0], &[20.0, 10.0]]);
    set_row_names(&mut left, &["first", "second"]);
    set_row_names(&mut right, &["second", "first"]);

    let BinaryInputPlan::Structured(plan) = super::plan_binary(left, right).unwrap() else {
        panic!("expected tabular plan")
    };
    let (_, variables) = plan.variables();
    assert_eq!(variables[0].0, "A");
    assert_eq!(variables[1].0, "B");
    let Value::Tensor(right_a) = &variables[0].2 else {
        panic!("expected numeric A")
    };
    let Value::Tensor(right_b) = &variables[1].2 else {
        panic!("expected numeric B")
    };
    assert_eq!(right_a.materialize_f64(), vec![10.0, 20.0]);
    assert_eq!(right_b.materialize_f64(), vec![30.0, 40.0]);
}

#[test]
fn rejects_mismatched_variable_and_row_identities() {
    let error = match super::plan_binary(table(&["A"], &[&[1.0]]), table(&["B"], &[&[1.0]])) {
        Err(error) => error,
        Ok(_) => panic!("mismatched variables must reject"),
    };
    assert!(error.contains("same variable names"));

    let mut left = table(&["A"], &[&[1.0, 2.0]]);
    let mut right = table(&["A"], &[&[3.0, 4.0]]);
    set_row_names(&mut left, &["first", "second"]);
    set_row_names(&mut right, &["first", "other"]);
    let error = match super::plan_binary(left, right) {
        Err(error) => error,
        Ok(_) => panic!("mismatched rows must reject"),
    };
    assert!(error.contains("same row names"));
}

#[test]
fn aligns_timetable_rows_without_rounding_wide_integer_times() {
    let first = (1_u64 << 53) + 1;
    let second = u64::MAX;
    let left = timetable(&[1.0, 2.0], vec![first, second]);
    let right = timetable(&[20.0, 10.0], vec![second, first]);

    let BinaryInputPlan::Structured(plan) = super::plan_binary(left, right).unwrap() else {
        panic!("expected timetable plan")
    };
    let (_, variables) = plan.variables();
    let Value::Tensor(right_a) = &variables[0].2 else {
        panic!("expected numeric A")
    };
    assert_eq!(right_a.materialize_f64(), vec![10.0, 20.0]);
}
