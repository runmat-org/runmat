use super::{StructArray, StructValue};
use crate::Value;
use std::collections::HashSet;

fn structure(first: f64, second: f64) -> StructValue {
    let mut structure = StructValue::new();
    structure.insert("first", Value::Num(first));
    structure.insert("second", Value::Num(second));
    structure
}

#[test]
fn array_replacement_moves_matching_field_columns() {
    let target = StructArray::new(
        vec![
            structure(1.0, 10.0),
            structure(2.0, 20.0),
            structure(3.0, 30.0),
        ],
        vec![3, 1],
    )
    .unwrap();
    let replacements =
        StructArray::new(vec![structure(7.0, 70.0), structure(8.0, 80.0)], vec![2, 1]).unwrap();
    let output = target.replace_linear_array(&[2, 0], replacements).unwrap();
    assert_eq!(
        output.field_values("first").unwrap(),
        [Value::Num(8.0), Value::Num(2.0), Value::Num(7.0)]
    );
    assert_eq!(
        output.field_values("second").unwrap(),
        [Value::Num(80.0), Value::Num(20.0), Value::Num(70.0)]
    );
}

#[test]
fn deletion_moves_each_field_column_and_normalizes_singletons() {
    let array =
        StructArray::new(vec![structure(1.0, 10.0), structure(2.0, 20.0)], vec![2, 1]).unwrap();
    let output = array
        .remove_linear(&HashSet::from([0]), vec![1, 1])
        .unwrap();
    let Value::Struct(output) = output else {
        panic!("one retained element must normalize to a scalar structure");
    };
    assert_eq!(output.fields["first"], Value::Num(2.0));
    assert_eq!(output.fields["second"], Value::Num(20.0));
}

#[test]
fn growth_accepts_removed_trailing_singletons_without_panicking() {
    let array = StructArray::new(
        vec![structure(1.0, 10.0), structure(2.0, 20.0)],
        vec![2, 1, 1],
    )
    .unwrap();
    let output = array.grow(vec![2, 1], &structure(0.0, 0.0)).unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(
        output.field_values("first").unwrap(),
        [Value::Num(1.0), Value::Num(2.0)]
    );
}
