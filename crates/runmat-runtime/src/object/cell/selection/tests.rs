use runmat_value::{CellArray, Value};

use super::{cell_selector_extent, expand_cell_subscripts};
use crate::object::cell::{assign_cell_value, index_cell_value};

fn nd_cell() -> CellArray {
    CellArray::from_column_major(
        (1..=8).map(|value| Value::Num(f64::from(value))).collect(),
        vec![2, 2, 2],
    )
    .expect("N-D cell")
}

#[test]
fn nd_subscripts_follow_column_major_visible_order() {
    let cell = nd_cell();
    assert_eq!(
        index_cell_value(&cell, &[2, 1, 2]).expect("N-D subscript"),
        Value::Num(6.0)
    );
    assert_eq!(
        index_cell_value(&cell, &[6]).expect("linear subscript"),
        Value::Num(6.0)
    );
}

#[test]
fn final_selector_collapses_trailing_dimensions() {
    let cell = nd_cell();
    assert_eq!(cell_selector_extent(&cell, 1, 0).unwrap(), 8);
    assert_eq!(cell_selector_extent(&cell, 2, 0).unwrap(), 2);
    assert_eq!(cell_selector_extent(&cell, 2, 1).unwrap(), 4);
    assert_eq!(cell_selector_extent(&cell, 3, 2).unwrap(), 2);
}

#[test]
fn nd_colon_expansion_uses_first_dimension_fastest() {
    let cell = nd_cell();
    let values = expand_cell_subscripts(
        &cell,
        &[Value::String(":".into()), Value::Num(1.0), Value::Num(2.0)],
    )
    .expect("N-D colon selection");
    assert_eq!(values, vec![Value::Num(5.0), Value::Num(6.0)]);
}

#[test]
fn nd_assignment_uses_the_same_storage_mapping_as_reads() {
    let result = assign_cell_value(nd_cell(), &[2, 1, 2], Value::Num(60.0), |_old, _new| {})
        .expect("N-D assignment");
    let Value::Cell(cell) = result else {
        panic!("expected cell result");
    };
    assert_eq!(
        index_cell_value(&cell, &[2, 1, 2]).expect("N-D read"),
        Value::Num(60.0)
    );
    assert_eq!(
        index_cell_value(&cell, &[6]).expect("linear read"),
        Value::Num(60.0)
    );
}
