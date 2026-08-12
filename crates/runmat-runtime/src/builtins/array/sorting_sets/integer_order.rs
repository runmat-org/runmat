//! Exact ordering helpers shared by integer-aware sorting-set builtins.

use std::cmp::Ordering;

use runmat_value::IntValue;

pub(super) fn compare(
    a: &IntValue,
    b: &IntValue,
    descending: bool,
    comparison_by_abs: bool,
) -> Ordering {
    let primary = if comparison_by_abs {
        absolute_magnitude(a).cmp(&absolute_magnitude(b))
    } else {
        Ordering::Equal
    };
    let ordering = if primary == Ordering::Equal {
        if comparison_by_abs {
            compare_raw(a, b).reverse()
        } else {
            compare_raw(a, b)
        }
    } else {
        primary
    };
    if descending {
        ordering.reverse()
    } else {
        ordering
    }
}

fn compare_raw(a: &IntValue, b: &IntValue) -> Ordering {
    match (a, b) {
        (IntValue::I8(a), IntValue::I8(b)) => a.cmp(b),
        (IntValue::I16(a), IntValue::I16(b)) => a.cmp(b),
        (IntValue::I32(a), IntValue::I32(b)) => a.cmp(b),
        (IntValue::I64(a), IntValue::I64(b)) => a.cmp(b),
        (IntValue::U8(a), IntValue::U8(b)) => a.cmp(b),
        (IntValue::U16(a), IntValue::U16(b)) => a.cmp(b),
        (IntValue::U32(a), IntValue::U32(b)) => a.cmp(b),
        (IntValue::U64(a), IntValue::U64(b)) => a.cmp(b),
        _ => unreachable!("integer storage is homogeneous"),
    }
}

fn absolute_magnitude(value: &IntValue) -> u128 {
    match value {
        IntValue::I8(value) => i128::from(*value).unsigned_abs(),
        IntValue::I16(value) => i128::from(*value).unsigned_abs(),
        IntValue::I32(value) => i128::from(*value).unsigned_abs(),
        IntValue::I64(value) => i128::from(*value).unsigned_abs(),
        IntValue::U8(value) => u128::from(*value),
        IntValue::U16(value) => u128::from(*value),
        IntValue::U32(value) => u128::from(*value),
        IntValue::U64(value) => u128::from(*value),
    }
}
