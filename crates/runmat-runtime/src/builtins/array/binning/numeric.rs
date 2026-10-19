use std::cmp::Ordering;

use runmat_value::{IntValue, NumericScalar};

pub(crate) fn assign(
    value: NumericScalar,
    edges: &[NumericScalar],
    included_right: bool,
) -> Option<usize> {
    compare(value, value)?;
    for bin in 0..edges.len() - 1 {
        let lower_cmp = compare(value, edges[bin])?;
        let upper_cmp = compare(value, edges[bin + 1])?;
        let hit = if included_right {
            (lower_cmp == Ordering::Greater || (bin == 0 && lower_cmp == Ordering::Equal))
                && upper_cmp != Ordering::Greater
        } else {
            lower_cmp != Ordering::Less
                && (upper_cmp == Ordering::Less
                    || (bin == edges.len() - 2 && upper_cmp == Ordering::Equal))
        };
        if hit {
            return Some(bin + 1);
        }
    }
    None
}

pub(crate) fn compare(left: NumericScalar, right: NumericScalar) -> Option<Ordering> {
    match (integer(left), integer(right)) {
        (Some(left), Some(right)) => Some(left.cmp(&right)),
        (Some(left), None) => compare_integer_float(left, right.materialize_f64()),
        (None, Some(right)) => {
            compare_integer_float(right, left.materialize_f64()).map(Ordering::reverse)
        }
        (None, None) => left.materialize_f64().partial_cmp(&right.materialize_f64()),
    }
}

fn integer(value: NumericScalar) -> Option<i128> {
    value.into_int_value().map(|value| match value {
        IntValue::I8(value) => i128::from(value),
        IntValue::I16(value) => i128::from(value),
        IntValue::I32(value) => i128::from(value),
        IntValue::I64(value) => i128::from(value),
        IntValue::U8(value) => i128::from(value),
        IntValue::U16(value) => i128::from(value),
        IntValue::U32(value) => i128::from(value),
        IntValue::U64(value) => i128::from(value),
    })
}

fn compare_integer_float(integer: i128, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }
    let truncated = float.trunc() as i128;
    match integer.cmp(&truncated) {
        Ordering::Equal if float.fract() > 0.0 => Some(Ordering::Less),
        Ordering::Equal if float.fract() < 0.0 => Some(Ordering::Greater),
        ordering => Some(ordering),
    }
}
