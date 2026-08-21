use runmat_value::IntValue;

pub(super) fn zero_within(value: &IntValue, tol: f64) -> bool {
    if value.is_zero() {
        return true;
    }
    if tol <= 0.0 || !tol.is_finite() {
        return false;
    }
    magnitude(value) as f64 <= tol
}

pub(super) fn equal_within(left: &IntValue, right: &IntValue, tol: f64) -> bool {
    if left == right {
        return true;
    }
    if tol <= 0.0 || !tol.is_finite() {
        return false;
    }
    difference_magnitude(left, right) as f64 <= tol
}

pub(super) fn negated_equal_within(left: &IntValue, right: &IntValue, tol: f64) -> bool {
    if signed_sum_is_zero(left, right) {
        return true;
    }
    if tol <= 0.0 || !tol.is_finite() {
        return false;
    }
    signed_sum_magnitude(left, right) as f64 <= tol
}

fn magnitude(value: &IntValue) -> u128 {
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

fn difference_magnitude(left: &IntValue, right: &IntValue) -> u128 {
    macro_rules! same_class_difference {
        ($left:expr, $right:expr) => {{
            if $left >= $right {
                u128::from($left - $right)
            } else {
                u128::from($right - $left)
            }
        }};
    }

    match (left, right) {
        (IntValue::I8(left), IntValue::I8(right)) => i128::from(*left).abs_diff(i128::from(*right)),
        (IntValue::I16(left), IntValue::I16(right)) => {
            i128::from(*left).abs_diff(i128::from(*right))
        }
        (IntValue::I32(left), IntValue::I32(right)) => {
            i128::from(*left).abs_diff(i128::from(*right))
        }
        (IntValue::I64(left), IntValue::I64(right)) => {
            i128::from(*left).abs_diff(i128::from(*right))
        }
        (IntValue::U8(left), IntValue::U8(right)) => same_class_difference!(*left, *right),
        (IntValue::U16(left), IntValue::U16(right)) => same_class_difference!(*left, *right),
        (IntValue::U32(left), IntValue::U32(right)) => same_class_difference!(*left, *right),
        (IntValue::U64(left), IntValue::U64(right)) => {
            if left >= right {
                u128::from(left - right)
            } else {
                u128::from(right - left)
            }
        }
        _ => unreachable!("integer storage is homogeneous"),
    }
}

fn signed_sum_is_zero(left: &IntValue, right: &IntValue) -> bool {
    match (left, right) {
        (IntValue::I8(left), IntValue::I8(right)) => i16::from(*left) + i16::from(*right) == 0,
        (IntValue::I16(left), IntValue::I16(right)) => i32::from(*left) + i32::from(*right) == 0,
        (IntValue::I32(left), IntValue::I32(right)) => i64::from(*left) + i64::from(*right) == 0,
        (IntValue::I64(left), IntValue::I64(right)) => i128::from(*left) + i128::from(*right) == 0,
        (IntValue::U8(left), IntValue::U8(right)) => *left == 0 && *right == 0,
        (IntValue::U16(left), IntValue::U16(right)) => *left == 0 && *right == 0,
        (IntValue::U32(left), IntValue::U32(right)) => *left == 0 && *right == 0,
        (IntValue::U64(left), IntValue::U64(right)) => *left == 0 && *right == 0,
        _ => unreachable!("integer storage is homogeneous"),
    }
}

fn signed_sum_magnitude(left: &IntValue, right: &IntValue) -> u128 {
    match (left, right) {
        (IntValue::I8(left), IntValue::I8(right)) => {
            i128::from(i16::from(*left) + i16::from(*right)).unsigned_abs()
        }
        (IntValue::I16(left), IntValue::I16(right)) => {
            i128::from(i32::from(*left) + i32::from(*right)).unsigned_abs()
        }
        (IntValue::I32(left), IntValue::I32(right)) => {
            i128::from(i64::from(*left) + i64::from(*right)).unsigned_abs()
        }
        (IntValue::I64(left), IntValue::I64(right)) => {
            (i128::from(*left) + i128::from(*right)).unsigned_abs()
        }
        (IntValue::U8(left), IntValue::U8(right)) => u128::from(*left) + u128::from(*right),
        (IntValue::U16(left), IntValue::U16(right)) => u128::from(*left) + u128::from(*right),
        (IntValue::U32(left), IntValue::U32(right)) => u128::from(*left) + u128::from(*right),
        (IntValue::U64(left), IntValue::U64(right)) => u128::from(*left) + u128::from(*right),
        _ => unreachable!("integer storage is homogeneous"),
    }
}
