//! Comparison operations for language-compatible logic
//!
//! Implements comparison operators returning logical matrices/values.

use std::cmp::Ordering;

use runmat_builtins::Tensor;

use crate::builtins::logical::rel::integer_comparison::{
    compare_integer_values, integer_f64_order, matches_optional_relation, matches_relation,
    storage_value, IntegerComparisonOp,
};

/// Element-wise greater than comparison
pub fn matrix_gt(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matrix_compare(a, b, ">", IntegerComparisonOp::Gt, |x, y| x > y)
}

/// Element-wise greater than or equal comparison
pub fn matrix_ge(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matrix_compare(a, b, ">=", IntegerComparisonOp::Ge, |x, y| x >= y)
}

/// Element-wise less than comparison
pub fn matrix_lt(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matrix_compare(a, b, "<", IntegerComparisonOp::Lt, |x, y| x < y)
}

/// Element-wise less than or equal comparison
pub fn matrix_le(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matrix_compare(a, b, "<=", IntegerComparisonOp::Le, |x, y| x <= y)
}

/// Element-wise equality comparison
pub fn matrix_eq(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matrix_compare(a, b, "==", IntegerComparisonOp::Eq, |x, y| {
        (x - y).abs() < f64::EPSILON
    })
}

/// Element-wise inequality comparison
pub fn matrix_ne(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matrix_compare(a, b, "!=", IntegerComparisonOp::Ne, |x, y| {
        (x - y).abs() >= f64::EPSILON
    })
}

fn matrix_compare(
    a: &Tensor,
    b: &Tensor,
    symbol: &str,
    operation: IntegerComparisonOp,
    float_compare: impl Fn(f64, f64) -> bool,
) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} {} {}x{}",
            a.rows(),
            a.cols(),
            symbol,
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = match (a.integer_storage(), b.integer_storage()) {
        (Some(left), Some(right)) => (0..left.len())
            .map(|index| {
                let ordering =
                    compare_integer_values(storage_value(left, index), storage_value(right, index));
                logical_f64(matches_relation(ordering, operation))
            })
            .collect(),
        (Some(left), None) => (0..left.len())
            .map(|index| {
                logical_f64(matches_optional_relation(
                    integer_f64_order(storage_value(left, index), b.data[index]),
                    operation,
                ))
            })
            .collect(),
        (None, Some(right)) => (0..right.len())
            .map(|index| {
                let ordering = integer_f64_order(storage_value(right, index), a.data[index])
                    .map(Ordering::reverse);
                logical_f64(matches_optional_relation(ordering, operation))
            })
            .collect(),
        (None, None) => a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| logical_f64(float_compare(*x, *y)))
            .collect(),
    };

    Tensor::new_2d(data, a.rows(), a.cols())
}

fn logical_f64(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}
