//! Shared admission geometry and numerical primitives for regularized regression.

use runmat_value::Value;

/// Admission checks visible shape before any operand is downloaded.
/// Both solvers consume the first two dimensions as a column-major matrix,
/// with only singleton trailing dimensions allowed.
pub(super) fn predictor_matrix_dimensions(value: &Value) -> Result<(usize, usize), String> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => &array.shape,
        Value::GpuTensor(handle) => {
            return crate::builtins::common::linalg::matrix_dimensions_for("X", handle.shape())
        }
        Value::Num(_) | Value::Int(_) => return Ok((1, 1)),
        _ => return Err("X must be a real numeric predictor matrix".into()),
    };
    crate::builtins::common::linalg::matrix_dimensions_for("X", shape)
}

pub(super) fn geometric_sequence(max_value: f64, ratio: f64, count: usize) -> Vec<f64> {
    let min_value = max_value * ratio;
    (0..count)
        .map(|index| {
            let position = index as f64 / (count - 1) as f64;
            max_value * (min_value / max_value).powf(position)
        })
        .collect()
}

pub(super) fn sorted_unique_lambdas(lambda: &[f64], tolerance: f64) -> Option<Vec<f64>> {
    if lambda.is_empty() {
        return None;
    }
    let mut values = lambda.to_vec();
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| (*left - *right).abs() <= tolerance);
    Some(values)
}

pub(super) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}
