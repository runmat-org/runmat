//! MATLAB-compatible `peaks` builtin for RunMat.
//!
//! `peaks` is a sample-data function that evaluates a well-known 3-D test surface over an n×n
//! grid spanning [-3, 3] × [-3, 3].  The formula is:
//!
//! ```text
//! Z = 3*(1-x)^2 * exp(-x^2 - (y+1)^2)
//!     - 10*(x/5 - x^3 - y^5) * exp(-x^2 - y^2)
//!     - 1/3 * exp(-(x+1)^2 - y^2)
//! ```
//!
//! Call forms
//! ----------
//! * `peaks`        – 49×49 Z matrix (MATLAB default)
//! * `peaks(n)`     – n×n Z matrix over the standard [-3,3] grid
//! * `peaks(X, Y)`  – evaluate at caller-supplied coordinate matrices
//! * `[X,Y,Z] = peaks(…)` – also return the coordinate matrices

use runmat_builtins::{ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::tensor;

const DEFAULT_N: usize = 49;

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("peaks").build()
}

fn peaks_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

#[runtime_builtin(
    name = "peaks",
    category = "array/creation",
    summary = "Sample data: 3-D test surface on an n-by-n grid.",
    keywords = "peaks,sample,surface,test,demo",
    accel = "array_construct",
    type_resolver(peaks_type),
    builtin_path = "crate::builtins::array::creation::peaks"
)]
async fn peaks_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();

    match rest.len() {
        // peaks  or  peaks()
        0 => {
            let (x_flat, y_flat) = make_axis(DEFAULT_N);
            let (x_mat, y_mat) = make_grids(&x_flat, &y_flat, DEFAULT_N, DEFAULT_N);
            let z_mat = compute_z(&x_mat, &y_mat, DEFAULT_N, DEFAULT_N);
            build_output(x_mat, y_mat, z_mat, DEFAULT_N, DEFAULT_N, out_count)
        }

        // peaks(n)
        1 => {
            let n = parse_scalar_n(&rest[0]).await?;
            let (x_flat, y_flat) = make_axis(n);
            let (x_mat, y_mat) = make_grids(&x_flat, &y_flat, n, n);
            let z_mat = compute_z(&x_mat, &y_mat, n, n);
            build_output(x_mat, y_mat, z_mat, n, n, out_count)
        }

        // peaks(X, Y)
        2 => {
            let x_mat = gather_tensor(&rest[0]).await?;
            let y_mat = gather_tensor(&rest[1]).await?;
            let (rows, cols) = matrix_shape(&x_mat)?;
            let (y_rows, y_cols) = matrix_shape(&y_mat)?;
            if rows != y_rows || cols != y_cols {
                return Err(builtin_error("peaks: X and Y must have the same size"));
            }
            let z_mat = compute_z(&x_mat.data, &y_mat.data, rows, cols);
            build_output(x_mat.data, y_mat.data, z_mat, rows, cols, out_count)
        }

        _ => Err(builtin_error("peaks: expected 0, 1, or 2 input arguments")),
    }
}

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

/// Linearly-spaced values from -3 to 3 (n points) for both axes.
fn make_axis(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![3.0], vec![3.0]);
    }
    let axis: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    (axis.clone(), axis)
}

/// Build flat X and Y coordinate matrices stored column-major.
///
/// meshgrid(x_axis, y_axis): X[row,col] = x_axis[col], Y[row,col] = y_axis[row].
/// Column-major: element (row, col) lives at index `row + col * rows`.
fn make_grids(x_axis: &[f64], y_axis: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let size = rows * cols;
    let mut x_mat = vec![0.0f64; size];
    let mut y_mat = vec![0.0f64; size];
    for col in 0..cols {
        for row in 0..rows {
            x_mat[row + col * rows] = *x_axis.get(col).unwrap_or(&0.0);
            y_mat[row + col * rows] = *y_axis.get(row).unwrap_or(&0.0);
        }
    }
    (x_mat, y_mat)
}

// ---------------------------------------------------------------------------
// Surface formula
// ---------------------------------------------------------------------------

#[inline]
fn peaks_at(x: f64, y: f64) -> f64 {
    3.0 * (1.0 - x).powi(2) * (-(x.powi(2)) - (y + 1.0).powi(2)).exp()
        - 10.0 * (x / 5.0 - x.powi(3) - y.powi(5)) * (-(x.powi(2)) - y.powi(2)).exp()
        - 1.0 / 3.0 * (-(x + 1.0).powi(2) - y.powi(2)).exp()
}

fn compute_z(x_mat: &[f64], y_mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let _ = (rows, cols);
    x_mat
        .iter()
        .zip(y_mat.iter())
        .map(|(&x, &y)| peaks_at(x, y))
        .collect()
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn build_output(
    x_flat: Vec<f64>,
    y_flat: Vec<f64>,
    z_flat: Vec<f64>,
    rows: usize,
    cols: usize,
    out_count: Option<usize>,
) -> crate::BuiltinResult<Value> {
    let shape = vec![rows, cols];

    match out_count {
        // Caller explicitly requested 3 outputs: [X, Y, Z] = peaks(…)
        Some(3) => {
            let x_val = make_tensor(x_flat, shape.clone())?;
            let y_val = make_tensor(y_flat, shape.clone())?;
            let z_val = make_tensor(z_flat, shape)?;
            Ok(Value::OutputList(vec![x_val, y_val, z_val]))
        }
        // Caller requested 2 outputs: [X, Y] = peaks(…) — unusual but allowed
        Some(2) => {
            let x_val = make_tensor(x_flat, shape.clone())?;
            let y_val = make_tensor(y_flat, shape)?;
            Ok(Value::OutputList(vec![x_val, y_val]))
        }
        // Default single output: Z = peaks(…)
        _ => make_tensor(z_flat, shape),
    }
}

fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
    if shape.contains(&0) {
        return Tensor::new(Vec::new(), shape)
            .map(tensor::tensor_into_value)
            .map_err(|e| builtin_error(format!("peaks: {e}")));
    }
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|e| builtin_error(format!("peaks: {e}")))
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

async fn parse_scalar_n(value: &Value) -> crate::BuiltinResult<usize> {
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|e| builtin_error(format!("peaks: {e}")))?
    else {
        return Err(builtin_error("peaks: n must be a numeric scalar"));
    };
    if !raw.is_finite() {
        return Err(builtin_error("peaks: n must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-6 {
        return Err(builtin_error("peaks: n must be an integer"));
    }
    if rounded < 0.0 {
        return Err(builtin_error("peaks: n must be non-negative"));
    }
    Ok(rounded as usize)
}

async fn gather_tensor(value: &Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t.clone()),
        Value::Num(v) => {
            Tensor::new(vec![*v], vec![1, 1]).map_err(|e| builtin_error(format!("peaks: {e}")))
        }
        _ => Err(builtin_error("peaks: X and Y must be numeric matrices")),
    }
}

fn matrix_shape(tensor: &Tensor) -> crate::BuiltinResult<(usize, usize)> {
    match tensor.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        [n] => Ok((1, *n)),
        _ => Err(builtin_error("peaks: X and Y must be 2-D matrices")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn peaks_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::peaks_builtin(rest))
    }

    #[test]
    fn peaks_default_shape() {
        let value = peaks_builtin(vec![]).expect("peaks");
        match value {
            Value::Tensor(t) => assert_eq!(t.shape, vec![49, 49]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_n_shape() {
        let value = peaks_builtin(vec![Value::Num(20.0)]).expect("peaks");
        match value {
            Value::Tensor(t) => assert_eq!(t.shape, vec![20, 20]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_zero_is_empty() {
        let value = peaks_builtin(vec![Value::Num(0.0)]).expect("peaks");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_one_is_scalar() {
        // At n=1 the single grid point maps to the stop endpoint (x=3, y=3).
        // tensor_into_value may collapse a 1×1 tensor to Value::Num.
        let expected = peaks_at(3.0, 3.0);
        let value = peaks_builtin(vec![Value::Num(1.0)]).expect("peaks");
        let got = match value {
            Value::Num(v) => v,
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                t.data[0]
            }
            other => panic!("expected scalar or 1×1 tensor, got {other:?}"),
        };
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn peaks_formula_known_value() {
        // The origin (x=0, y=0) is a well-known point.
        // Z = 3*(1)^2*exp(0 - 1) - 10*(0 - 0 - 0)*exp(0) - 1/3*exp(-1 - 0)
        //   = 3*exp(-1) - 0 - 1/3*exp(-1)
        //   = exp(-1) * (3 - 1/3)
        //   = exp(-1) * 8/3
        let expected = std::f64::consts::E.recip() * 8.0 / 3.0;
        let got = peaks_at(0.0, 0.0);
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn peaks_xy_form() {
        use runmat_builtins::Tensor;
        let x = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let y = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let value =
            peaks_builtin(vec![Value::Tensor(x.clone()), Value::Tensor(y.clone())]).expect("peaks");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                for i in 0..4 {
                    let expected = peaks_at(x.data[i], y.data[i]);
                    assert!((t.data[i] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_too_many_args_errors() {
        let err =
            peaks_builtin(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]).unwrap_err();
        assert!(err.to_string().contains("0, 1, or 2"));
    }

    #[test]
    fn peaks_non_integer_n_errors() {
        let err = peaks_builtin(vec![Value::Num(3.7)]).unwrap_err();
        assert!(err.to_string().contains("integer"));
    }

    #[test]
    fn peaks_negative_n_errors() {
        let err = peaks_builtin(vec![Value::Num(-1.0)]).unwrap_err();
        assert!(err.to_string().contains("non-negative"));
    }
}
