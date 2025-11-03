use runmat_builtins::{ComplexTensor, Tensor, Value};

use super::tensor;

/// Perform column-major matrix multiplication for real tensors.
pub(crate) fn matmul_real(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.cols() != b.rows() {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let rows = a.rows();
    let cols = b.cols();
    let mut data = vec![0.0; rows * cols];

    for j in 0..cols {
        for i in 0..rows {
            let mut sum = 0.0;
            for k in 0..a.cols() {
                sum += a.data[i + k * rows] * b.data[k + j * b.rows()];
            }
            data[i + j * rows] = sum;
        }
    }

    Tensor::new_2d(data, rows, cols).map_err(|e| format!("matmul: {e}"))
}

/// Multiply a real tensor by a real scalar.
pub(crate) fn scalar_mul_real(a: &Tensor, scalar: f64) -> Tensor {
    let data: Vec<f64> = a.data.iter().map(|x| x * scalar).collect();
    Tensor::new(data, a.shape.clone()).expect("scalar_mul_real: invalid tensor")
}

/// Multiply a real tensor by a complex scalar, producing a complex tensor.
pub(crate) fn scalar_mul_complex(a: &Tensor, cr: f64, ci: f64) -> ComplexTensor {
    let data: Vec<(f64, f64)> = a.data.iter().map(|&x| (x * cr, x * ci)).collect();
    ComplexTensor::new(data, a.shape.clone())
        .expect("scalar_mul_complex: invalid tensor")
}

/// Multiply a complex tensor by a complex scalar.
pub(crate) fn scalar_mul_complex_tensor(a: &ComplexTensor, cr: f64, ci: f64) -> ComplexTensor {
    let data: Vec<(f64, f64)> = a
        .data
        .iter()
        .map(|&(ar, ai)| (ar * cr - ai * ci, ar * ci + ai * cr))
        .collect();
    ComplexTensor::new_2d(data, a.rows, a.cols).expect("scalar_mul_complex_tensor: invalid tensor")
}

/// Multiply two complex matrices.
pub(crate) fn matmul_complex(
    a: &ComplexTensor,
    b: &ComplexTensor,
) -> Result<ComplexTensor, String> {
    if a.cols != b.rows {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }
    let rows = a.rows;
    let cols = b.cols;
    let kdim = a.cols;
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for k in 0..kdim {
                let (ar, ai) = a.data[i + k * rows];
                let (br, bi) = b.data[k + j * b.rows];
                acc_re += ar * br - ai * bi;
                acc_im += ar * bi + ai * br;
            }
            data[i + j * rows] = (acc_re, acc_im);
        }
    }
    ComplexTensor::new_2d(data, rows, cols)
}

/// Multiply a complex tensor by a real tensor.
pub(crate) fn matmul_complex_real(a: &ComplexTensor, b: &Tensor) -> Result<ComplexTensor, String> {
    if a.cols != b.rows() {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows,
            a.cols,
            b.rows(),
            b.cols()
        ));
    }
    let rows = a.rows;
    let cols = b.cols();
    let kdim = a.cols;
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for k in 0..kdim {
                let (ar, ai) = a.data[i + k * rows];
                let br = b.data[k + j * b.rows()];
                acc_re += ar * br;
                acc_im += ai * br;
            }
            data[i + j * rows] = (acc_re, acc_im);
        }
    }
    ComplexTensor::new_2d(data, rows, cols)
}

/// Multiply a real tensor by a complex tensor.
pub(crate) fn matmul_real_complex(a: &Tensor, b: &ComplexTensor) -> Result<ComplexTensor, String> {
    if a.cols() != b.rows {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows(),
            a.cols(),
            b.rows,
            b.cols
        ));
    }
    let rows = a.rows();
    let cols = b.cols;
    let kdim = a.cols();
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for k in 0..kdim {
                let ar = a.data[i + k * rows];
                let (br, bi) = b.data[k + j * b.rows];
                acc_re += ar * br;
                acc_im += ar * bi;
            }
            data[i + j * rows] = (acc_re, acc_im);
        }
    }
    ComplexTensor::new_2d(data, rows, cols)
}

/// Determine the 2-D matrix dimensions for a builtin, allowing trailing singleton axes.
pub(crate) fn matrix_dimensions_for(
    label: &str,
    shape: &[usize],
) -> Result<(usize, usize), String> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((shape[0], 1)),
        _ => {
            if shape.len() > 2 && shape.iter().skip(2).any(|&dim| dim != 1) {
                Err(format!("{label}: inputs must be 2-D matrices or vectors"))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

/// Parse an optional tolerance argument shared by SVD-derived builtins.
pub(crate) fn parse_tolerance_arg(name: &str, args: &[Value]) -> Result<Option<f64>, String> {
    match args.len() {
        0 => Ok(None),
        1 => {
            let raw = match &args[0] {
                Value::Num(n) => *n,
                Value::Int(i) => i.to_f64(),
                Value::Tensor(t) if tensor::is_scalar_tensor(t) => t.data[0],
                Value::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Value::LogicalArray(l) if l.len() == 1 => {
                    if l.data[0] != 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                other => {
                    return Err(format!(
                        "{name}: tolerance must be a real scalar, got {other:?}"
                    ))
                }
            };
            if !raw.is_finite() {
                return Err(format!("{name}: tolerance must be finite"));
            }
            if raw < 0.0 {
                return Err(format!("{name}: tolerance must be >= 0"));
            }
            Ok(Some(raw))
        }
        _ => Err(format!("{name}: too many input arguments")),
    }
}

/// MATLAB-compatible default tolerance used by `pinv`, `rank`, and related routines.
pub(crate) fn svd_default_tolerance(singular: &[f64], rows: usize, cols: usize) -> f64 {
    let max_sv = singular
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let max_dim = rows.max(cols) as f64;
    max_dim * eps_like(max_sv)
}

/// Equivalent of MATLAB's `eps(x)` for real scalars.
pub(crate) fn eps_like(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value.is_infinite() {
        return f64::INFINITY;
    }
    let abs = value.abs();
    let next = f64::from_bits(abs.to_bits().wrapping_add(1));
    next - abs
}

/// Reciprocal condition number for diagonal matrices given the min/max diagonal elements.
#[inline]
pub(crate) fn diagonal_rcond(min_diag: f64, max_diag: f64) -> f64 {
    if max_diag == 0.0 {
        0.0
    } else {
        min_diag / max_diag
    }
}

/// Reciprocal condition number derived from singular values (min/max ratio).
#[inline]
pub(crate) fn singular_value_rcond(singular_values: &[f64]) -> f64 {
    if singular_values.is_empty() {
        return 1.0;
    }
    let mut min_sv = f64::INFINITY;
    let mut max_sv = 0.0_f64;
    for &sv in singular_values {
        let abs = sv.abs();
        if !abs.is_finite() {
            return 0.0;
        }
        min_sv = min_sv.min(abs);
        max_sv = max_sv.max(abs);
    }
    if max_sv == 0.0 {
        0.0
    } else {
        min_sv / max_sv
    }
}
