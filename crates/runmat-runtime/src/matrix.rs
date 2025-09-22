//! Matrix operations for MATLAB-compatible arithmetic
//!
//! Implements element-wise and matrix operations following MATLAB semantics.

use runmat_builtins::Tensor;
use runmat_macros::runtime_builtin;

/// Matrix addition: C = A + B
pub fn matrix_add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} + {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Matrix subtraction: C = A - B
pub fn matrix_sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} - {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x - y)
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Matrix multiplication: C = A * B
pub fn matrix_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    // If a or b originate from GPUArray handles (future: track provenance), a GPU provider could handle this.
    // For now, keep CPU implementation, but allow accelerator facade to be invoked when Value::GpuTensor is used via a separate entry point.
    if a.cols() != b.rows() {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let rows = a.rows();
    let cols = b.cols();
    let mut data = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..a.cols() {
                // Column-major: A(i,k) at i + k*rows, B(k,j) at k + j*rows_b
                sum += a.data[i + k * rows] * b.data[k + j * b.rows()];
            }
            // C(i,j) at i + j*rows
            data[i + j * rows] = sum;
        }
    }

    Tensor::new_2d(data, rows, cols)
}

/// GPU-aware matmul entry: if both inputs are GpuTensor handles, call provider; otherwise fall back to CPU.
pub fn value_matmul(
    a: &runmat_builtins::Value,
    b: &runmat_builtins::Value,
) -> Result<runmat_builtins::Value, String> {
    use runmat_builtins::Value;
    // GPU path first
    if let (Value::GpuTensor(ha), Value::GpuTensor(hb)) = (a, b) {
        if let Some(p) = runmat_accelerate_api::provider() {
            match p.matmul(ha, hb) {
                Ok(hc) => {
                    let ht = p.download(&hc).map_err(|e| e.to_string())?;
                    return Ok(Value::Tensor(
                        runmat_builtins::Tensor::new(ht.data, ht.shape)
                            .map_err(|e| e.to_string())?,
                    ));
                }
                Err(_) => {
                    // Fallback: download and compute on CPU
                    let ta = p.download(ha).map_err(|e| e.to_string())?;
                    let tb = p.download(hb).map_err(|e| e.to_string())?;
                    let ca = runmat_builtins::Tensor::new(ta.data, ta.shape)
                        .map_err(|e| e.to_string())?;
                    let cb = runmat_builtins::Tensor::new(tb.data, tb.shape)
                        .map_err(|e| e.to_string())?;
                    return Ok(Value::Tensor(matrix_mul(&ca, &cb)?));
                }
            }
        }
    }
    // Mixed GPU/CPU: gather GPU tensor(s) and recurse
    if matches!(a, Value::GpuTensor(_)) || matches!(b, Value::GpuTensor(_)) {
        let to_host = |v: &Value| -> Result<Value, String> {
            match v {
                Value::GpuTensor(h) => {
                    if let Some(p) = runmat_accelerate_api::provider() {
                        let ht = p.download(h).map_err(|e| e.to_string())?;
                        Ok(Value::Tensor(
                            runmat_builtins::Tensor::new(ht.data, ht.shape)
                                .map_err(|e| e.to_string())?,
                        ))
                    } else {
                        let total: usize = h.shape.iter().product();
                        Ok(Value::Tensor(
                            runmat_builtins::Tensor::new(vec![0.0; total], h.shape.clone())
                                .map_err(|e| e.to_string())?,
                        ))
                    }
                }
                other => Ok(other.clone()),
            }
        };
        let ah = to_host(a)?;
        let bh = to_host(b)?;
        return value_matmul(&ah, &bh);
    }
    // CPU cases
    match (a, b) {
        // Complex scalars and real/complex mixes
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            Ok(Value::Complex(ar * br - ai * bi, ar * bi + ai * br))
        }
        (Value::Complex(ar, ai), Value::Num(s)) => Ok(Value::Complex(ar * s, ai * s)),
        (Value::Num(s), Value::Complex(br, bi)) => Ok(Value::Complex(s * br, s * bi)),
        (Value::Tensor(t), Value::Complex(cr, ci)) => {
            // real matrix times complex scalar
            Ok(Value::ComplexTensor(matrix_scalar_mul_complex(t, *cr, *ci)))
        }
        (Value::Complex(cr, ci), Value::Tensor(t)) => {
            // complex scalar times real matrix
            Ok(Value::ComplexTensor(matrix_scalar_mul_complex(t, *cr, *ci)))
        }
        (Value::ComplexTensor(ct), Value::Num(s)) => Ok(Value::ComplexTensor(
            matrix_scalar_mul_complex_tensor(ct, *s, 0.0),
        )),
        (Value::Num(s), Value::ComplexTensor(ct)) => Ok(Value::ComplexTensor(
            matrix_scalar_mul_complex_tensor(ct, *s, 0.0),
        )),
        (Value::ComplexTensor(ct), Value::Complex(cr, ci)) => Ok(Value::ComplexTensor(
            matrix_scalar_mul_complex_tensor(ct, *cr, *ci),
        )),
        (Value::Complex(cr, ci), Value::ComplexTensor(ct)) => Ok(Value::ComplexTensor(
            matrix_scalar_mul_complex_tensor(ct, *cr, *ci),
        )),
        (Value::Tensor(ta), Value::Tensor(tb)) => Ok(Value::Tensor(matrix_mul(ta, tb)?)),
        (Value::ComplexTensor(ta), Value::ComplexTensor(tb)) => {
            Ok(Value::ComplexTensor(complex_matrix_mul(ta, tb)?))
        }
        (Value::ComplexTensor(ta), Value::Tensor(tb)) => {
            Ok(Value::ComplexTensor(complex_real_matrix_mul(ta, tb)?))
        }
        (Value::Tensor(ta), Value::ComplexTensor(tb)) => {
            Ok(Value::ComplexTensor(real_complex_matrix_mul(ta, tb)?))
        }
        (Value::Tensor(ta), Value::Num(s)) => Ok(Value::Tensor(matrix_scalar_mul(ta, *s))),
        (Value::Num(s), Value::Tensor(tb)) => Ok(Value::Tensor(matrix_scalar_mul(tb, *s))),
        (Value::Tensor(ta), Value::Int(i)) => Ok(Value::Tensor(matrix_scalar_mul(ta, i.to_f64()))),
        (Value::Int(i), Value::Tensor(tb)) => Ok(Value::Tensor(matrix_scalar_mul(tb, i.to_f64()))),
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x * y)),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(x.to_f64() * y)),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x * y.to_f64())),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(x.to_f64() * y.to_f64())),
        _ => Err("matmul: unsupported operand types".to_string()),
    }
}

fn complex_matrix_mul(
    a: &runmat_builtins::ComplexTensor,
    b: &runmat_builtins::ComplexTensor,
) -> Result<runmat_builtins::ComplexTensor, String> {
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
    runmat_builtins::ComplexTensor::new_2d(data, rows, cols)
}

fn complex_real_matrix_mul(
    a: &runmat_builtins::ComplexTensor,
    b: &runmat_builtins::Tensor,
) -> Result<runmat_builtins::ComplexTensor, String> {
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
    runmat_builtins::ComplexTensor::new_2d(data, rows, cols)
}

fn real_complex_matrix_mul(
    a: &runmat_builtins::Tensor,
    b: &runmat_builtins::ComplexTensor,
) -> Result<runmat_builtins::ComplexTensor, String> {
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
    runmat_builtins::ComplexTensor::new_2d(data, rows, cols)
}

fn matrix_scalar_mul_complex(a: &Tensor, cr: f64, ci: f64) -> runmat_builtins::ComplexTensor {
    let data: Vec<(f64, f64)> = a.data.iter().map(|&x| (x * cr, x * ci)).collect();
    runmat_builtins::ComplexTensor::new_2d(data, a.rows(), a.cols()).unwrap()
}

fn matrix_scalar_mul_complex_tensor(
    a: &runmat_builtins::ComplexTensor,
    cr: f64,
    ci: f64,
) -> runmat_builtins::ComplexTensor {
    let data: Vec<(f64, f64)> = a
        .data
        .iter()
        .map(|&(ar, ai)| (ar * cr - ai * ci, ar * ci + ai * cr))
        .collect();
    runmat_builtins::ComplexTensor::new_2d(data, a.rows, a.cols).unwrap()
}

#[runtime_builtin(name = "mtimes", accel = "matmul")]
fn mtimes_builtin(
    a: runmat_builtins::Value,
    b: runmat_builtins::Value,
) -> Result<runmat_builtins::Value, String> {
    use runmat_builtins::Value;
    match (&a, &b) {
        (Value::GpuTensor(_), Value::GpuTensor(_)) => value_matmul(&a, &b),
        (Value::Tensor(ta), Value::Tensor(tb)) => Ok(Value::Tensor(matrix_mul(ta, tb)?)),
        (Value::Tensor(ta), Value::Num(s)) => Ok(Value::Tensor(matrix_scalar_mul(ta, *s))),
        (Value::Num(s), Value::Tensor(tb)) => Ok(Value::Tensor(matrix_scalar_mul(tb, *s))),
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x * y)),
        _ => Err("mtimes: unsupported operand types".to_string()),
    }
}

/// Scalar multiplication: C = A * s
pub fn matrix_scalar_mul(a: &Tensor, scalar: f64) -> Tensor {
    let data: Vec<f64> = a.data.iter().map(|x| x * scalar).collect();
    Tensor::new_2d(data, a.rows(), a.cols()).unwrap() // Always valid
}

/// Matrix transpose: C = A'
pub fn matrix_transpose(a: &Tensor) -> Tensor {
    let mut data = vec![0.0; a.rows() * a.cols()];
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            // dst(j,i) = src(i,j)
            data[j * a.rows() + i] = a.data[i + j * a.rows()];
        }
    }
    Tensor::new_2d(data, a.cols(), a.rows()).unwrap() // Always valid
}

/// Matrix power: C = A^n (for positive integer n)
/// This computes A * A * ... * A (n times) via repeated multiplication
pub fn matrix_power(a: &Tensor, n: i32) -> Result<Tensor, String> {
    if a.rows() != a.cols() {
        return Err(format!(
            "Matrix must be square for matrix power: {}x{}",
            a.rows(),
            a.cols()
        ));
    }

    if n < 0 {
        return Err("Negative matrix powers not supported yet".to_string());
    }

    if n == 0 {
        // A^0 = I (identity matrix)
        return Ok(matrix_eye(a.rows));
    }

    if n == 1 {
        // A^1 = A
        return Ok(a.clone());
    }

    // Compute A^n via repeated multiplication
    // Use binary exponentiation for efficiency
    let mut result = matrix_eye(a.rows());
    let mut base = a.clone();
    let mut exp = n as u32;

    while exp > 0 {
        if exp % 2 == 1 {
            result = matrix_mul(&result, &base)?;
        }
        base = matrix_mul(&base, &base)?;
        exp /= 2;
    }

    Ok(result)
}

/// Complex matrix power: C = A^n (for positive integer n)
/// Uses binary exponentiation with complex matrix multiply
pub fn complex_matrix_power(
    a: &runmat_builtins::ComplexTensor,
    n: i32,
) -> Result<runmat_builtins::ComplexTensor, String> {
    if a.rows != a.cols {
        return Err(format!(
            "Matrix must be square for matrix power: {}x{}",
            a.rows, a.cols
        ));
    }
    if n < 0 {
        return Err("Negative matrix powers not supported yet".to_string());
    }
    if n == 0 {
        return Ok(complex_matrix_eye(a.rows));
    }
    if n == 1 {
        return Ok(a.clone());
    }
    let mut result = complex_matrix_eye(a.rows);
    let mut base = a.clone();
    let mut exp = n as u32;
    while exp > 0 {
        if exp % 2 == 1 {
            result = complex_matrix_mul(&result, &base)?;
        }
        base = complex_matrix_mul(&base, &base)?;
        exp /= 2;
    }
    Ok(result)
}

fn complex_matrix_eye(n: usize) -> runmat_builtins::ComplexTensor {
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); n * n];
    for i in 0..n {
        data[i * n + i] = (1.0, 0.0);
    }
    runmat_builtins::ComplexTensor::new_2d(data, n, n).unwrap()
}

/// Create identity matrix
pub fn matrix_eye(n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::new_2d(data, n, n).unwrap() // Always valid
}

// Simple built-in function for testing matrix operations
#[runtime_builtin(name = "matrix_zeros")]
fn matrix_zeros_builtin(rows: i32, cols: i32) -> Result<Tensor, String> {
    if rows < 0 || cols < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Tensor::zeros(vec![rows as usize, cols as usize]))
}

#[runtime_builtin(name = "matrix_ones")]
fn matrix_ones_builtin(rows: i32, cols: i32) -> Result<Tensor, String> {
    if rows < 0 || cols < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Tensor::ones(vec![rows as usize, cols as usize]))
}

#[runtime_builtin(name = "matrix_eye")]
fn matrix_eye_builtin(n: i32) -> Result<Tensor, String> {
    if n < 0 {
        return Err("Matrix size must be non-negative".to_string());
    }
    Ok(matrix_eye(n as usize))
}

#[runtime_builtin(name = "matrix_transpose")]
fn matrix_transpose_builtin(a: Tensor) -> Result<Tensor, String> {
    Ok(matrix_transpose(&a))
}
