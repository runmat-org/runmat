//! Element-wise operations for matrices and scalars
//!
//! This module implements language-compatible element-wise operations (.*,  ./,  .^)
//! These operations work element-by-element on matrices and support scalar broadcasting.

use crate::matrix::matrix_power;
use runmat_builtins::{Tensor, Value};

fn complex_pow_scalar(base_re: f64, base_im: f64, exp_re: f64, exp_im: f64) -> (f64, f64) {
    if base_re == 0.0 && base_im == 0.0 && exp_re == 0.0 && exp_im == 0.0 {
        return (1.0, 0.0);
    }
    if base_re == 0.0 && base_im == 0.0 && exp_im == 0.0 && exp_re > 0.0 {
        return (0.0, 0.0);
    }
    let r = (base_re.hypot(base_im)).max(0.0);
    if r == 0.0 {
        return (0.0, 0.0);
    }
    let theta = base_im.atan2(base_re);
    let ln_r = r.ln();
    let a = exp_re * ln_r - exp_im * theta;
    let b = exp_re * theta + exp_im * ln_r;
    let mag = a.exp();
    (mag * b.cos(), mag * b.sin())
}

async fn to_host_value(v: &Value) -> Result<Value, String> {
    match v {
        Value::GpuTensor(h) => {
            if runmat_accelerate_api::provider_for_handle(h).is_some() {
                let gathered = crate::dispatcher::gather_if_needed_async(v)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(gathered)
            } else {
                // Fallback: zeros tensor with same shape
                let total: usize = h.shape.iter().product();
                Ok(Value::Tensor(
                    Tensor::new(vec![0.0; total], h.shape.clone()).map_err(|e| e.to_string())?,
                ))
            }
        }
        other => Ok(other.clone()),
    }
}

/// Element-wise negation: -A
/// Supports scalars and matrices
pub fn elementwise_neg(a: &Value) -> Result<Value, String> {
    match a {
        Value::Num(x) => Ok(Value::Num(-x)),
        Value::Complex(re, im) => Ok(Value::Complex(-*re, -*im)),
        Value::Int(x) => {
            let v = x.to_i64();
            if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                Ok(Value::Int(runmat_builtins::IntValue::I32(-(v as i32))))
            } else {
                Ok(Value::Int(runmat_builtins::IntValue::I64(-v)))
            }
        }
        Value::Bool(b) => Ok(Value::Bool(!b)), // Boolean negation
        Value::Tensor(m) => {
            let data: Vec<f64> = m.data.iter().map(|x| -x).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        _ => Err(format!("Negation not supported for type: -{a:?}")),
    }
}

/// Element-wise multiplication: A .* B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
#[async_recursion::async_recursion(?Send)]
pub async fn elementwise_mul(a: &Value, b: &Value) -> Result<Value, String> {
    // GPU+scalar: keep on device if provider supports scalar mul
    if let Some(p) = runmat_accelerate_api::provider() {
        match (a, b) {
            (Value::GpuTensor(ga), Value::Num(s)) => {
                if let Ok(hc) = p.scalar_mul(ga, *s) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            (Value::Num(s), Value::GpuTensor(gb)) => {
                if let Ok(hc) = p.scalar_mul(gb, *s) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            (Value::GpuTensor(ga), Value::Int(i)) => {
                if let Ok(hc) = p.scalar_mul(ga, i.to_f64()) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            (Value::Int(i), Value::GpuTensor(gb)) => {
                if let Ok(hc) = p.scalar_mul(gb, i.to_f64()) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            _ => {}
        }
    }
    // If exactly one is GPU and no scalar fast-path, gather to host and recurse
    if matches!(a, Value::GpuTensor(_)) ^ matches!(b, Value::GpuTensor(_)) {
        let ah = to_host_value(a).await?;
        let bh = to_host_value(b).await?;
        return elementwise_mul(&ah, &bh).await;
    }
    if let Some(p) = runmat_accelerate_api::provider() {
        if let (Value::GpuTensor(ha), Value::GpuTensor(hb)) = (a, b) {
            if let Ok(hc) = p.elem_mul(ha, hb).await {
                return Ok(Value::GpuTensor(hc));
            }
        }
    }
    match (a, b) {
        // Complex scalars
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            Ok(Value::Complex(ar * br - ai * bi, ar * bi + ai * br))
        }
        (Value::Complex(ar, ai), Value::Num(s)) => Ok(Value::Complex(ar * s, ai * s)),
        (Value::Num(s), Value::Complex(br, bi)) => Ok(Value::Complex(s * br, s * bi)),
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x * y)),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(x.to_f64() * y)),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x * y.to_f64())),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(x.to_f64() * y.to_f64())),

        // Matrix-scalar cases (broadcasting)
        (Value::Tensor(m), Value::Num(s)) => {
            let data: Vec<f64> = m.data.iter().map(|x| x * s).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let scalar = s.to_f64();
            let data: Vec<f64> = m.data.iter().map(|x| x * scalar).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Num(s), Value::Tensor(m)) => {
            let data: Vec<f64> = m.data.iter().map(|x| s * x).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Int(s), Value::Tensor(m)) => {
            let scalar = s.to_f64();
            let data: Vec<f64> = m.data.iter().map(|x| scalar * x).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }

        // Matrix-matrix case
        (Value::Tensor(m1), Value::Tensor(m2)) => {
            if m1.rows() != m2.rows() || m1.cols() != m2.cols() {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise multiplication: {}x{} .* {}x{}",
                    m1.rows(),
                    m1.cols(),
                    m2.rows(),
                    m2.cols()
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| x * y)
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m1.rows(), m1.cols())?))
        }

        // Complex tensors
        (Value::ComplexTensor(m1), Value::ComplexTensor(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise multiplication: {}x{} .* {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(m1.data.len());
            for i in 0..m1.data.len() {
                let (ar, ai) = m1.data[i];
                let (br, bi) = m2.data[i];
                out.push((ar * br - ai * bi, ar * bi + ai * br));
            }
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new(out, m1.shape.clone())
                    .map_err(|e| format!(".*: {e}"))?,
            ))
        }
        (Value::ComplexTensor(m), Value::Num(s)) => {
            let data: Vec<(f64, f64)> = m.data.iter().map(|(re, im)| (re * s, im * s)).collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m.rows, m.cols)?,
            ))
        }
        (Value::Num(s), Value::ComplexTensor(m)) => {
            let data: Vec<(f64, f64)> = m.data.iter().map(|(re, im)| (s * re, s * im)).collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m.rows, m.cols)?,
            ))
        }

        _ => Err(format!(
            "Element-wise multiplication not supported for types: {a:?} .* {b:?}"
        )),
    }
}

// elementwise_add has been retired in favor of the `plus` builtin

// elementwise_sub has been retired in favor of the `minus` builtin

/// Element-wise division: A ./ B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
#[async_recursion::async_recursion(?Send)]
pub async fn elementwise_div(a: &Value, b: &Value) -> Result<Value, String> {
    // GPU+scalar: use scalar div when form is G ./ s or left-scalar s ./ G
    if let Some(p) = runmat_accelerate_api::provider() {
        match (a, b) {
            (Value::GpuTensor(ga), Value::Num(s)) => {
                if let Ok(hc) = p.scalar_div(ga, *s) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            (Value::GpuTensor(ga), Value::Int(i)) => {
                if let Ok(hc) = p.scalar_div(ga, i.to_f64()) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            (Value::Num(s), Value::GpuTensor(gb)) => {
                if let Ok(hc) = p.scalar_rdiv(gb, *s) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            (Value::Int(i), Value::GpuTensor(gb)) => {
                if let Ok(hc) = p.scalar_rdiv(gb, i.to_f64()) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            _ => {}
        }
    }
    if matches!(a, Value::GpuTensor(_)) ^ matches!(b, Value::GpuTensor(_)) {
        let ah = to_host_value(a).await?;
        let bh = to_host_value(b).await?;
        return elementwise_div(&ah, &bh).await;
    }
    if let Some(p) = runmat_accelerate_api::provider() {
        if let (Value::GpuTensor(ha), Value::GpuTensor(hb)) = (a, b) {
            if let Ok(hc) = p.elem_div(ha, hb).await {
                return Ok(Value::GpuTensor(hc));
            }
        }
    }
    match (a, b) {
        // Complex scalars
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            let denom = br * br + bi * bi;
            if denom == 0.0 {
                return Ok(Value::Num(f64::NAN));
            }
            Ok(Value::Complex(
                (ar * br + ai * bi) / denom,
                (ai * br - ar * bi) / denom,
            ))
        }
        (Value::Complex(ar, ai), Value::Num(s)) => Ok(Value::Complex(ar / s, ai / s)),
        (Value::Num(s), Value::Complex(br, bi)) => {
            let denom = br * br + bi * bi;
            if denom == 0.0 {
                return Ok(Value::Num(f64::NAN));
            }
            Ok(Value::Complex((s * br) / denom, (-s * bi) / denom))
        }
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => {
            if *y == 0.0 {
                Ok(Value::Num(f64::INFINITY * x.signum()))
            } else {
                Ok(Value::Num(x / y))
            }
        }
        (Value::Int(x), Value::Num(y)) => {
            if *y == 0.0 {
                Ok(Value::Num(f64::INFINITY * x.to_f64().signum()))
            } else {
                Ok(Value::Num(x.to_f64() / y))
            }
        }
        (Value::Num(x), Value::Int(y)) => {
            if y.is_zero() {
                Ok(Value::Num(f64::INFINITY * x.signum()))
            } else {
                Ok(Value::Num(x / y.to_f64()))
            }
        }
        (Value::Int(x), Value::Int(y)) => {
            if y.is_zero() {
                Ok(Value::Num(f64::INFINITY * x.to_f64().signum()))
            } else {
                Ok(Value::Num(x.to_f64() / y.to_f64()))
            }
        }

        // Matrix-scalar cases (broadcasting)
        (Value::Tensor(m), Value::Num(s)) => {
            if *s == 0.0 {
                let data: Vec<f64> = m.data.iter().map(|x| f64::INFINITY * x.signum()).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            } else {
                let data: Vec<f64> = m.data.iter().map(|x| x / s).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            }
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let scalar = s.to_f64();
            if scalar == 0.0 {
                let data: Vec<f64> = m.data.iter().map(|x| f64::INFINITY * x.signum()).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            } else {
                let data: Vec<f64> = m.data.iter().map(|x| x / scalar).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            }
        }
        (Value::Num(s), Value::Tensor(m)) => {
            let data: Vec<f64> = m
                .data
                .iter()
                .map(|x| {
                    if *x == 0.0 {
                        f64::INFINITY * s.signum()
                    } else {
                        s / x
                    }
                })
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Int(s), Value::Tensor(m)) => {
            let scalar = s.to_f64();
            let data: Vec<f64> = m
                .data
                .iter()
                .map(|x| {
                    if *x == 0.0 {
                        f64::INFINITY * scalar.signum()
                    } else {
                        scalar / x
                    }
                })
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }

        // Matrix-matrix case
        (Value::Tensor(m1), Value::Tensor(m2)) => {
            if m1.rows() != m2.rows() || m1.cols() != m2.cols() {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise division: {}x{} ./ {}x{}",
                    m1.rows(),
                    m1.cols(),
                    m2.rows(),
                    m2.cols()
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| {
                    if *y == 0.0 {
                        f64::INFINITY * x.signum()
                    } else {
                        x / y
                    }
                })
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m1.rows(), m1.cols())?))
        }

        // Complex tensors
        (Value::ComplexTensor(m1), Value::ComplexTensor(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise division: {}x{} ./ {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let data: Vec<(f64, f64)> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|((ar, ai), (br, bi))| {
                    let denom = br * br + bi * bi;
                    if denom == 0.0 {
                        (f64::NAN, f64::NAN)
                    } else {
                        ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
                    }
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m1.rows, m1.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Num(s)) => {
            let data: Vec<(f64, f64)> = m.data.iter().map(|(re, im)| (re / s, im / s)).collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m.rows, m.cols)?,
            ))
        }
        (Value::Num(s), Value::ComplexTensor(m)) => {
            let data: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(br, bi)| {
                    let denom = br * br + bi * bi;
                    if denom == 0.0 {
                        (f64::NAN, f64::NAN)
                    } else {
                        ((s * br) / denom, (-s * bi) / denom)
                    }
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m.rows, m.cols)?,
            ))
        }

        _ => Err(format!(
            "Element-wise division not supported for types: {a:?} ./ {b:?}"
        )),
    }
}

/// Regular power operation: A ^ B  
/// For matrices, this is matrix exponentiation (A^n where n is integer)
/// For scalars, this is regular exponentiation
pub fn power(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        // Scalar cases - include complex
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            let (r, i) = complex_pow_scalar(*ar, *ai, *br, *bi);
            Ok(Value::Complex(r, i))
        }
        (Value::Complex(ar, ai), Value::Num(y)) => {
            let (r, i) = complex_pow_scalar(*ar, *ai, *y, 0.0);
            Ok(Value::Complex(r, i))
        }
        (Value::Num(x), Value::Complex(br, bi)) => {
            let (r, i) = complex_pow_scalar(*x, 0.0, *br, *bi);
            Ok(Value::Complex(r, i))
        }
        (Value::Complex(ar, ai), Value::Int(y)) => {
            let yv = y.to_f64();
            let (r, i) = complex_pow_scalar(*ar, *ai, yv, 0.0);
            Ok(Value::Complex(r, i))
        }
        (Value::Int(x), Value::Complex(br, bi)) => {
            let xv = x.to_f64();
            let (r, i) = complex_pow_scalar(xv, 0.0, *br, *bi);
            Ok(Value::Complex(r, i))
        }

        // Scalar cases - real only
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x.powf(*y))),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(x.to_f64().powf(*y))),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x.powf(y.to_f64()))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(x.to_f64().powf(y.to_f64()))),

        // Matrix^scalar case - matrix exponentiation
        (Value::Tensor(m), Value::Num(s)) => {
            // Check if scalar is an integer for matrix power
            if s.fract() == 0.0 {
                let n = *s as i32;
                let result = matrix_power(m, n)?;
                Ok(Value::Tensor(result))
            } else {
                Err("Matrix power requires integer exponent".to_string())
            }
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let result = matrix_power(m, s.to_i64() as i32)?;
            Ok(Value::Tensor(result))
        }

        // Complex matrix^integer case
        (Value::ComplexTensor(m), Value::Num(s)) => {
            if s.fract() == 0.0 {
                let n = *s as i32;
                let result = crate::matrix::complex_matrix_power(m, n)?;
                Ok(Value::ComplexTensor(result))
            } else {
                Err("Matrix power requires integer exponent".to_string())
            }
        }
        (Value::ComplexTensor(m), Value::Int(s)) => {
            let result = crate::matrix::complex_matrix_power(m, s.to_i64() as i32)?;
            Ok(Value::ComplexTensor(result))
        }

        // Other cases not supported for regular matrix power
        _ => Err(format!(
            "Power operation not supported for types: {a:?} ^ {b:?}"
        )),
    }
}

/// Element-wise power: A .^ B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
pub fn elementwise_pow(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        // Complex scalar cases
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            let (r, i) = complex_pow_scalar(*ar, *ai, *br, *bi);
            Ok(Value::Complex(r, i))
        }
        (Value::Complex(ar, ai), Value::Num(y)) => {
            let (r, i) = complex_pow_scalar(*ar, *ai, *y, 0.0);
            Ok(Value::Complex(r, i))
        }
        (Value::Num(x), Value::Complex(br, bi)) => {
            let (r, i) = complex_pow_scalar(*x, 0.0, *br, *bi);
            Ok(Value::Complex(r, i))
        }
        (Value::Complex(ar, ai), Value::Int(y)) => {
            let yv = y.to_f64();
            let (r, i) = complex_pow_scalar(*ar, *ai, yv, 0.0);
            Ok(Value::Complex(r, i))
        }
        (Value::Int(x), Value::Complex(br, bi)) => {
            let xv = x.to_f64();
            let (r, i) = complex_pow_scalar(xv, 0.0, *br, *bi);
            Ok(Value::Complex(r, i))
        }
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x.powf(*y))),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(x.to_f64().powf(*y))),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x.powf(y.to_f64()))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(x.to_f64().powf(y.to_f64()))),

        // Matrix-scalar cases (broadcasting)
        (Value::Tensor(m), Value::Num(s)) => {
            let data: Vec<f64> = m.data.iter().map(|x| x.powf(*s)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let scalar = s.to_f64();
            let data: Vec<f64> = m.data.iter().map(|x| x.powf(scalar)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Num(s), Value::Tensor(m)) => {
            let data: Vec<f64> = m.data.iter().map(|x| s.powf(*x)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Int(s), Value::Tensor(m)) => {
            let scalar = s.to_f64();
            let data: Vec<f64> = m.data.iter().map(|x| scalar.powf(*x)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }

        // Matrix-matrix case
        (Value::Tensor(m1), Value::Tensor(m2)) => {
            if m1.rows() != m2.rows() || m1.cols() != m2.cols() {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise power: {}x{} .^ {}x{}",
                    m1.rows(),
                    m1.cols(),
                    m2.rows(),
                    m2.cols()
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| x.powf(*y))
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m1.rows(), m1.cols())?))
        }

        // Complex tensor element-wise power
        (Value::ComplexTensor(m1), Value::ComplexTensor(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise power: {}x{} .^ {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(m1.data.len());
            for i in 0..m1.data.len() {
                let (ar, ai) = m1.data[i];
                let (br, bi) = m2.data[i];
                out.push(complex_pow_scalar(ar, ai, br, bi));
            }
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m1.rows, m1.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Num(s)) => {
            let out: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(ar, ai)| complex_pow_scalar(*ar, *ai, *s, 0.0))
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Int(s)) => {
            let sv = s.to_f64();
            let out: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(ar, ai)| complex_pow_scalar(*ar, *ai, sv, 0.0))
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Complex(br, bi)) => {
            let out: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(ar, ai)| complex_pow_scalar(*ar, *ai, *br, *bi))
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::Num(s), Value::ComplexTensor(m)) => {
            let out: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(br, bi)| complex_pow_scalar(*s, 0.0, *br, *bi))
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::Int(s), Value::ComplexTensor(m)) => {
            let sv = s.to_f64();
            let out: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(br, bi)| complex_pow_scalar(sv, 0.0, *br, *bi))
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::Complex(br, bi), Value::ComplexTensor(m)) => {
            let out: Vec<(f64, f64)> = m
                .data
                .iter()
                .map(|(er, ei)| complex_pow_scalar(*br, *bi, *er, *ei))
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }

        _ => Err(format!(
            "Element-wise power not supported for types: {a:?} .^ {b:?}"
        )),
    }
}

// Element-wise operations are not directly exposed as runtime builtins because they need
// to handle multiple types (Value enum variants). Instead, they are called directly from
// the interpreter and JIT compiler using the elementwise_* functions above.

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_elementwise_mul_scalars() {
        assert_eq!(
            block_on(elementwise_mul(&Value::Num(3.0), &Value::Num(4.0))).unwrap(),
            Value::Num(12.0)
        );
        assert_eq!(
            block_on(elementwise_mul(
                &Value::Int(runmat_builtins::IntValue::I32(3)),
                &Value::Num(4.5)
            ))
            .unwrap(),
            Value::Num(13.5)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_elementwise_mul_matrix_scalar() {
        let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let result = block_on(elementwise_mul(&Value::Tensor(matrix), &Value::Num(2.0))).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.data, vec![2.0, 4.0, 6.0, 8.0]);
            assert_eq!(m.rows(), 2);
            assert_eq!(m.cols(), 2);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_elementwise_mul_matrices() {
        let m1 = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let m2 = Tensor::new_2d(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let result = block_on(elementwise_mul(&Value::Tensor(m1), &Value::Tensor(m2))).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.data, vec![2.0, 6.0, 12.0, 20.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_elementwise_div_with_zero() {
        let result = block_on(elementwise_div(&Value::Num(5.0), &Value::Num(0.0))).unwrap();
        if let Value::Num(n) = result {
            assert!(n.is_infinite() && n.is_sign_positive());
        } else {
            panic!("Expected numeric result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_elementwise_pow() {
        let matrix = Tensor::new_2d(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let result = elementwise_pow(&Value::Tensor(matrix), &Value::Num(2.0)).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.data, vec![4.0, 9.0, 16.0, 25.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_dimension_mismatch() {
        let m1 = Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap();
        let m2 = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        assert!(block_on(elementwise_mul(&Value::Tensor(m1), &Value::Tensor(m2))).is_err());
    }
}
