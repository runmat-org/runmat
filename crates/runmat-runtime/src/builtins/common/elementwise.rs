//! Element-wise operations for matrices and scalars
//!
//! This module implements language-compatible element-wise operations (.*,  ./,  .^)
//! These operations work element-by-element on matrices and support scalar broadcasting.

use crate::builtins::common::matrix::matrix_power;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

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

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor_utils::is_scalar_tensor(t) => {
            Some(tensor_utils::tensor_value_f64(t, 0))
        }
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        Value::ComplexTensor(t) if tensor_utils::is_scalar_complex_tensor(t) => {
            let value = tensor_utils::complex_tensor_value_complex64(t, 0);
            Some((value.re, value.im))
        }
        _ => None,
    }
}

enum ComplexTensorValues<'a> {
    Raw(&'a [(f64, f64)]),
    Exact(Vec<num_complex::Complex64>),
}

impl ComplexTensorValues<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Raw(values) => values.len(),
            Self::Exact(values) => values.len(),
        }
    }

    fn value_at(&self, index: usize) -> (f64, f64) {
        match self {
            Self::Raw(values) => values[index],
            Self::Exact(values) => {
                let value = values[index];
                (value.re, value.im)
            }
        }
    }
}

fn complex_tensor_values(tensor: &runmat_builtins::ComplexTensor) -> ComplexTensorValues<'_> {
    if tensor.integer_data.is_some() {
        ComplexTensorValues::Exact(tensor_utils::complex_tensor_values_complex64(tensor))
    } else {
        ComplexTensorValues::Raw(&tensor.data)
    }
}

fn scalar_power_value(base: &Value, exponent: &Value) -> Option<Value> {
    let base_is_complex = matches!(base, Value::Complex(_, _) | Value::ComplexTensor(_));
    let exp_is_complex = matches!(exponent, Value::Complex(_, _) | Value::ComplexTensor(_));
    let base_val =
        scalar_complex_value(base).or_else(|| scalar_real_value(base).map(|v| (v, 0.0)))?;
    let exp_val =
        scalar_complex_value(exponent).or_else(|| scalar_real_value(exponent).map(|v| (v, 0.0)))?;
    let (br, bi) = base_val;
    let (er, ei) = exp_val;
    if base_is_complex || exp_is_complex || bi != 0.0 || ei != 0.0 {
        let (re, im) = complex_pow_scalar(br, bi, er, ei);
        return Some(Value::Complex(re, im));
    }
    let pow = br.powf(er);
    if pow.is_nan() {
        let (re, im) = complex_pow_scalar(br, 0.0, er, 0.0);
        Some(Value::Complex(re, im))
    } else {
        Some(Value::Num(pow))
    }
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
        Value::Int(value) => Ok(Value::Int(negate_integer_scalar(value.clone()))),
        Value::Bool(b) => Ok(Value::Bool(!b)), // Boolean negation
        Value::Tensor(m) => {
            if let Some(storage) = m.integer_storage() {
                return Tensor::new_integer(negate_integer_storage(storage), m.shape.clone())
                    .map(Value::Tensor);
            }
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| -x).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        _ => Err(format!("Negation not supported for type: -{a:?}")),
    }
}

fn negate_integer_scalar(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value.saturating_neg()),
        IntValue::I16(value) => IntValue::I16(value.saturating_neg()),
        IntValue::I32(value) => IntValue::I32(value.saturating_neg()),
        IntValue::I64(value) => IntValue::I64(value.saturating_neg()),
        IntValue::U8(_) => IntValue::U8(0),
        IntValue::U16(_) => IntValue::U16(0),
        IntValue::U32(_) => IntValue::U32(0),
        IntValue::U64(_) => IntValue::U64(0),
    }
}

fn negate_integer_storage(storage: &IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.iter().map(|value| value.saturating_neg()).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.iter().map(|value| value.saturating_neg()).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.iter().map(|value| value.saturating_neg()).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.iter().map(|value| value.saturating_neg()).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(vec![0; values.len()]),
        IntegerStorage::U16(values) => IntegerStorage::U16(vec![0; values.len()]),
        IntegerStorage::U32(values) => IntegerStorage::U32(vec![0; values.len()]),
        IntegerStorage::U64(values) => IntegerStorage::U64(vec![0; values.len()]),
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
    if let Some(result) = try_integer_binary(a, b, IntegerBinaryOp::Multiply, "times")? {
        return Ok(result);
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
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| x * s).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let scalar = s.to_f64();
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| x * scalar).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Num(s), Value::Tensor(m)) => {
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| s * x).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Int(s), Value::Tensor(m)) => {
            let scalar = s.to_f64();
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| scalar * x).collect();
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
            let lhs = tensor_utils::tensor_values_f64_cow(m1);
            let rhs = tensor_utils::tensor_values_f64_cow(m2);
            let data: Vec<f64> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x * y).collect();
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
            let lhs = complex_tensor_values(m1);
            let rhs = complex_tensor_values(m2);
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(lhs.len());
            for i in 0..lhs.len() {
                let (ar, ai) = lhs.value_at(i);
                let (br, bi) = rhs.value_at(i);
                out.push((ar * br - ai * bi, ar * bi + ai * br));
            }
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new(out, m1.shape.clone())
                    .map_err(|e| format!(".*: {e}"))?,
            ))
        }
        (Value::ComplexTensor(m), Value::Num(s)) => {
            let values = complex_tensor_values(m);
            let data: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (re, im) = values.value_at(index);
                    (re * s, im * s)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m.rows, m.cols)?,
            ))
        }
        (Value::Num(s), Value::ComplexTensor(m)) => {
            let values = complex_tensor_values(m);
            let data: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (re, im) = values.value_at(index);
                    (s * re, s * im)
                })
                .collect();
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
    if let Some(result) = try_integer_binary(a, b, IntegerBinaryOp::Divide, "rdivide")? {
        return Ok(result);
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
            let values = tensor_utils::tensor_values_f64_cow(m);
            if *s == 0.0 {
                let data: Vec<f64> = values.iter().map(|x| f64::INFINITY * x.signum()).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            } else {
                let data: Vec<f64> = values.iter().map(|x| x / s).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            }
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let scalar = s.to_f64();
            let values = tensor_utils::tensor_values_f64_cow(m);
            if scalar == 0.0 {
                let data: Vec<f64> = values.iter().map(|x| f64::INFINITY * x.signum()).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            } else {
                let data: Vec<f64> = values.iter().map(|x| x / scalar).collect();
                Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
            }
        }
        (Value::Num(s), Value::Tensor(m)) => {
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values
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
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values
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
            let lhs = tensor_utils::tensor_values_f64_cow(m1);
            let rhs = tensor_utils::tensor_values_f64_cow(m2);
            let data: Vec<f64> = lhs
                .iter()
                .zip(rhs.iter())
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
            let lhs = complex_tensor_values(m1);
            let rhs = complex_tensor_values(m2);
            let data: Vec<(f64, f64)> = (0..lhs.len())
                .map(|index| {
                    let (ar, ai) = lhs.value_at(index);
                    let (br, bi) = rhs.value_at(index);
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
            let values = complex_tensor_values(m);
            let data: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (re, im) = values.value_at(index);
                    (re / s, im / s)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, m.rows, m.cols)?,
            ))
        }
        (Value::Num(s), Value::ComplexTensor(m)) => {
            let values = complex_tensor_values(m);
            let data: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (br, bi) = values.value_at(index);
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
    if scalar_power_integer_candidate(a) && scalar_power_integer_candidate(b) {
        if let Some(result) = try_integer_binary(a, b, IntegerBinaryOp::Power, "power")? {
            return Ok(result);
        }
    }
    if let Some(result) = scalar_power_value(a, b) {
        return Ok(result);
    }
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
            let result = matrix_power(m, matrix_power_exponent_from_f64(*s)?)?;
            Ok(Value::Tensor(result))
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let result = matrix_power(m, matrix_power_exponent_from_int(s)?)?;
            Ok(Value::Tensor(result))
        }

        // Complex matrix^integer case
        (Value::ComplexTensor(m), Value::Num(s)) => {
            let result = crate::builtins::common::matrix::complex_matrix_power(
                m,
                matrix_power_exponent_from_f64(*s)?,
            )?;
            Ok(Value::ComplexTensor(result))
        }
        (Value::ComplexTensor(m), Value::Int(s)) => {
            let result = crate::builtins::common::matrix::complex_matrix_power(
                m,
                matrix_power_exponent_from_int(s)?,
            )?;
            Ok(Value::ComplexTensor(result))
        }

        // Other cases not supported for regular matrix power
        _ => Err(format!(
            "Power operation not supported for types: {a:?} ^ {b:?}"
        )),
    }
}

fn scalar_power_integer_candidate(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor_utils::is_scalar_tensor(tensor),
        Value::LogicalArray(array) => array.data.len() == 1,
        _ => false,
    }
}

fn matrix_power_exponent_from_f64(value: f64) -> Result<i32, String> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err("Matrix power requires integer exponent".to_string());
    }
    if value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err("Matrix power exponent is outside the supported int32 range".to_string());
    }
    Ok(value as i32)
}

fn matrix_power_exponent_from_int(value: &IntValue) -> Result<i32, String> {
    value
        .try_to_i32()
        .ok_or_else(|| "Matrix power exponent is outside the supported int32 range".to_string())
}

/// Element-wise power: A .^ B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
pub fn elementwise_pow(a: &Value, b: &Value) -> Result<Value, String> {
    if let Some(result) = try_integer_binary(a, b, IntegerBinaryOp::Power, "power")? {
        return Ok(result);
    }
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
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| x.powf(*s)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Tensor(m), Value::Int(s)) => {
            let scalar = s.to_f64();
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| x.powf(scalar)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Num(s), Value::Tensor(m)) => {
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| s.powf(*x)).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, m.rows(), m.cols())?))
        }
        (Value::Int(s), Value::Tensor(m)) => {
            let scalar = s.to_f64();
            let values = tensor_utils::tensor_values_f64_cow(m);
            let data: Vec<f64> = values.iter().map(|x| scalar.powf(*x)).collect();
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
            let lhs = tensor_utils::tensor_values_f64_cow(m1);
            let rhs = tensor_utils::tensor_values_f64_cow(m2);
            let data: Vec<f64> = lhs
                .iter()
                .zip(rhs.iter())
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
            let lhs = complex_tensor_values(m1);
            let rhs = complex_tensor_values(m2);
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(lhs.len());
            for i in 0..lhs.len() {
                let (ar, ai) = lhs.value_at(i);
                let (br, bi) = rhs.value_at(i);
                out.push(complex_pow_scalar(ar, ai, br, bi));
            }
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m1.rows, m1.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Num(s)) => {
            let values = complex_tensor_values(m);
            let out: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (ar, ai) = values.value_at(index);
                    complex_pow_scalar(ar, ai, *s, 0.0)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Int(s)) => {
            let sv = s.to_f64();
            let values = complex_tensor_values(m);
            let out: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (ar, ai) = values.value_at(index);
                    complex_pow_scalar(ar, ai, sv, 0.0)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::ComplexTensor(m), Value::Complex(br, bi)) => {
            let values = complex_tensor_values(m);
            let out: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (ar, ai) = values.value_at(index);
                    complex_pow_scalar(ar, ai, *br, *bi)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::Num(s), Value::ComplexTensor(m)) => {
            let values = complex_tensor_values(m);
            let out: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (br, bi) = values.value_at(index);
                    complex_pow_scalar(*s, 0.0, br, bi)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::Int(s), Value::ComplexTensor(m)) => {
            let sv = s.to_f64();
            let values = complex_tensor_values(m);
            let out: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (br, bi) = values.value_at(index);
                    complex_pow_scalar(sv, 0.0, br, bi)
                })
                .collect();
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(out, m.rows, m.cols)?,
            ))
        }
        (Value::Complex(br, bi), Value::ComplexTensor(m)) => {
            let values = complex_tensor_values(m);
            let out: Vec<(f64, f64)> = (0..values.len())
                .map(|index| {
                    let (er, ei) = values.value_at(index);
                    complex_pow_scalar(*br, *bi, er, ei)
                })
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

    #[test]
    fn matrix_power_typed_exponent_parser_is_exact() {
        assert_eq!(
            matrix_power_exponent_from_int(&IntValue::U16(7)).unwrap(),
            7
        );
        assert!(matrix_power_exponent_from_int(&IntValue::U64(u64::MAX)).is_err());
        assert!(matrix_power_exponent_from_f64(f64::INFINITY).is_err());
        assert!(matrix_power_exponent_from_f64(i32::MAX as f64 + 1.0).is_err());
    }

    #[test]
    fn scalar_power_reads_typed_complex_integer_storage_exactly() {
        let storage = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I16(vec![4]),
        )
        .expect("complex integer storage");
        let mut tensor = runmat_builtins::ComplexTensor::new_integer(storage, vec![1, 1])
            .expect("complex tensor");
        tensor.data.clear();

        let result = scalar_power_value(&Value::ComplexTensor(tensor), &Value::Num(1.0))
            .expect("scalar power");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 3.0).abs() < 1e-12);
                assert!((im - 4.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    fn mirrorless_complex_integer_tensor(
        real: Vec<i16>,
        imag: Vec<i16>,
        shape: Vec<usize>,
    ) -> runmat_builtins::ComplexTensor {
        let storage = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::I16(real),
            IntegerStorage::I16(imag),
        )
        .expect("complex integer storage");
        let mut tensor =
            runmat_builtins::ComplexTensor::new_integer(storage, shape).expect("complex tensor");
        tensor.data.clear();
        tensor
    }

    #[test]
    fn elementwise_mul_reads_typed_complex_integer_storage_exactly() {
        let lhs = mirrorless_complex_integer_tensor(vec![3, -2], vec![4, 5], vec![1, 2]);
        let rhs = mirrorless_complex_integer_tensor(vec![1, 6], vec![-2, 1], vec![1, 2]);

        let Value::ComplexTensor(result) = block_on(elementwise_mul(
            &Value::ComplexTensor(lhs),
            &Value::ComplexTensor(rhs),
        ))
        .expect("mul") else {
            panic!("expected complex tensor");
        };
        assert_eq!(result.shape, vec![1, 2]);
        assert_eq!(result.data, vec![(11.0, -2.0), (-17.0, 28.0)]);
    }

    #[test]
    fn elementwise_div_reads_typed_complex_integer_storage_exactly() {
        let lhs = mirrorless_complex_integer_tensor(vec![3, -2], vec![4, 5], vec![1, 2]);

        let Value::ComplexTensor(result) = block_on(elementwise_div(
            &Value::ComplexTensor(lhs),
            &Value::Num(2.0),
        ))
        .expect("div") else {
            panic!("expected complex tensor");
        };
        assert_eq!(result.shape, vec![1, 2]);
        assert_eq!(result.data, vec![(1.5, 2.0), (-1.0, 2.5)]);
    }

    #[test]
    fn elementwise_pow_reads_typed_complex_integer_storage_exactly() {
        let base = mirrorless_complex_integer_tensor(vec![3, 1], vec![4, -2], vec![1, 2]);

        let Value::ComplexTensor(result) =
            elementwise_pow(&Value::ComplexTensor(base), &Value::Num(2.0)).expect("pow")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(result.shape, vec![1, 2]);
        assert!((result.data[0].0 + 7.0).abs() < 1e-12);
        assert!((result.data[0].1 - 24.0).abs() < 1e-12);
        assert!((result.data[1].0 + 3.0).abs() < 1e-12);
        assert!((result.data[1].1 + 4.0).abs() < 1e-12);
    }

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
            Value::Int(runmat_builtins::IntValue::I32(14))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_elementwise_mul_matrix_scalar() {
        let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let result = block_on(elementwise_mul(&Value::Tensor(matrix), &Value::Num(2.0))).unwrap();

        if let Value::Tensor(m) = result {
            assert_eq!(m.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
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
            assert_eq!(m.materialize_f64(), vec![2.0, 6.0, 12.0, 20.0]);
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
            assert_eq!(m.materialize_f64(), vec![4.0, 9.0, 16.0, 25.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn elementwise_neg_preserves_all_typed_integer_classes_and_shape() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MIN, -2, 0, i8::MAX]),
                IntegerStorage::I8(vec![i8::MAX, 2, 0, -i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, -2, 0, i16::MAX]),
                IntegerStorage::I16(vec![i16::MAX, 2, 0, -i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, -2, 0, i32::MAX]),
                IntegerStorage::I32(vec![i32::MAX, 2, 0, -i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, -2, 0, i64::MAX]),
                IntegerStorage::I64(vec![i64::MAX, 2, 0, -i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![0, 2, u8::MAX]),
                IntegerStorage::U8(vec![0, 0, 0]),
            ),
            (
                IntegerStorage::U16(vec![0, 2, u16::MAX]),
                IntegerStorage::U16(vec![0, 0, 0]),
            ),
            (
                IntegerStorage::U32(vec![0, 2, u32::MAX]),
                IntegerStorage::U32(vec![0, 0, 0]),
            ),
            (
                IntegerStorage::U64(vec![0, 2, u64::MAX]),
                IntegerStorage::U64(vec![0, 0, 0]),
            ),
        ];
        for (input, expected) in cases {
            let shape = vec![1, expected.len(), 1];
            let tensor = Tensor::new_integer(input, shape.clone()).expect("integer tensor");
            let Value::Tensor(result) = elementwise_neg(&Value::Tensor(tensor)).expect("neg")
            else {
                panic!("expected tensor");
            };
            assert_eq!(result.shape, shape);
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn elementwise_neg_preserves_scalar_integer_class() {
        assert_eq!(
            elementwise_neg(&Value::Int(IntValue::I64(i64::MIN))).expect("neg"),
            Value::Int(IntValue::I64(i64::MAX))
        );
        assert_eq!(
            elementwise_neg(&Value::Int(IntValue::U64(u64::MAX))).expect("neg"),
            Value::Int(IntValue::U64(0))
        );
    }

    #[test]
    fn transitional_elementwise_helpers_preserve_exact_integer_storage() {
        let lhs = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 63) + 1]),
            vec![1, 2],
        )
        .expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).expect("rhs");

        let Value::Tensor(product) = block_on(elementwise_mul(
            &Value::Tensor(lhs.clone()),
            &Value::Tensor(rhs),
        ))
        .expect("mul") else {
            panic!("expected integer tensor product");
        };
        assert_eq!(
            product.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX]))
        );

        let Value::Tensor(quotient) =
            block_on(elementwise_div(&Value::Tensor(lhs), &Value::Num(2.0))).expect("div")
        else {
            panic!("expected integer tensor quotient");
        };
        assert_eq!(
            quotient.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 63, (1_u64 << 62) + 1]))
        );
    }

    #[test]
    fn elementwise_integer_operations_read_typed_storage_not_poisoned_mirrors() {
        let input = Tensor::new_integer(IntegerStorage::I64(vec![2, 3]), vec![1, 2])
            .expect("integer tensor");

        let Value::Tensor(product) = block_on(elementwise_mul(
            &Value::Tensor(input.clone()),
            &Value::Num(0.5),
        ))
        .expect("product") else {
            panic!("expected tensor");
        };
        assert_eq!(
            product.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 2]))
        );

        let Value::Tensor(quotient) = block_on(elementwise_div(
            &Value::Num(6.0),
            &Value::Tensor(input.clone()),
        ))
        .expect("quotient") else {
            panic!("expected tensor");
        };
        assert_eq!(
            quotient.integer_storage(),
            Some(&IntegerStorage::I64(vec![3, 2]))
        );

        let Value::Tensor(powered) =
            elementwise_pow(&Value::Tensor(input), &Value::Num(0.5)).expect("power")
        else {
            panic!("expected tensor");
        };
        assert_eq!(
            powered.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 2]))
        );
    }

    #[test]
    fn transitional_power_helpers_preserve_exact_scalar_and_array_integers() {
        let scalar_power =
            power(&Value::Int(IntValue::U64(u64::MAX)), &Value::Num(1.0)).expect("scalar power");
        assert_eq!(scalar_power, Value::Int(IntValue::U64(u64::MAX)));

        let scalar_tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("scalar tensor");
        let scalar_tensor_power =
            power(&Value::Tensor(scalar_tensor), &Value::Num(1.0)).expect("tensor scalar power");
        assert_eq!(scalar_tensor_power, Value::Int(IntValue::U64(u64::MAX)));

        let complex_base =
            Tensor::new_integer(IntegerStorage::U8(vec![3]), vec![1, 1]).expect("complex base");
        let complex_power = power(&Value::Tensor(complex_base), &Value::Complex(1.0, 0.0))
            .expect("complex exponent power");
        let Value::Complex(re, im) = complex_power else {
            panic!("expected complex scalar");
        };
        assert!((re - 3.0).abs() < 1e-12);
        assert_eq!(im, 0.0);

        let base =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 2]), vec![1, 2]).expect("base");
        let exponent =
            Tensor::new_integer(IntegerStorage::U64(vec![1, 64]), vec![1, 2]).expect("exponent");
        let Value::Tensor(result) =
            elementwise_pow(&Value::Tensor(base), &Value::Tensor(exponent)).expect("pow")
        else {
            panic!("expected integer tensor power");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_dimension_mismatch() {
        let m1 = Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap();
        let m2 = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        assert!(block_on(elementwise_mul(&Value::Tensor(m1), &Value::Tensor(m2))).is_err());
    }
}
