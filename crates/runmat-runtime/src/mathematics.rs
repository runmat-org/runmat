//! Mathematical functions
//!
//! This module provides comprehensive mathematical functions including trigonometric,
//! logarithmic, exponential, hyperbolic, and other mathematical operations.
//! All functions are optimized for performance and handle both scalars and matrices.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

// Trigonometric functions - full MATLAB-compatible variants over Value
fn sin_complex(re: f64, im: f64) -> (f64, f64) {
    (re.sin() * im.cosh(), re.cos() * im.sinh())
}

fn cos_complex(re: f64, im: f64) -> (f64, f64) {
    (re.cos() * im.cosh(), -(re.sin() * im.sinh()))
}

fn div_complex(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
    let denom = b_re * b_re + b_im * b_im;
    (
        (a_re * b_re + a_im * b_im) / denom,
        (a_im * b_re - a_re * b_im) / denom,
    )
}

#[runtime_builtin(
    name = "cos",
    category = "math/trigonometry",
    summary = "Cosine of input in radians (element-wise).",
    examples = "y = cos(0)",
    keywords = "cosine,trig,angle",
    related = "sin,tan",
    accel = "unary"
)]
fn cos_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.unary_cos(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("cos: unsupported for gpuArray".to_string())
        }
        Value::Num(n) => Ok(Value::Num(n.cos())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().cos())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.cos()).collect();
            Ok(Value::Tensor(Tensor::new(data, t.shape.clone())?))
        }
        Value::Complex(re, im) => {
            let (r, i) = cos_complex(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => {
            let out: Vec<(f64, f64)> = ct
                .data
                .iter()
                .map(|&(re, im)| cos_complex(re, im))
                .collect();
            Ok(Value::ComplexTensor(runmat_builtins::ComplexTensor::new(
                out,
                ct.shape.clone(),
            )?))
        }
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0f64.cos() } else { 0.0f64.cos() })
                .collect();
            Ok(Value::Tensor(Tensor::new(data, la.shape.clone())?))
        }
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| (ch as u32 as f64).cos()).collect();
            Ok(Value::Tensor(Tensor::new(data, vec![ca.rows, ca.cols])?))
        }
        Value::String(_) | Value::StringArray(_) => Err("cos: expected numeric input".to_string()),
        other => Err(format!("cos: unsupported input {other:?}")),
    }
}

#[runtime_builtin(
    name = "tan",
    category = "math/trigonometry",
    summary = "Tangent of input in radians (element-wise).",
    examples = "y = tan(pi/4)",
    keywords = "tangent,trig,angle",
    related = "sin,cos"
)]
fn tan_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::Num(n) => Ok(Value::Num(n.tan())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().tan())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.tan()).collect();
            Ok(Value::Tensor(Tensor::new(data, t.shape.clone())?))
        }
        Value::Complex(re, im) => {
            let (sr, si) = sin_complex(re, im);
            let (cr, ci) = cos_complex(re, im);
            let (r, i) = div_complex(sr, si, cr, ci);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => {
            let out: Vec<(f64, f64)> = ct
                .data
                .iter()
                .map(|&(re, im)| {
                    let (sr, si) = sin_complex(re, im);
                    let (cr, ci) = cos_complex(re, im);
                    div_complex(sr, si, cr, ci)
                })
                .collect();
            Ok(Value::ComplexTensor(runmat_builtins::ComplexTensor::new(
                out,
                ct.shape.clone(),
            )?))
        }
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0f64.tan() } else { 0.0f64.tan() })
                .collect();
            Ok(Value::Tensor(Tensor::new(data, la.shape.clone())?))
        }
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| (ch as u32 as f64).tan()).collect();
            Ok(Value::Tensor(Tensor::new(data, vec![ca.rows, ca.cols])?))
        }
        Value::String(_) | Value::StringArray(_) => Err("tan: expected numeric input".to_string()),
        Value::GpuTensor(_) => Err("tan: unsupported for gpuArray".to_string()),
        other => Err(format!("tan: unsupported input {other:?}")),
    }
}

#[runtime_builtin(name = "asin")]
fn asin_builtin(x: f64) -> Result<f64, String> {
    if !(-1.0..=1.0).contains(&x) {
        Err("Input must be in range [-1, 1] for asin".to_string())
    } else {
        Ok(x.asin())
    }
}

#[runtime_builtin(name = "acos")]
fn acos_builtin(x: f64) -> Result<f64, String> {
    if !(-1.0..=1.0).contains(&x) {
        Err("Input must be in range [-1, 1] for acos".to_string())
    } else {
        Ok(x.acos())
    }
}

#[runtime_builtin(name = "atan")]
fn atan_builtin(x: f64) -> Result<f64, String> {
    Ok(x.atan())
}

#[runtime_builtin(name = "atan2")]
fn atan2_builtin(y: f64, x: f64) -> Result<f64, String> {
    Ok(y.atan2(x))
}

// Hyperbolic functions

#[runtime_builtin(name = "sinh")]
fn sinh_builtin(x: f64) -> Result<f64, String> {
    Ok(x.sinh())
}

#[runtime_builtin(name = "cosh")]
fn cosh_builtin(x: f64) -> Result<f64, String> {
    Ok(x.cosh())
}

#[runtime_builtin(name = "tanh")]
fn tanh_builtin(x: f64) -> Result<f64, String> {
    Ok(x.tanh())
}

#[runtime_builtin(name = "asinh")]
fn asinh_builtin(x: f64) -> Result<f64, String> {
    Ok(x.asinh())
}

#[runtime_builtin(name = "acosh")]
fn acosh_builtin(x: f64) -> Result<f64, String> {
    if x < 1.0 {
        Err("Input must be >= 1 for acosh".to_string())
    } else {
        Ok(x.acosh())
    }
}

#[runtime_builtin(name = "atanh")]
fn atanh_builtin(x: f64) -> Result<f64, String> {
    if x <= -1.0 || x >= 1.0 {
        Err("Input must be in range (-1, 1) for atanh".to_string())
    } else {
        Ok(x.atanh())
    }
}

// Logarithmic and exponential functions

#[runtime_builtin(name = "ln")]
fn ln_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for ln".to_string())
    } else {
        Ok(x.ln())
    }
}

#[runtime_builtin(name = "exp", accel = "unary")]
fn exp_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.unary_exp(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("exp: unsupported for gpuArray".to_string())
        }
        Value::Num(n) => Ok(Value::Num(n.exp())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().exp())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.exp()).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, t.rows(), t.cols())?))
        }
        other => Err(format!("exp: unsupported input {other:?}")),
    }
}

#[runtime_builtin(name = "log2")]
fn log2_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for log2".to_string())
    } else {
        Ok(x.log2())
    }
}

#[runtime_builtin(name = "log10")]
fn log10_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for log10".to_string())
    } else {
        Ok(x.log10())
    }
}

#[runtime_builtin(name = "exp2")]
fn exp2_builtin(x: f64) -> Result<f64, String> {
    Ok(2.0_f64.powf(x))
}

#[runtime_builtin(name = "exp10")]
fn exp10_builtin(x: f64) -> Result<f64, String> {
    Ok(10.0_f64.powf(x))
}

#[runtime_builtin(name = "pow")]
fn pow_builtin(base: f64, exponent: f64) -> Result<f64, String> {
    Ok(base.powf(exponent))
}

// Basic math functions
#[runtime_builtin(name = "abs", accel = "unary")]
fn abs_runtime_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.unary_abs(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("abs: unsupported for gpuArray".to_string())
        }
        Value::Num(n) => Ok(Value::Num(n.abs())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().abs())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.abs()).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, t.rows(), t.cols())?))
        }
        Value::Complex(re, im) => Ok(Value::Num((re * re + im * im).sqrt())),
        Value::ComplexTensor(ct) => {
            let data: Vec<f64> = ct
                .data
                .iter()
                .map(|(re, im)| (re * re + im * im).sqrt())
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, ct.rows, ct.cols)?))
        }
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let (rows, cols) = match la.shape.len() {
                0 => (0, 0),
                1 => (1, la.shape[0]),
                _ => (la.shape[0], la.shape[1]),
            };
            Ok(Value::Tensor(Tensor::new_2d(data, rows, cols)?))
        }
        other => Err(format!("abs: unsupported input {other:?}")),
    }
}

// Rounding and related functions

#[runtime_builtin(name = "round")]
fn round_builtin(x: f64) -> Result<f64, String> {
    Ok(x.round())
}

#[runtime_builtin(name = "floor")]
fn floor_builtin(x: f64) -> Result<f64, String> {
    Ok(x.floor())
}

#[runtime_builtin(name = "ceil")]
fn ceil_builtin(x: f64) -> Result<f64, String> {
    Ok(x.ceil())
}

#[runtime_builtin(name = "trunc")]
fn trunc_builtin(x: f64) -> Result<f64, String> {
    Ok(x.trunc())
}

#[runtime_builtin(name = "fract")]
fn fract_builtin(x: f64) -> Result<f64, String> {
    Ok(x.fract())
}

#[runtime_builtin(name = "sign")]
fn sign_builtin(x: f64) -> Result<f64, String> {
    if x > 0.0 {
        Ok(1.0)
    } else if x < 0.0 {
        Ok(-1.0)
    } else {
        Ok(0.0)
    }
}

// Complex number support (basic)

#[runtime_builtin(name = "real")]
fn real_builtin(x: f64) -> Result<f64, String> {
    Ok(x) // For real numbers, real part is the number itself
}

#[runtime_builtin(name = "imag")]
fn imag_builtin(_x: f64) -> Result<f64, String> {
    Ok(0.0) // For real numbers, imaginary part is always 0
}

#[runtime_builtin(name = "angle")]
fn angle_builtin(x: f64) -> Result<f64, String> {
    if x >= 0.0 {
        Ok(0.0)
    } else {
        Ok(std::f64::consts::PI)
    }
}

// Special functions

#[runtime_builtin(name = "gamma")]
fn gamma_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("Gamma function undefined for non-positive integers".to_string());
    }

    // Stirling's approximation for large x
    if x > 10.0 {
        let term1 = (2.0 * std::f64::consts::PI / x).sqrt();
        let term2 = (x / std::f64::consts::E).powf(x);
        return Ok(term1 * term2);
    }

    // For smaller values, use recursive relation Γ(x+1) = x * Γ(x)
    if x < 1.0 {
        return Ok(gamma_builtin(x + 1.0)? / x);
    }

    // Simple approximation for 1 <= x <= 10
    let mut result = 1.0;
    let mut n = x;
    while n > 2.0 {
        n -= 1.0;
        result *= n;
    }

    Ok(result)
}

#[runtime_builtin(name = "factorial")]
fn factorial_builtin(n: i32) -> Result<f64, String> {
    if n < 0 {
        return Err("Factorial undefined for negative numbers".to_string());
    }

    if n > 170 {
        return Err("Factorial overflow for n > 170".to_string());
    }

    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }

    Ok(result)
}

// Statistical functions

#[cfg(test)]
#[allow(dead_code)]
fn sum_builtin(matrix: Tensor) -> Result<f64, String> {
    Ok(matrix.data.iter().sum())
}

#[cfg(test)]
#[allow(dead_code)]
fn mean_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.is_empty() {
        return Err("Cannot compute mean of empty matrix".to_string());
    }
    Ok(matrix.data.iter().sum::<f64>() / matrix.data.len() as f64)
}

#[runtime_builtin(name = "std")]
fn std_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute standard deviation".to_string());
    }

    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (matrix.data.len() - 1) as f64;

    Ok(variance.sqrt())
}

#[runtime_builtin(name = "var")]
fn var_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute variance".to_string());
    }

    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (matrix.data.len() - 1) as f64;

    Ok(variance)
}

#[runtime_builtin(name = "sqrt", accel = "unary")]
fn sqrt_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.unary_sqrt(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("sqrt: unsupported for gpuArray".to_string())
        }
        Value::Num(n) => Ok(Value::Num(n.sqrt())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().sqrt())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.sqrt()).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, t.rows(), t.cols())?))
        }
        other => Err(format!("sqrt: unsupported input {other:?}")),
    }
}

// Unit tests for mathematics live under crates/runmat-runtime/tests/mathematics.rs
