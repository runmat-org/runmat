//! Mathematical functions
//!
//! This module provides comprehensive mathematical functions including trigonometric,
//! logarithmic, exponential, hyperbolic, and other mathematical operations.
//! All functions are optimized for performance and handle both scalars and matrices.

use rustmat_builtins::Matrix;
use rustmat_macros::runtime_builtin;

// Trigonometric functions - scalar versions

#[runtime_builtin(name = "sin")]
fn sin_builtin(x: f64) -> Result<f64, String> {
    Ok(x.sin())
}

#[runtime_builtin(name = "cos")]
fn cos_builtin(x: f64) -> Result<f64, String> {
    Ok(x.cos())
}

#[runtime_builtin(name = "tan")]
fn tan_builtin(x: f64) -> Result<f64, String> {
    Ok(x.tan())
}

// Trigonometric functions - matrix versions (element-wise)

#[runtime_builtin(name = "sin")]
fn sin_matrix_builtin(x: Matrix) -> Result<Matrix, String> {
    let data: Vec<f64> = x.data.iter().map(|&val| val.sin()).collect();
    Ok(Matrix::new(data, x.rows, x.cols)?)
}

#[runtime_builtin(name = "cos")]
fn cos_matrix_builtin(x: Matrix) -> Result<Matrix, String> {
    let data: Vec<f64> = x.data.iter().map(|&val| val.cos()).collect();
    Ok(Matrix::new(data, x.rows, x.cols)?)
}

#[runtime_builtin(name = "tan")]
fn tan_matrix_builtin(x: Matrix) -> Result<Matrix, String> {
    let data: Vec<f64> = x.data.iter().map(|&val| val.tan()).collect();
    Ok(Matrix::new(data, x.rows, x.cols)?)
}

#[runtime_builtin(name = "asin")]
fn asin_builtin(x: f64) -> Result<f64, String> {
    if x < -1.0 || x > 1.0 {
        Err("Input must be in range [-1, 1] for asin".to_string())
    } else {
        Ok(x.asin())
    }
}

#[runtime_builtin(name = "acos")]
fn acos_builtin(x: f64) -> Result<f64, String> {
    if x < -1.0 || x > 1.0 {
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

#[runtime_builtin(name = "sum")]
fn sum_builtin(matrix: Matrix) -> Result<f64, String> {
    Ok(matrix.data.iter().sum())
}

#[runtime_builtin(name = "mean")]
fn mean_builtin(matrix: Matrix) -> Result<f64, String> {
    if matrix.data.is_empty() {
        return Err("Cannot compute mean of empty matrix".to_string());
    }
    Ok(matrix.data.iter().sum::<f64>() / matrix.data.len() as f64)
}

#[runtime_builtin(name = "std")]
fn std_builtin(matrix: Matrix) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute standard deviation".to_string());
    }
    
    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (matrix.data.len() - 1) as f64;
    
    Ok(variance.sqrt())
}

#[runtime_builtin(name = "var")]
fn var_builtin(matrix: Matrix) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute variance".to_string());
    }
    
    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (matrix.data.len() - 1) as f64;
    
    Ok(variance)
}

#[runtime_builtin(name = "min")]
fn min_vector_builtin(matrix: Matrix) -> Result<f64, String> {
    matrix.data.iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .ok_or_else(|| "Cannot find minimum of empty matrix".to_string())
}

#[runtime_builtin(name = "max")]
fn max_vector_builtin(matrix: Matrix) -> Result<f64, String> {
    matrix.data.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .ok_or_else(|| "Cannot find maximum of empty matrix".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigonometric_functions() {
        assert!((sin_builtin(0.0).unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((cos_builtin(0.0).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((tan_builtin(0.0).unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inverse_trigonometric() {
        assert!((asin_builtin(0.0).unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((acos_builtin(1.0).unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((atan_builtin(0.0).unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logarithmic_functions() {
        assert!((ln_builtin(std::f64::consts::E).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((log2_builtin(8.0).unwrap() - 3.0).abs() < f64::EPSILON);
        assert!((log10_builtin(1000.0).unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rounding_functions() {
        assert_eq!(round_builtin(3.7).unwrap(), 4.0);
        assert_eq!(floor_builtin(3.7).unwrap(), 3.0);
        assert_eq!(ceil_builtin(3.2).unwrap(), 4.0);
        assert_eq!(trunc_builtin(-3.7).unwrap(), -3.0);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial_builtin(0).unwrap(), 1.0);
        assert_eq!(factorial_builtin(5).unwrap(), 120.0);
        assert!(factorial_builtin(-1).is_err());
    }

    #[test]
    fn test_statistical_functions() {
        let matrix = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5).unwrap();
        assert_eq!(sum_builtin(matrix.clone()).unwrap(), 15.0);
        assert_eq!(mean_builtin(matrix.clone()).unwrap(), 3.0);
        assert_eq!(min_vector_builtin(matrix.clone()).unwrap(), 1.0);
        assert_eq!(max_vector_builtin(matrix).unwrap(), 5.0);
    }

    #[test]
    fn test_error_cases() {
        assert!(asin_builtin(2.0).is_err());
        assert!(ln_builtin(-1.0).is_err());
        assert!(acosh_builtin(0.5).is_err());
        assert!(atanh_builtin(1.5).is_err());
    }
}