//! Mathematical constants and special values
//!
//! This module provides language-compatible mathematical constants like pi, e, inf, nan, etc.
//! These are registered as global constants that can be accessed as variables.

#[cfg(test)]
#[allow(unused_imports)]
use runmat_builtins::Value;
#[cfg(test)]
#[allow(unused_imports)]
use runmat_macros::runtime_builtin;

/// Logical scalar true (language-compatible)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "true")]
fn true_builtin() -> Result<bool, String> {
    Ok(true)
}

/// Logical scalar false (language-compatible)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "false")]
fn false_builtin() -> Result<bool, String> {
    Ok(false)
}

/// Mathematical constant π (pi)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "pi")]
fn pi_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::PI)
}

/// Mathematical constant e (Euler's number)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "e")]
fn e_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::E)
}

/// Positive infinity
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "inf")]
fn inf_builtin() -> Result<f64, String> {
    Ok(f64::INFINITY)
}

/// Not-a-Number (NaN)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "nan")]
fn nan_builtin() -> Result<f64, String> {
    Ok(f64::NAN)
}

/// Machine epsilon for double precision
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "eps")]
fn eps_builtin() -> Result<f64, String> {
    Ok(f64::EPSILON)
}

/// Alternative name for infinity (language compatibility)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "Inf")]
fn inf_language_builtin() -> Result<f64, String> {
    Ok(f64::INFINITY)
}

/// Alternative name for NaN (language compatibility)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "NaN")]
fn nan_language_builtin() -> Result<f64, String> {
    Ok(f64::NAN)
}

/// Mathematical constant √2
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "sqrt2")]
fn sqrt2_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::SQRT_2)
}

/// Mathematical constant ln(2)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "log2")]
fn log2_const_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::LN_2)
}

/// Mathematical constant ln(10)
#[cfg(test)]
#[allow(dead_code)]
#[runmat_macros::runtime_builtin(name = "log10")]
fn log10_const_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::LN_10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_constant() {
        let result = pi_builtin().unwrap();
        assert!((result - std::f64::consts::PI).abs() < f64::EPSILON);
    }

    #[test]
    fn test_e_constant() {
        let result = e_builtin().unwrap();
        assert!((result - std::f64::consts::E).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inf_constant() {
        let result = inf_builtin().unwrap();
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[test]
    fn test_nan_constant() {
        let result = nan_builtin().unwrap();
        assert!(result.is_nan());
    }

    #[test]
    fn test_eps_constant() {
        let result = eps_builtin().unwrap();
        assert_eq!(result, f64::EPSILON);
    }

    #[test]
    fn test_true_false_builtins() {
        assert!(true_builtin().unwrap());
        assert!(!false_builtin().unwrap());
    }

    #[test]
    fn test_language_compatibility() {
        // Test language-style names
        assert_eq!(inf_language_builtin().unwrap(), inf_builtin().unwrap());
        assert!(nan_language_builtin().unwrap().is_nan());
    }
}

// Inventory registrations moved to builtins/constants.
