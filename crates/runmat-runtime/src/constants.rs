//! Mathematical constants and special values
//!
//! This module provides MATLAB-compatible mathematical constants like pi, e, inf, nan, etc.
//! These are registered as global constants that can be accessed as variables.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

/// Mathematical constant π (pi)
#[runtime_builtin(name = "pi")]
fn pi_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::PI)
}

/// Mathematical constant e (Euler's number)
#[runtime_builtin(name = "e")]
fn e_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::E)
}

/// Positive infinity
#[runtime_builtin(name = "inf")]
fn inf_builtin() -> Result<f64, String> {
    Ok(f64::INFINITY)
}

/// Not-a-Number (NaN)
#[runtime_builtin(name = "nan")]
fn nan_builtin() -> Result<f64, String> {
    Ok(f64::NAN)
}

/// Machine epsilon for double precision
#[runtime_builtin(name = "eps")]
fn eps_builtin() -> Result<f64, String> {
    Ok(f64::EPSILON)
}

/// Alternative name for infinity (MATLAB compatibility)
#[runtime_builtin(name = "Inf")]
fn inf_matlab_builtin() -> Result<f64, String> {
    Ok(f64::INFINITY)
}

/// Alternative name for NaN (MATLAB compatibility)
#[runtime_builtin(name = "NaN")]
fn nan_matlab_builtin() -> Result<f64, String> {
    Ok(f64::NAN)
}

/// Mathematical constant √2
#[runtime_builtin(name = "sqrt2")]
fn sqrt2_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::SQRT_2)
}

/// Mathematical constant ln(2)
#[runtime_builtin(name = "log2")]
fn log2_const_builtin() -> Result<f64, String> {
    Ok(std::f64::consts::LN_2)
}

/// Mathematical constant ln(10)
#[runtime_builtin(name = "log10")]
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
    fn test_matlab_compatibility() {
        // Test MATLAB-style names
        assert_eq!(inf_matlab_builtin().unwrap(), inf_builtin().unwrap());
        assert!(nan_matlab_builtin().unwrap().is_nan());
    }
}

// Register constants that can be accessed as variables (not functions)
runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "pi",
        value: Value::Num(std::f64::consts::PI),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "e",
        value: Value::Num(std::f64::consts::E),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "inf",
        value: Value::Num(f64::INFINITY),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "Inf",
        value: Value::Num(f64::INFINITY),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "nan",
        value: Value::Num(f64::NAN),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "NaN",
        value: Value::Num(f64::NAN),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "eps",
        value: Value::Num(f64::EPSILON),
    }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant {
        name: "sqrt2",
        value: Value::Num(std::f64::consts::SQRT_2),
    }
}
