//! Element-wise operations for matrices and scalars
//!
//! This module implements MATLAB-compatible element-wise operations (.*,  ./,  .^)
//! These operations work element-by-element on matrices and support scalar broadcasting.

use crate::matrix::matrix_power;
use runmat_builtins::{Matrix, Value};

/// Element-wise negation: -A
/// Supports scalars and matrices
pub fn elementwise_neg(a: &Value) -> Result<Value, String> {
    match a {
        Value::Num(x) => Ok(Value::Num(-x)),
        Value::Int(x) => Ok(Value::Int(-x)),
        Value::Bool(b) => Ok(Value::Bool(!b)), // Boolean negation
        Value::Matrix(m) => {
            let data: Vec<f64> = m.data.iter().map(|x| -x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        _ => Err(format!("Negation not supported for type: -{a:?}")),
    }
}

/// Element-wise multiplication: A .* B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
pub fn elementwise_mul(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x * y)),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(*x as f64 * y)),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x * (*y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(*x as f64 * *y as f64)),

        // Matrix-scalar cases (broadcasting)
        (Value::Matrix(m), Value::Num(s)) => {
            let data: Vec<f64> = m.data.iter().map(|x| x * s).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Matrix(m), Value::Int(s)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| x * scalar).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Num(s), Value::Matrix(m)) => {
            let data: Vec<f64> = m.data.iter().map(|x| s * x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Int(s), Value::Matrix(m)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| scalar * x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }

        // Matrix-matrix case
        (Value::Matrix(m1), Value::Matrix(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise multiplication: {}x{} .* {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| x * y)
                .collect();
            Ok(Value::Matrix(Matrix::new(data, m1.rows, m1.cols)?))
        }

        _ => Err(format!(
            "Element-wise multiplication not supported for types: {a:?} .* {b:?}"
        )),
    }
}

/// Element-wise addition: A + B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
pub fn elementwise_add(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x + y)),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(*x as f64 + y)),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x + (*y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(*x as f64 + *y as f64)),

        // Matrix-scalar cases (broadcasting)
        (Value::Matrix(m), Value::Num(s)) => {
            let data: Vec<f64> = m.data.iter().map(|x| x + s).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Matrix(m), Value::Int(s)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| x + scalar).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Num(s), Value::Matrix(m)) => {
            let data: Vec<f64> = m.data.iter().map(|x| s + x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Int(s), Value::Matrix(m)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| scalar + x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }

        // Matrix-matrix case
        (Value::Matrix(m1), Value::Matrix(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for addition: {}x{} + {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| x + y)
                .collect();
            Ok(Value::Matrix(Matrix::new(data, m1.rows, m1.cols)?))
        }

        _ => Err(format!("Addition not supported for types: {a:?} + {b:?}")),
    }
}

/// Element-wise subtraction: A - B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
pub fn elementwise_sub(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x - y)),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num(*x as f64 - y)),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x - (*y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num(*x as f64 - *y as f64)),

        // Matrix-scalar cases (broadcasting)
        (Value::Matrix(m), Value::Num(s)) => {
            let data: Vec<f64> = m.data.iter().map(|x| x - s).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Matrix(m), Value::Int(s)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| x - scalar).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Num(s), Value::Matrix(m)) => {
            let data: Vec<f64> = m.data.iter().map(|x| s - x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Int(s), Value::Matrix(m)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| scalar - x).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }

        // Matrix-matrix case
        (Value::Matrix(m1), Value::Matrix(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for subtraction: {}x{} - {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| x - y)
                .collect();
            Ok(Value::Matrix(Matrix::new(data, m1.rows, m1.cols)?))
        }

        _ => Err(format!(
            "Subtraction not supported for types: {a:?} - {b:?}"
        )),
    }
}

/// Element-wise division: A ./ B
/// Supports matrix-matrix, matrix-scalar, and scalar-matrix operations
pub fn elementwise_div(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
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
                Ok(Value::Num(f64::INFINITY * (*x as f64).signum()))
            } else {
                Ok(Value::Num(*x as f64 / y))
            }
        }
        (Value::Num(x), Value::Int(y)) => {
            if *y == 0 {
                Ok(Value::Num(f64::INFINITY * x.signum()))
            } else {
                Ok(Value::Num(x / (*y as f64)))
            }
        }
        (Value::Int(x), Value::Int(y)) => {
            if *y == 0 {
                Ok(Value::Num(f64::INFINITY * (*x as f64).signum()))
            } else {
                Ok(Value::Num(*x as f64 / *y as f64))
            }
        }

        // Matrix-scalar cases (broadcasting)
        (Value::Matrix(m), Value::Num(s)) => {
            if *s == 0.0 {
                let data: Vec<f64> = m.data.iter().map(|x| f64::INFINITY * x.signum()).collect();
                Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
            } else {
                let data: Vec<f64> = m.data.iter().map(|x| x / s).collect();
                Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
            }
        }
        (Value::Matrix(m), Value::Int(s)) => {
            let scalar = *s as f64;
            if scalar == 0.0 {
                let data: Vec<f64> = m.data.iter().map(|x| f64::INFINITY * x.signum()).collect();
                Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
            } else {
                let data: Vec<f64> = m.data.iter().map(|x| x / scalar).collect();
                Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
            }
        }
        (Value::Num(s), Value::Matrix(m)) => {
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
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Int(s), Value::Matrix(m)) => {
            let scalar = *s as f64;
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
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }

        // Matrix-matrix case
        (Value::Matrix(m1), Value::Matrix(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise division: {}x{} ./ {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
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
            Ok(Value::Matrix(Matrix::new(data, m1.rows, m1.cols)?))
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
        // Scalar cases - same as elementwise
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x.powf(*y))),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num((*x as f64).powf(*y))),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x.powf(*y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num((*x as f64).powf(*y as f64))),

        // Matrix^scalar case - matrix exponentiation
        (Value::Matrix(m), Value::Num(s)) => {
            // Check if scalar is an integer for matrix power
            if s.fract() == 0.0 {
                let n = *s as i32;
                let result = matrix_power(m, n)?;
                Ok(Value::Matrix(result))
            } else {
                Err("Matrix power requires integer exponent".to_string())
            }
        }
        (Value::Matrix(m), Value::Int(s)) => {
            let result = matrix_power(m, *s)?;
            Ok(Value::Matrix(result))
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
        // Scalar-scalar case
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x.powf(*y))),
        (Value::Int(x), Value::Num(y)) => Ok(Value::Num((*x as f64).powf(*y))),
        (Value::Num(x), Value::Int(y)) => Ok(Value::Num(x.powf(*y as f64))),
        (Value::Int(x), Value::Int(y)) => Ok(Value::Num((*x as f64).powf(*y as f64))),

        // Matrix-scalar cases (broadcasting)
        (Value::Matrix(m), Value::Num(s)) => {
            let data: Vec<f64> = m.data.iter().map(|x| x.powf(*s)).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Matrix(m), Value::Int(s)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| x.powf(scalar)).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Num(s), Value::Matrix(m)) => {
            let data: Vec<f64> = m.data.iter().map(|x| s.powf(*x)).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }
        (Value::Int(s), Value::Matrix(m)) => {
            let scalar = *s as f64;
            let data: Vec<f64> = m.data.iter().map(|x| scalar.powf(*x)).collect();
            Ok(Value::Matrix(Matrix::new(data, m.rows, m.cols)?))
        }

        // Matrix-matrix case
        (Value::Matrix(m1), Value::Matrix(m2)) => {
            if m1.rows != m2.rows || m1.cols != m2.cols {
                return Err(format!(
                    "Matrix dimensions must agree for element-wise power: {}x{} .^ {}x{}",
                    m1.rows, m1.cols, m2.rows, m2.cols
                ));
            }
            let data: Vec<f64> = m1
                .data
                .iter()
                .zip(m2.data.iter())
                .map(|(x, y)| x.powf(*y))
                .collect();
            Ok(Value::Matrix(Matrix::new(data, m1.rows, m1.cols)?))
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

    #[test]
    fn test_elementwise_mul_scalars() {
        assert_eq!(
            elementwise_mul(&Value::Num(3.0), &Value::Num(4.0)).unwrap(),
            Value::Num(12.0)
        );
        assert_eq!(
            elementwise_mul(&Value::Int(3), &Value::Num(4.5)).unwrap(),
            Value::Num(13.5)
        );
    }

    #[test]
    fn test_elementwise_mul_matrix_scalar() {
        let matrix = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let result = elementwise_mul(&Value::Matrix(matrix), &Value::Num(2.0)).unwrap();

        if let Value::Matrix(m) = result {
            assert_eq!(m.data, vec![2.0, 4.0, 6.0, 8.0]);
            assert_eq!(m.rows, 2);
            assert_eq!(m.cols, 2);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn test_elementwise_mul_matrices() {
        let m1 = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let m2 = Matrix::new(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let result = elementwise_mul(&Value::Matrix(m1), &Value::Matrix(m2)).unwrap();

        if let Value::Matrix(m) = result {
            assert_eq!(m.data, vec![2.0, 6.0, 12.0, 20.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn test_elementwise_div_with_zero() {
        let result = elementwise_div(&Value::Num(5.0), &Value::Num(0.0)).unwrap();
        if let Value::Num(n) = result {
            assert!(n.is_infinite() && n.is_sign_positive());
        } else {
            panic!("Expected numeric result");
        }
    }

    #[test]
    fn test_elementwise_pow() {
        let matrix = Matrix::new(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        let result = elementwise_pow(&Value::Matrix(matrix), &Value::Num(2.0)).unwrap();

        if let Value::Matrix(m) = result {
            assert_eq!(m.data, vec![4.0, 9.0, 16.0, 25.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let m1 = Matrix::new(vec![1.0, 2.0], 1, 2).unwrap();
        let m2 = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        assert!(elementwise_mul(&Value::Matrix(m1), &Value::Matrix(m2)).is_err());
    }
}
