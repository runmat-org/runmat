//! Comparison operations for language-compatible logic
//!
//! Implements comparison operators returning logical matrices/values.

use runmat_builtins::Tensor;
#[cfg(test)]
use runmat_builtins::Value;

/// Element-wise greater than comparison
pub fn matrix_gt(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} > {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Element-wise greater than or equal comparison
pub fn matrix_ge(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} >= {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x >= y { 1.0 } else { 0.0 })
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Element-wise less than comparison
pub fn matrix_lt(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} < {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Element-wise less than or equal comparison
pub fn matrix_le(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} <= {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x <= y { 1.0 } else { 0.0 })
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Element-wise equality comparison
pub fn matrix_eq(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} == {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| {
            if (x - y).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Element-wise inequality comparison
pub fn matrix_ne(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} != {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| {
            if (x - y).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

// Built-in comparison functions
#[cfg(test)]
#[allow(dead_code)]
fn gt_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a > b { 1.0 } else { 0.0 })
}

#[cfg(test)]
#[allow(dead_code)]
fn ge_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a >= b { 1.0 } else { 0.0 })
}

#[cfg(test)]
#[allow(dead_code)]
fn lt_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a < b { 1.0 } else { 0.0 })
}

#[cfg(test)]
#[allow(dead_code)]
fn le_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a <= b { 1.0 } else { 0.0 })
}

#[cfg(test)]
#[allow(dead_code)]
fn eq_builtin(a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        // Handle identity semantics
        (Value::HandleObject(ha), Value::HandleObject(hb)) => {
            let pa = unsafe { ha.target.as_raw() } as usize;
            let pb = unsafe { hb.target.as_raw() } as usize;
            Ok(Value::Num(if pa == pb { 1.0 } else { 0.0 }))
        }
        (Value::HandleObject(ha), other) => {
            let pb = match other {
                Value::HandleObject(hb) => (unsafe { hb.target.as_raw() }) as usize,
                _ => 0usize,
            };
            let pa = (unsafe { ha.target.as_raw() }) as usize;
            Ok(Value::Num(if pa == pb && pb != 0 { 1.0 } else { 0.0 }))
        }
        (other, Value::HandleObject(hb)) => {
            let pa = match other {
                Value::HandleObject(ha) => (unsafe { ha.target.as_raw() }) as usize,
                _ => 0usize,
            };
            let pb = (unsafe { hb.target.as_raw() }) as usize;
            Ok(Value::Num(if pa == pb && pa != 0 { 1.0 } else { 0.0 }))
        }
        // Complex equality: element-wise on scalars; uses exact float compare for now
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            Ok(Value::Num(((ar == br) && (ai == bi)) as i32 as f64))
        }
        (Value::Complex(ar, ai), Value::Num(bn)) => {
            Ok(Value::Num(((ai == 0.0) && (ar == bn)) as i32 as f64))
        }
        (Value::Num(an), Value::Complex(br, bi)) => {
            Ok(Value::Num(((bi == 0.0) && (br == an)) as i32 as f64))
        }
        (Value::CharArray(ca), Value::CharArray(cb)) => {
            if ca.rows != cb.rows || ca.cols != cb.cols {
                return Err("shape mismatch for char array comparison".to_string());
            }
            let out: Vec<f64> = ca
                .data
                .iter()
                .zip(cb.data.iter())
                .map(|(x, y)| if x == y { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, vec![ca.rows, cb.cols]).map_err(|e| format!("eq: {e}"))?,
            ))
        }
        (Value::CharArray(ca), Value::String(s)) => {
            let ss: String = ca.data.iter().collect();
            Ok(Value::Num(if ss == s { 1.0 } else { 0.0 }))
        }
        (Value::String(s), Value::CharArray(ca)) => {
            let ss: String = ca.data.iter().collect();
            Ok(Value::Num(if s == ss { 1.0 } else { 0.0 }))
        }
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err("shape mismatch for string array comparison".to_string());
            }
            let out: Vec<f64> = sa
                .data
                .iter()
                .zip(sb.data.iter())
                .map(|(x, y)| if x == y { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, sa.shape).map_err(|e| format!("eq: {e}"))?,
            ))
        }
        (Value::StringArray(sa), Value::String(s)) => {
            let out: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if x == &s { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, sa.shape).map_err(|e| format!("eq: {e}"))?,
            ))
        }
        (Value::String(s), Value::StringArray(sa)) => {
            let out: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if &s == x { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, sa.shape).map_err(|e| format!("eq: {e}"))?,
            ))
        }
        (Value::String(a), Value::String(b)) => Ok(Value::Num(if a == b { 1.0 } else { 0.0 })),
        (Value::Num(a), Value::Num(b)) => Ok(Value::Num(if (a - b).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        })),
        (Value::Int(a), Value::Int(b)) => {
            Ok(Value::Num(if a.to_i64() == b.to_i64() { 1.0 } else { 0.0 }))
        }
        (Value::Int(a), Value::Num(b)) => {
            Ok(Value::Num(if (a.to_f64() - b).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }))
        }
        (Value::Num(a), Value::Int(b)) => {
            Ok(Value::Num(if (a - b.to_f64()).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }))
        }
        (a, b) => {
            let aa: f64 = (&a).try_into()?;
            let bb: f64 = (&b).try_into()?;
            Ok(Value::Num(if (aa - bb).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }))
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn ne_builtin(a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        // Handle identity semantics
        (Value::HandleObject(ha), Value::HandleObject(hb)) => {
            let pa = unsafe { ha.target.as_raw() } as usize;
            let pb = unsafe { hb.target.as_raw() } as usize;
            Ok(Value::Num(if pa != pb { 1.0 } else { 0.0 }))
        }
        (Value::HandleObject(ha), other) => {
            let pb = match other {
                Value::HandleObject(hb) => (unsafe { hb.target.as_raw() }) as usize,
                _ => 0usize,
            };
            let pa = (unsafe { ha.target.as_raw() }) as usize;
            Ok(Value::Num(if pa != pb || pb == 0 { 1.0 } else { 0.0 }))
        }
        (other, Value::HandleObject(hb)) => {
            let pa = match other {
                Value::HandleObject(ha) => (unsafe { ha.target.as_raw() }) as usize,
                _ => 0usize,
            };
            let pb = (unsafe { hb.target.as_raw() }) as usize;
            Ok(Value::Num(if pa != pb || pa == 0 { 1.0 } else { 0.0 }))
        }
        // Complex inequality
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            Ok(Value::Num(((ar != br) || (ai != bi)) as i32 as f64))
        }
        (Value::Complex(ar, ai), Value::Num(bn)) => {
            Ok(Value::Num(((ai != 0.0) || (ar != bn)) as i32 as f64))
        }
        (Value::Num(an), Value::Complex(br, bi)) => {
            Ok(Value::Num(((bi != 0.0) || (br != an)) as i32 as f64))
        }
        (Value::CharArray(ca), Value::CharArray(cb)) => {
            if ca.rows != cb.rows || ca.cols != cb.cols {
                return Err("shape mismatch for char array comparison".to_string());
            }
            let out: Vec<f64> = ca
                .data
                .iter()
                .zip(cb.data.iter())
                .map(|(x, y)| if x != y { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, vec![ca.rows, cb.cols]).map_err(|e| format!("ne: {e}"))?,
            ))
        }
        (Value::CharArray(ca), Value::String(s)) => {
            let ss: String = ca.data.iter().collect();
            Ok(Value::Num(if ss != s { 1.0 } else { 0.0 }))
        }
        (Value::String(s), Value::CharArray(ca)) => {
            let ss: String = ca.data.iter().collect();
            Ok(Value::Num(if s != ss { 1.0 } else { 0.0 }))
        }
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err("shape mismatch for string array comparison".to_string());
            }
            let out: Vec<f64> = sa
                .data
                .iter()
                .zip(sb.data.iter())
                .map(|(x, y)| if x != y { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, sa.shape).map_err(|e| format!("ne: {e}"))?,
            ))
        }
        (Value::StringArray(sa), Value::String(s)) => {
            let out: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if x != &s { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, sa.shape).map_err(|e| format!("ne: {e}"))?,
            ))
        }
        (Value::String(s), Value::StringArray(sa)) => {
            let out: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if &s != x { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                Tensor::new(out, sa.shape).map_err(|e| format!("ne: {e}"))?,
            ))
        }
        (Value::String(a), Value::String(b)) => Ok(Value::Num(if a != b { 1.0 } else { 0.0 })),
        (Value::Num(a), Value::Num(b)) => Ok(Value::Num(if (a - b).abs() >= f64::EPSILON {
            1.0
        } else {
            0.0
        })),
        (Value::Int(a), Value::Int(b)) => {
            Ok(Value::Num(if a.to_i64() != b.to_i64() { 1.0 } else { 0.0 }))
        }
        (Value::Int(a), Value::Num(b)) => {
            Ok(Value::Num(if (a.to_f64() - b).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }))
        }
        (Value::Num(a), Value::Int(b)) => {
            Ok(Value::Num(if (a - b.to_f64()).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }))
        }
        (a, b) => {
            let aa: f64 = (&a).try_into()?;
            let bb: f64 = (&b).try_into()?;
            Ok(Value::Num(if (aa - bb).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }))
        }
    }
}
