//! Borrowed native input policy for the one existing LU backing.
use super::{DenseFactorError, FaerFactorScalar};
use num_complex::Complex64;
use runmat_value::{ComplexElement, Value};
pub(super) enum Policy<T> {
    Exact,
    Converted(fn(Complex64) -> Option<T>),
}
pub(super) enum Floating<'a> {
    Real32(&'a [f32]),
    Real64(&'a [f64]),
    Complex32(&'a [ComplexElement<f32>]),
    Complex64(&'a [ComplexElement<f64>]),
}
pub(super) enum Input<'a, T> {
    Exact(&'a [T]),
    Converted {
        source: Floating<'a>,
        convert: fn(Complex64) -> Option<T>,
    },
}
impl<'a, T: FaerFactorScalar> Input<'a, T> {
    pub fn borrow(value: &'a Value, policy: Policy<T>) -> Result<Self, DenseFactorError> {
        let Policy::Converted(convert) = policy else {
            return T::source(value)
                .map(Self::Exact)
                .ok_or(DenseFactorError::UnsupportedClass);
        };
        let source = match value {
            Value::Tensor(t) => {
                if let Some(v) = t.as_f32_slice() {
                    Floating::Real32(v)
                } else if let Some(v) = t.as_f64_slice() {
                    Floating::Real64(v)
                } else {
                    return Err(DenseFactorError::UnsupportedClass);
                }
            }
            Value::ComplexTensor(t) => {
                if let Some(v) = t.as_f32_slice() {
                    Floating::Complex32(v)
                } else if let Some(v) = t.as_f64_slice() {
                    Floating::Complex64(v)
                } else {
                    return Err(DenseFactorError::UnsupportedClass);
                }
            }
            _ => return Err(DenseFactorError::UnsupportedClass),
        };
        Ok(Self::Converted { source, convert })
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Exact(v) => v.len(),
            Self::Converted { source, .. } => match source {
                Floating::Real32(v) => v.len(),
                Floating::Real64(v) => v.len(),
                Floating::Complex32(v) => v.len(),
                Floating::Complex64(v) => v.len(),
            },
        }
    }
    pub fn read(&self, index: usize) -> Result<T, DenseFactorError> {
        match self {
            Self::Exact(v) => Ok(v[index]),
            Self::Converted { source, convert } => {
                let value = match source {
                    Floating::Real32(v) => Complex64::new(v[index] as f64, 0.0),
                    Floating::Real64(v) => Complex64::new(v[index], 0.0),
                    Floating::Complex32(v) => Complex64::new(v[index].0 as f64, v[index].1 as f64),
                    Floating::Complex64(v) => Complex64::new(v[index].0, v[index].1),
                };
                convert(value).ok_or(DenseFactorError::NonFinite)
            }
        }
    }
}
