//! Borrowed dense member of a shifted pencil; no assembled dense A-sigma*B.
use super::super::{ManagedSparseLuError, ManagedSparseScalar};
use runmat_value::{ComplexElement, Value};
enum Values<'a> {
    Real32(&'a [f32]),
    Real64(&'a [f64]),
    Complex32(&'a [ComplexElement<f32>]),
    Complex64(&'a [ComplexElement<f64>]),
}
pub(in crate::builtins::math::linalg::ops::sparse_factor::managed_lu) struct Dense<'a> {
    values: Values<'a>,
    pub order: usize,
}
impl<'a> Dense<'a> {
    pub fn borrow(value: &'a Value) -> Result<Self, ManagedSparseLuError> {
        let (shape, values) = match value {
            Value::Tensor(t) => {
                let values = if let Some(values) = t.as_f32_slice() {
                    Values::Real32(values)
                } else if let Some(values) = t.as_f64_slice() {
                    Values::Real64(values)
                } else {
                    return Err(ManagedSparseLuError::UnsupportedClass);
                };
                (t.shape(), values)
            }
            Value::ComplexTensor(t) => {
                let values = if let Some(values) = t.as_f32_slice() {
                    Values::Complex32(values)
                } else if let Some(values) = t.as_f64_slice() {
                    Values::Complex64(values)
                } else {
                    return Err(ManagedSparseLuError::UnsupportedClass);
                };
                (t.shape(), values)
            }
            _ => return Err(ManagedSparseLuError::UnsupportedClass),
        };
        let n = shape.first().copied().unwrap_or(0);
        if shape.get(1) != Some(&n) || shape.iter().skip(2).any(|&size| size != 1) {
            return Err(ManagedSparseLuError::DimensionMismatch);
        }
        Ok(Self { values, order: n })
    }
    pub fn matches<T: ManagedSparseScalar>(&self) -> bool {
        !matches!(self.values, Values::Complex32(_) | Values::Complex64(_))
            || T::from_parts(0.0, 1.0).is_some()
    }
    pub fn read<T: ManagedSparseScalar>(
        &self,
        row: usize,
        column: usize,
    ) -> Result<T, ManagedSparseLuError> {
        let index = column * self.order + row;
        let (real, imag) = match self.values {
            Values::Real32(v) => (v[index] as f64, 0.0),
            Values::Real64(v) => (v[index], 0.0),
            Values::Complex32(v) => (v[index].0 as f64, v[index].1 as f64),
            Values::Complex64(v) => (v[index].0, v[index].1),
        };
        T::from_parts(real, imag).ok_or(ManagedSparseLuError::NonFiniteInput)
    }
}
