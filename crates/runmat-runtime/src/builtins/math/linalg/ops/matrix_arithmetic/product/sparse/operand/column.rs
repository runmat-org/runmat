//! Constant-time scalar iteration over actual borrowed column storage.
use super::*;
enum Values<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    C32(&'a [ComplexElement<f32>]),
    C64(&'a [ComplexElement<f64>]),
    Logical(&'a [u8]),
    Unit(usize),
}
pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::product::sparse) struct Column<'a> {
    values: Values<'a>,
    rows: Option<&'a [usize]>,
    next: usize,
}
impl<'a> Column<'a> {
    pub(super) fn pattern(rows: &'a [usize]) -> Self {
        Self {
            values: Values::Unit(rows.len()),
            rows: Some(rows),
            next: 0,
        }
    }
    pub(super) fn logical(values: &'a [u8]) -> Self {
        Self {
            values: Values::Logical(values),
            rows: None,
            next: 0,
        }
    }
    pub(super) fn floating(value: LinearOperator<'a>, index: usize) -> Self {
        macro_rules! column {
            ($view:expr,$kind:ident) => {
                match $view {
                    TypedLinearOperator::Dense(view) => Self {
                        values: Values::$kind(
                            &view.as_slice()[index * view.rows()..(index + 1) * view.rows()],
                        ),
                        rows: None,
                        next: 0,
                    },
                    TypedLinearOperator::Sparse(view) => {
                        let (rows, values) = view.column(index);
                        Self {
                            values: Values::$kind(values),
                            rows: Some(rows),
                            next: 0,
                        }
                    }
                    TypedLinearOperator::Scalar(_) => unreachable!("scalar matrix path uses times"),
                }
            };
        }
        match value {
            LinearOperator::RealF32(view) => column!(view, F32),
            LinearOperator::RealF64(view) => column!(view, F64),
            LinearOperator::ComplexF32(view) => column!(view, C32),
            LinearOperator::ComplexF64(view) => column!(view, C64),
        }
    }
}
impl Iterator for Column<'_> {
    type Item = (usize, Complex64);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next;
        let value = match &self.values {
            Values::F32(values) => Complex64::new(f64::from(*values.get(index)?), 0.),
            Values::F64(values) => Complex64::new(*values.get(index)?, 0.),
            Values::C32(values) => {
                let v = values.get(index)?;
                Complex64::new(f64::from(v.0), f64::from(v.1))
            }
            Values::C64(values) => {
                let v = values.get(index)?;
                Complex64::new(v.0, v.1)
            }
            Values::Logical(values) => Complex64::new(f64::from(*values.get(index)? != 0), 0.),
            Values::Unit(count) => {
                if index >= *count {
                    return None;
                }
                Complex64::new(1., 0.)
            }
        };
        self.next += 1;
        Some((self.rows.map_or(index, |rows| rows[index]), value))
    }
}
