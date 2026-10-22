//! Exceptional borrowed triangular stage of a scalar-multiplied operator.
//! The ordinary factor solve remains on faer's in-place path.
use super::*;
use crate::builtins::math::linalg::ops::triangular_operator::TriangularScalar;

pub(super) fn borrowed_coefficient<T: DenseScalar>(
    packed: &NativeDenseFactorMatrix,
    index: usize,
) -> Option<T> {
    if let Some(values) = T::elements(packed) {
        return Some(values[index]);
    }
    if !T::COMPLEX {
        return None;
    }
    match packed {
        NativeDenseFactorMatrix::Real(t) if t.numeric_dtype() == T::DTYPE => {
            if let Some(values) = t.as_f32_slice() {
                T::from_parts(f64::from(values[index]), 0.0)
            } else {
                T::from_parts(t.as_f64_slice()?[index], 0.0)
            }
        }
        _ => None,
    }
}

pub(super) struct Stage<'a> {
    pub packed: &'a NativeDenseFactorMatrix,
    pub order: usize,
    pub upper: bool,
    pub adjoint: bool,
    pub unit_diagonal: bool,
    pub scale: (f64, f64),
    pub singular: bool,
}

pub(super) fn solve<T: DenseScalar + TriangularScalar>(
    stage: Stage<'_>,
    mut rhs: MatMut<'_, T::Element>,
) -> Result<(), DenseFactorError> {
    let Stage {
        packed,
        order,
        upper,
        adjoint,
        unit_diagonal,
        scale,
        singular,
    } = stage;
    let (leading, _) = packed.dimensions();
    if order != 0 && borrowed_coefficient::<T>(packed, 0).is_none() {
        return Err(DenseFactorError::UnsupportedClass);
    }
    let forward = upper == adjoint;
    for column in 0..rhs.ncols() {
        for step in 0..order {
            let row = if forward { step } else { order - 1 - step };
            let mut value = T::from_element(rhs[(row, column)]);
            for prior in 0..step {
                let solved = if forward { prior } else { order - 1 - prior };
                let coefficient = if adjoint {
                    borrowed_coefficient::<T>(packed, row * leading + solved)
                        .ok_or(DenseFactorError::UnsupportedClass)?
                        .conjugated()
                } else {
                    borrowed_coefficient::<T>(packed, solved * leading + row)
                        .ok_or(DenseFactorError::UnsupportedClass)?
                };
                let coefficient = coefficient
                    .scaled_complex(scale)
                    .ok_or(DenseFactorError::NonFinite)?;
                T::subtract_product(
                    &mut value,
                    coefficient,
                    T::from_element(rhs[(solved, column)]),
                );
                if !value.is_finite() && !singular {
                    return Err(DenseFactorError::NonFinite);
                }
            }
            let diagonal = if unit_diagonal {
                <T as TriangularScalar>::one()
            } else {
                let coefficient = borrowed_coefficient::<T>(packed, row * leading + row)
                    .ok_or(DenseFactorError::UnsupportedClass)?;
                let oriented = if adjoint {
                    coefficient.conjugated()
                } else {
                    coefficient
                };
                oriented
                    .scaled_complex(scale)
                    .ok_or(DenseFactorError::NonFinite)?
            };
            let solved = if singular {
                value.divide_ieee(diagonal)
            } else {
                TriangularScalar::divide(value, diagonal).ok_or(DenseFactorError::NonFinite)?
            };
            if !solved.is_finite() && !singular {
                return Err(DenseFactorError::NonFinite);
            }
            rhs[(row, column)] = solved.to_factor();
        }
    }
    Ok(())
}
