//! The single class-join rule for ordinary and admitted collectors.
use super::{classify::ClassifiedValue, UniformScalarError};
use runmat_value::NumericDType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Class {
    Logical,
    F64,
    F32,
    Integer(NumericDType),
    ComplexF64,
    ComplexF32,
    IntegerComplex(NumericDType),
    Char,
}
impl Class {
    pub(super) fn of(value: &ClassifiedValue) -> Result<Self, UniformScalarError> {
        Ok(match value {
            ClassifiedValue::Logical(_) => Self::Logical,
            ClassifiedValue::F64(_) => Self::F64,
            ClassifiedValue::F32(_) => Self::F32,
            ClassifiedValue::Integer(value) => Self::Integer(value.numeric_dtype()),
            ClassifiedValue::ComplexF64(_) => Self::ComplexF64,
            ClassifiedValue::ComplexF32(_) => Self::ComplexF32,
            ClassifiedValue::IntegerComplex(real, imaginary) => {
                if real.numeric_dtype() != imaginary.numeric_dtype() {
                    return Err(UniformScalarError::HETEROGENEOUS);
                }
                Self::IntegerComplex(real.numeric_dtype())
            }
            ClassifiedValue::Char(_) => Self::Char,
        })
    }
    pub(super) fn join(self, next: Self) -> Result<Self, UniformScalarError> {
        use Class::*;
        match (self, next) {
            (left, right) if left == right => Ok(left),
            (Logical, F64) | (F64, Logical) => Ok(F64),
            (Logical | F64, ComplexF64) | (ComplexF64, Logical | F64) => Ok(ComplexF64),
            (F32, ComplexF32) | (ComplexF32, F32) => Ok(ComplexF32),
            _ => Err(UniformScalarError::HETEROGENEOUS),
        }
    }
}
