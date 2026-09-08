use runmat_value::{IntValue, IntegerStorage, Value};

use super::classify::{self, ClassifiedValue};
use super::{finish, push, UniformScalarError};

pub(crate) enum UniformScalarCollector {
    Pending,
    Logical(Vec<u8>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    Integer {
        prototype: IntegerStorage,
        values: Vec<IntValue>,
    },
    ComplexF64(Vec<(f64, f64)>),
    ComplexF32(Vec<(f32, f32)>),
    IntegerComplex {
        prototype: IntegerStorage,
        real: Vec<IntValue>,
        imaginary: Vec<IntValue>,
    },
    Char(Vec<char>),
}

impl UniformScalarCollector {
    pub(crate) fn new() -> Self {
        Self::Pending
    }

    pub(crate) fn push(&mut self, value: &Value) -> Result<(), UniformScalarError> {
        let classified = classify::value(value)?;
        match self {
            Self::Pending => self.start(classified),
            Self::Logical(values) => {
                match push::logical(values, classified)? {
                    Some(push::LogicalPromotion::F64(values)) => *self = Self::F64(values),
                    Some(push::LogicalPromotion::ComplexF64(values)) => {
                        *self = Self::ComplexF64(values)
                    }
                    None => {}
                }
                Ok(())
            }
            Self::F64(values) => {
                if let Some(complex) = push::f64(values, classified)? {
                    *self = Self::ComplexF64(complex);
                }
                Ok(())
            }
            Self::F32(values) => {
                if let Some(complex) = push::f32(values, classified)? {
                    *self = Self::ComplexF32(complex);
                }
                Ok(())
            }
            Self::Integer { prototype, values } => push::integer(prototype, values, classified),
            Self::ComplexF64(values) => push::complex_f64(values, classified),
            Self::ComplexF32(values) => push::complex_f32(values, classified),
            Self::IntegerComplex {
                prototype,
                real,
                imaginary,
            } => push::integer_complex(prototype, real, imaginary, classified),
            Self::Char(values) => push::character(values, classified),
        }
    }

    pub(crate) fn finish(self, shape: &[usize]) -> Result<Value, UniformScalarError> {
        finish::value(self, shape)
    }

    fn start(&mut self, value: ClassifiedValue) -> Result<(), UniformScalarError> {
        *self = match value {
            ClassifiedValue::Logical(value) => Self::Logical(vec![value as u8]),
            ClassifiedValue::F64(value) => Self::F64(vec![value]),
            ClassifiedValue::F32(value) => Self::F32(vec![value]),
            ClassifiedValue::Integer(value) => Self::Integer {
                prototype: IntegerStorage::from_scalar(value.clone()),
                values: vec![value],
            },
            ClassifiedValue::ComplexF64(value) => Self::ComplexF64(vec![value]),
            ClassifiedValue::ComplexF32(value) => Self::ComplexF32(vec![value]),
            ClassifiedValue::IntegerComplex(real, imaginary) => Self::IntegerComplex {
                prototype: IntegerStorage::from_scalar(real.clone()),
                real: vec![real],
                imaginary: vec![imaginary],
            },
            ClassifiedValue::Char(value) => Self::Char(vec![value]),
        };
        Ok(())
    }
}
