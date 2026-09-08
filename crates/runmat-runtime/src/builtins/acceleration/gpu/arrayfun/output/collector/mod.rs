mod push;

use crate::BuiltinResult;
use runmat_value::{IntValue, IntegerStorage, Value};

use super::classify::{classify_value, ClassifiedValue};

pub(in crate::builtins::acceleration::gpu::arrayfun) enum UniformCollector {
    Pending,
    F64(Vec<f64>),
    F32(Vec<f32>),
    Integer {
        prototype: IntegerStorage,
        values: Vec<IntValue>,
    },
    Logical(Vec<u8>),
    ComplexF64(Vec<(f64, f64)>),
    ComplexF32(Vec<(f32, f32)>),
    IntegerComplex {
        real_prototype: IntegerStorage,
        imag_prototype: IntegerStorage,
        real_values: Vec<IntValue>,
        imag_values: Vec<IntValue>,
    },
    Char(Vec<char>),
}

impl UniformCollector {
    pub(in crate::builtins::acceleration::gpu::arrayfun) fn push(
        &mut self,
        value: &Value,
    ) -> BuiltinResult<()> {
        match self {
            Self::Pending => self.start(classify_value(value)?),
            Self::Logical(values) => push::logical(values, classify_value(value)?),
            Self::F64(values) => {
                if let Some(complex) = push::f64(values, classify_value(value)?)? {
                    *self = Self::ComplexF64(complex);
                }
                Ok(())
            }
            Self::F32(values) => {
                if let Some(complex) = push::f32(values, classify_value(value)?)? {
                    *self = Self::ComplexF32(complex);
                }
                Ok(())
            }
            Self::Integer { prototype, values } => {
                push::integer(prototype, values, classify_value(value)?)
            }
            Self::ComplexF64(values) => push::complex_f64(values, classify_value(value)?),
            Self::ComplexF32(values) => push::complex_f32(values, classify_value(value)?),
            Self::IntegerComplex {
                real_prototype,
                imag_prototype,
                real_values,
                imag_values,
            } => push::integer_complex(
                real_prototype,
                imag_prototype,
                real_values,
                imag_values,
                classify_value(value)?,
            ),
            Self::Char(values) => push::character(values, classify_value(value)?),
        }
    }

    fn start(&mut self, value: ClassifiedValue) -> BuiltinResult<()> {
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
            ClassifiedValue::IntegerComplex(real, imag) => Self::IntegerComplex {
                real_prototype: IntegerStorage::from_scalar(real.clone()),
                imag_prototype: IntegerStorage::from_scalar(imag.clone()),
                real_values: vec![real],
                imag_values: vec![imag],
            },
            ClassifiedValue::Char(value) => Self::Char(vec![value]),
        };
        Ok(())
    }
}
