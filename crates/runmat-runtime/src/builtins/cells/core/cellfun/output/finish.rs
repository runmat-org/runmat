use crate::BuiltinResult;
use runmat_value::{CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, Tensor, Value};

use super::super::error;
use super::collector::UniformCollector;

impl UniformCollector {
    pub(in crate::builtins::cells::core::cellfun) fn finish(
        self,
        shape: &[usize],
    ) -> BuiltinResult<Value> {
        match self {
            Self::Pending => tensor(vec![0.0; shape.iter().product()], shape),
            Self::Logical(values) => LogicalArray::new(values, shape.to_vec())
                .map(Value::LogicalArray)
                .map_err(|reason| error::internal(format!("cellfun: {reason}"))),
            Self::F64(values) => tensor(values, shape),
            Self::F32(values) => Tensor::from_f32(values, shape.to_vec())
                .map(Value::Tensor)
                .map_err(|reason| error::internal(format!("cellfun: {reason}"))),
            Self::Integer { prototype, values } => {
                let storage = prototype
                    .from_same_class_values(values)
                    .map_err(|reason| error::internal(format!("cellfun: {reason}")))?;
                Tensor::new_integer(storage, shape.to_vec())
                    .map(Value::Tensor)
                    .map_err(|reason| error::internal(format!("cellfun: {reason}")))
            }
            Self::ComplexF64(values) => ComplexTensor::new(values, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|reason| error::internal(format!("cellfun: {reason}"))),
            Self::ComplexF32(values) => ComplexTensor::from_f32(values, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|reason| error::internal(format!("cellfun: {reason}"))),
            Self::IntegerComplex {
                prototype,
                real,
                imaginary,
            } => {
                let real = prototype
                    .from_same_class_values(real)
                    .map_err(|reason| error::internal(format!("cellfun: {reason}")))?;
                let imaginary = prototype
                    .from_same_class_values(imaginary)
                    .map_err(|reason| error::internal(format!("cellfun: {reason}")))?;
                let storage = IntegerComplexStorage::new(real, imaginary)
                    .map_err(|reason| error::internal(format!("cellfun: {reason}")))?;
                ComplexTensor::new_integer(storage, shape.to_vec())
                    .map(Value::ComplexTensor)
                    .map_err(|reason| error::internal(format!("cellfun: {reason}")))
            }
            Self::Char(values) => character(values, shape),
        }
    }
}

fn tensor(values: Vec<f64>, shape: &[usize]) -> BuiltinResult<Value> {
    Tensor::new(values, shape.to_vec())
        .map(Value::Tensor)
        .map_err(|reason| error::internal(format!("cellfun: {reason}")))
}

fn character(values: Vec<char>, shape: &[usize]) -> BuiltinResult<Value> {
    let normalized = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    if normalized.len() > 2 {
        return Err(error::uniform(
            "cellfun: character outputs with UniformOutput=true must be 2-D",
        ));
    }
    let rows = normalized.first().copied().unwrap_or(1);
    let columns = normalized.get(1).copied().unwrap_or(1);
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| error::internal("cellfun: character output size exceeds platform limits"))?;
    if expected != values.len() {
        return Err(error::uniform(
            "cellfun: callback returned the wrong number of characters",
        ));
    }
    let mut row_major = vec!['\0'; expected];
    for column in 0..columns {
        for row in 0..rows {
            row_major[row * columns + column] = values[row + column * rows];
        }
    }
    CharArray::new(row_major, rows, columns)
        .map(Value::CharArray)
        .map_err(|reason| error::internal(format!("cellfun: {reason}")))
}
