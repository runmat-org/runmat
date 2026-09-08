use runmat_value::{
    CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, NumericDType, NumericStorage,
    Tensor, Value,
};

use super::super::error::arrayfun_internal;
use super::contract::OutputContract;

pub(in crate::builtins::acceleration::gpu::arrayfun) fn empty_uniform(
    shape: &[usize],
    contract: OutputContract,
) -> crate::BuiltinResult<Value> {
    let len = shape.iter().copied().product();
    match contract {
        OutputContract::Logical => LogicalArray::new(vec![0; len], shape.to_vec())
            .map(Value::LogicalArray)
            .map_err(arrayfun_internal),
        OutputContract::Numeric(class) => {
            Tensor::from_numeric_storage(NumericStorage::zeros(class, len), shape.to_vec())
                .map(Value::Tensor)
                .map_err(arrayfun_internal)
        }
        OutputContract::Complex(class) => complex(class, len, shape),
        OutputContract::Character => characters(shape),
        OutputContract::Dynamic => Tensor::new(vec![0.0; len], shape.to_vec())
            .map(Value::Tensor)
            .map_err(arrayfun_internal),
    }
}

fn complex(class: NumericDType, len: usize, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = match class {
        NumericDType::F64 | NumericDType::F32 => {
            ComplexTensor::from_f64_values_with_dtype(vec![(0.0, 0.0); len], shape.to_vec(), class)
        }
        _ => {
            let real = integer_zeros(class, len)?;
            let imaginary = integer_zeros(class, len)?;
            IntegerComplexStorage::new(real, imaginary)
                .and_then(|storage| ComplexTensor::new_integer(storage, shape.to_vec()))
        }
    }
    .map_err(arrayfun_internal)?;
    Ok(Value::ComplexTensor(tensor))
}

fn integer_zeros(
    class: NumericDType,
    len: usize,
) -> crate::BuiltinResult<runmat_value::IntegerStorage> {
    NumericStorage::zeros(class, len)
        .into_integer_storage()
        .map_err(|_| arrayfun_internal("complex output class must be numeric"))
}

fn characters(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let normalized = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    if normalized.len() > 2 {
        return Err(arrayfun_internal(
            "character callback outputs require a two-dimensional result",
        ));
    }
    let rows = normalized.first().copied().unwrap_or(1);
    let columns = normalized.get(1).copied().unwrap_or(1);
    CharArray::new(Vec::new(), rows, columns)
        .map(Value::CharArray)
        .map_err(arrayfun_internal)
}
