use runmat_builtins::BSXFUN_ERROR_FUNCTION_ERROR;
use runmat_value::{
    CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, NumericDType, NumericStorage,
    Tensor, Value,
};

pub(super) fn value(
    shape: &[usize],
    contract: super::super::callback::OutputContract,
) -> crate::BuiltinResult<Value> {
    let len = shape.iter().copied().product();
    match contract {
        super::super::callback::OutputContract::Logical => {
            LogicalArray::new(vec![0; len], shape.to_vec())
                .map(Value::LogicalArray)
                .map_err(super::super::error::internal)
        }
        super::super::callback::OutputContract::Numeric(class) => {
            Tensor::from_numeric_storage(NumericStorage::zeros(class, len), shape.to_vec())
                .map(Value::Tensor)
                .map_err(super::super::error::internal)
        }
        super::super::callback::OutputContract::Complex(class) => complex(class, len, shape),
        super::super::callback::OutputContract::Character => characters(Vec::new(), shape),
        super::super::callback::OutputContract::Dynamic => {
            Tensor::new(vec![0.0; len], shape.to_vec())
                .map(Value::Tensor)
                .map_err(super::super::error::internal)
        }
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
    .map_err(super::super::error::internal)?;
    Ok(Value::ComplexTensor(tensor))
}

fn integer_zeros(
    class: NumericDType,
    len: usize,
) -> crate::BuiltinResult<runmat_value::IntegerStorage> {
    NumericStorage::zeros(class, len)
        .into_integer_storage()
        .map_err(|_| super::super::error::internal("complex output class must be numeric"))
}

pub(super) fn characters(characters: Vec<char>, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let normalized = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    if normalized.len() > 2 {
        return Err(super::super::error::detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            Some("character callback outputs must form a 2-D char array".to_string()),
        ));
    }
    let rows = normalized.first().copied().unwrap_or(1);
    let columns = normalized.get(1).copied().unwrap_or(1);
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| super::super::error::internal("character output size exceeds limits"))?;
    if expected != characters.len() {
        return Err(super::super::error::detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            Some("callback returned the wrong number of characters".to_string()),
        ));
    }
    let mut row_major = vec!['\0'; expected];
    for column in 0..columns {
        for row in 0..rows {
            row_major[row * columns + column] = characters[row + column * rows];
        }
    }
    CharArray::new(row_major, rows, columns)
        .map(Value::CharArray)
        .map_err(super::super::error::internal)
}
