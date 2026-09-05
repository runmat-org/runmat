use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::{builtin_error_with_detail, CONJ_ERROR_INTERNAL, CONJ_ERROR_INVALID_INPUT};

pub(super) fn execute(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(real, imaginary) => Ok(Value::Complex(real, -imaginary)),
        Value::ComplexTensor(tensor) => conjugate_complex_tensor(tensor),
        Value::CharArray(chars) => conjugate_chars(chars),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        value @ (Value::LogicalArray(_) | Value::Bool(_)) => Ok(value),
        value @ (Value::Tensor(_) | Value::Num(_) | Value::Int(_)) => {
            let tensor = tensor::value_into_tensor_for("conj", value)
                .map_err(|error| builtin_error_with_detail(&CONJ_ERROR_INVALID_INPUT, error))?;
            Ok(tensor::tensor_into_value(tensor))
        }
        other => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            format!("unsupported input type {other:?}; expected numeric, logical, or char data"),
        )),
    }
}

fn conjugate_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imaginary)| (real, -imaginary))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imaginary)| (real, -imaginary))
                .collect(),
        ),
        ComplexStorage::Integer(storage) => ComplexStorage::Integer(
            IntegerComplexStorage::new(
                storage.real,
                conjugate_integer_imaginary_storage(storage.imag),
            )
            .map_err(|error| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, error))?,
        ),
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(|error| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, error))
}

pub(crate) fn conjugate_integer_imaginary_storage(storage: IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.into_iter().map(i8::saturating_neg).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.into_iter().map(i16::saturating_neg).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.into_iter().map(i32::saturating_neg).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.into_iter().map(i64::saturating_neg).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(vec![0; values.len()]),
        IntegerStorage::U16(values) => IntegerStorage::U16(vec![0; values.len()]),
        IntegerStorage::U32(values) => IntegerStorage::U32(vec![0; values.len()]),
        IntegerStorage::U64(values) => IntegerStorage::U64(vec![0; values.len()]),
    }
}

fn conjugate_chars(chars: CharArray) -> BuiltinResult<Value> {
    let data = chars
        .data
        .iter()
        .map(|&character| character as u32 as f64)
        .collect();
    let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|error| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, error))?;
    Ok(tensor::tensor_into_value(tensor))
}
