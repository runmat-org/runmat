use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::BuiltinResult;

use super::{errors, BUILTIN_NAME};

pub(super) enum NumericArray {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NumericArray {
    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Real(tensor) => &tensor.shape,
            Self::Complex(tensor) => &tensor.shape,
        }
    }

    pub(super) fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_))
    }

    pub(super) fn uses_single(&self) -> bool {
        match self {
            Self::Real(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
            Self::Complex(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
        }
    }

    pub(super) fn into_f32_components(self) -> BuiltinResult<Vec<(f32, f32)>> {
        match self {
            Self::Real(tensor) => {
                tensor
                    .into_numeric_storage()
                    .map_err(errors::internal)
                    .map(|storage| {
                        storage
                            .materialize_f32()
                            .into_iter()
                            .map(|value| (value, 0.0))
                            .collect()
                    })
            }
            Self::Complex(tensor) => match tensor.into_complex_storage() {
                ComplexStorage::F32(values) => Ok(values.into_iter().collect()),
                ComplexStorage::F64(values) => Ok(values
                    .into_iter()
                    .map(|(real, imaginary)| (real as f32, imaginary as f32))
                    .collect()),
                ComplexStorage::Integer(_) => Err(errors::invalid_input(
                    "complex fixed-width integer input is not supported",
                )),
            },
        }
    }

    pub(super) fn into_f64_components(self) -> BuiltinResult<Vec<(f64, f64)>> {
        match self {
            Self::Real(tensor) => {
                let storage = tensor.into_numeric_storage().map_err(errors::internal)?;
                let values = match storage {
                    NumericStorage::F64(values) => values,
                    NumericStorage::F32(_) => {
                        return Err(errors::internal(
                            "single storage entered the double pow2 domain",
                        ))
                    }
                    storage => integer_values_as_f64(storage)?,
                };
                Ok(values.into_iter().map(|value| (value, 0.0)).collect())
            }
            Self::Complex(tensor) => match tensor.into_complex_storage() {
                ComplexStorage::F64(values) => Ok(values.into_iter().collect()),
                ComplexStorage::F32(_) => Err(errors::internal(
                    "complex single storage entered the double pow2 domain",
                )),
                ComplexStorage::Integer(_) => Err(errors::invalid_input(
                    "complex fixed-width integer input is not supported",
                )),
            },
        }
    }
}

pub(super) fn into_numeric_array(value: Value) -> BuiltinResult<NumericArray> {
    match value {
        Value::Complex(real, imaginary) => ComplexTensor::new(vec![(real, imaginary)], vec![1, 1])
            .map(NumericArray::Complex)
            .map_err(errors::internal),
        Value::ComplexTensor(tensor) => Ok(NumericArray::Complex(tensor)),
        Value::CharArray(chars) => char_tensor(chars).map(NumericArray::Real),
        Value::String(_) | Value::StringArray(_) => {
            Err(errors::invalid_input("expected numeric input, got string"))
        }
        Value::GpuTensor(_) => Err(errors::internal(
            "resident input reached host numeric conversion",
        )),
        other => crate::builtins::common::tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map(NumericArray::Real)
            .map_err(errors::invalid_input),
    }
}

pub(super) fn scalar_component(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Num(number) => Some((*number, 0.0)),
        Value::Int(integer) => Some((integer.to_f64(), 0.0)),
        Value::Bool(logical) => Some((f64::from(*logical), 0.0)),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Some((f64::from(logical.data[0] != 0), 0.0))
        }
        Value::CharArray(chars) if chars.data.len() == 1 => chars
            .data
            .first()
            .map(|character| (*character as u32 as f64, 0.0)),
        Value::Complex(real, imaginary) => Some((*real, *imaginary)),
        _ => None,
    }
}

pub(super) fn integer_values_as_f64(storage: NumericStorage) -> BuiltinResult<Vec<f64>> {
    storage
        .into_integer_storage()
        .map(|values| values.to_f64_vec())
        .map_err(|_| errors::internal("floating storage reached an integer conversion boundary"))
}

fn char_tensor(chars: CharArray) -> BuiltinResult<Tensor> {
    let values = chars
        .data
        .iter()
        .map(|character| *character as u32 as f64)
        .collect();
    Tensor::new(values, vec![chars.rows, chars.cols]).map_err(errors::internal)
}
