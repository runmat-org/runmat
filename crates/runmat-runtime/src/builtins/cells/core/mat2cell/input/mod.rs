mod numeric;
mod text;

use runmat_value::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

use super::error::{
    mat2cell_error_with_message, MAT2CELL_ERROR_INTERNAL, MAT2CELL_ERROR_INVALID_INPUT,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

enum Storage {
    Numeric(Tensor),
    Complex(ComplexTensor),
    Logical(LogicalArray),
    String(StringArray),
    Character(CharArray),
}

pub(super) struct Input {
    storage: Storage,
    shape: Vec<usize>,
    dimensions: Vec<usize>,
}

impl Input {
    pub(super) fn new(value: Value) -> BuiltinResult<Self> {
        let storage = match value {
            Value::Tensor(value) => Storage::Numeric(value),
            Value::ComplexTensor(value) => Storage::Complex(value),
            Value::LogicalArray(value) => Storage::Logical(value),
            Value::String(value) => Storage::String(StringArray::new(vec![value], vec![1, 1]).map_err(internal)?),
            Value::StringArray(value) => Storage::String(value),
            Value::CharArray(value) => Storage::Character(value),
            Value::Complex(real, imaginary) => Storage::Complex(ComplexTensor::new(vec![(real, imaginary)], vec![1, 1]).map_err(internal)?),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Storage::Numeric(tensor::value_into_tensor_for("mat2cell", value).map_err(invalid)?),
            other => return Err(invalid(format!("unsupported input type {other:?}; expected a numeric, logical, string, or character array"))),
        };
        let shape = source_shape(&storage);
        let dimensions = normalized_dimensions(&shape);
        Ok(Self {
            storage,
            shape,
            dimensions,
        })
    }

    pub(super) fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub(super) fn extract(&self, start: &[usize], sizes: &[usize]) -> BuiltinResult<Value> {
        match &self.storage {
            Storage::Numeric(value) => numeric::real(value, &self.shape, start, sizes),
            Storage::Complex(value) => numeric::complex(value, &self.shape, start, sizes),
            Storage::Logical(value) => numeric::logical(value, &self.shape, start, sizes),
            Storage::String(value) => text::string(value, &self.shape, start, sizes),
            Storage::Character(value) => text::character(value, start, sizes),
        }
    }
}

fn source_shape(storage: &Storage) -> Vec<usize> {
    let shape = match storage {
        Storage::Numeric(value) => value.shape.clone(),
        Storage::Complex(value) => value.shape.clone(),
        Storage::Logical(value) => value.shape.clone(),
        Storage::String(value) if value.shape.is_empty() => vec![1, value.rows()],
        Storage::String(value) => value.shape.clone(),
        Storage::Character(value) => vec![value.rows, value.cols],
    };
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape,
    }
}

fn normalized_dimensions(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

fn invalid(detail: String) -> crate::RuntimeError {
    mat2cell_error_with_message(format!("mat2cell: {detail}"), &MAT2CELL_ERROR_INVALID_INPUT)
}

fn internal(error: impl std::fmt::Display) -> crate::RuntimeError {
    mat2cell_error_with_message(format!("mat2cell: {error}"), &MAT2CELL_ERROR_INTERNAL)
}
