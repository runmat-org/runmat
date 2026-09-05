use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::ProjectionKind;

pub(super) fn execute(kind: ProjectionKind, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(real, imaginary) => Ok(Value::Num(match kind {
            ProjectionKind::Real => real,
            ProjectionKind::Imaginary => imaginary,
        })),
        Value::ComplexTensor(tensor) => project_complex_tensor(kind, tensor),
        Value::CharArray(chars) => project_chars(kind, chars),
        Value::String(_) | Value::StringArray(_) => Err(kind.invalid("expected numeric input")),
        value @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => project_real_value(kind, value),
        other => Err(kind.invalid(format!(
            "unsupported input type {other:?}; expected numeric, logical, or char input"
        ))),
    }
}

fn project_real_value(kind: ProjectionKind, value: Value) -> BuiltinResult<Value> {
    let tensor =
        tensor::value_into_tensor_for(kind.name(), value).map_err(|error| kind.invalid(error))?;
    let tensor = match kind {
        ProjectionKind::Real => tensor,
        ProjectionKind::Imaginary => zero_tensor(kind, tensor)?,
    };
    Ok(tensor::tensor_into_value(tensor))
}

fn zero_tensor(kind: ProjectionKind, tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| kind.internal(error))?;
    Tensor::from_numeric_storage(storage.zeros_like(storage.len()), shape)
        .map_err(|error| kind.internal(error))
}

fn project_complex_tensor(kind: ProjectionKind, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = match (kind, tensor.into_complex_storage()) {
        (ProjectionKind::Real, ComplexStorage::F64(values)) => {
            NumericStorage::F64(values.into_iter().map(|(real, _)| real).collect())
        }
        (ProjectionKind::Imaginary, ComplexStorage::F64(values)) => {
            NumericStorage::F64(values.into_iter().map(|(_, imaginary)| imaginary).collect())
        }
        (ProjectionKind::Real, ComplexStorage::F32(values)) => {
            NumericStorage::F32(values.into_iter().map(|(real, _)| real).collect())
        }
        (ProjectionKind::Imaginary, ComplexStorage::F32(values)) => {
            NumericStorage::F32(values.into_iter().map(|(_, imaginary)| imaginary).collect())
        }
        (ProjectionKind::Real, ComplexStorage::Integer(storage)) => {
            NumericStorage::from_integer_storage(storage.real)
        }
        (ProjectionKind::Imaginary, ComplexStorage::Integer(storage)) => {
            NumericStorage::from_integer_storage(storage.imag)
        }
    };
    let tensor =
        Tensor::from_numeric_storage(storage, shape).map_err(|error| kind.internal(error))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn project_chars(kind: ProjectionKind, chars: CharArray) -> BuiltinResult<Value> {
    let data = match kind {
        ProjectionKind::Real => chars
            .data
            .iter()
            .map(|&character| character as u32 as f64)
            .collect(),
        ProjectionKind::Imaginary => vec![0.0; chars.rows * chars.cols],
    };
    let tensor =
        Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|error| kind.internal(error))?;
    Ok(tensor::tensor_into_value(tensor))
}
