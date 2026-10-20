//! Shape and emptiness predicates.

use super::{tensor, RuntimeError, Value};

pub fn value_shape_2d(value: &Value) -> (usize, usize) {
    match value {
        Value::Tensor(t) => dense_matrix_dimensions(t.shape()),
        Value::SparseTensor(t) => (t.rows, t.cols),
        Value::ComplexTensor(t) => (t.rows(), t.cols()),
        Value::SymbolicArray(array) => (array.rows, array.cols),
        Value::ObjectArray(array) => (
            array.shape().first().copied().unwrap_or(1),
            array.shape().get(1).copied().unwrap_or(1),
        ),
        Value::LogicalArray(a) => {
            let rows = a.shape.first().copied().unwrap_or(0);
            let cols = a.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        Value::Cell(c) => (c.rows, c.cols),
        Value::CharArray(c) => (c.rows, c.cols),
        Value::StringArray(s) => (s.rows, s.cols),
        Value::StructArray(array) => (
            array.shape().first().copied().unwrap_or(0),
            array.shape().get(1).copied().unwrap_or(1),
        ),
        Value::GpuTensor(handle) => {
            let rows = handle.shape().first().copied().unwrap_or(1);
            let cols = handle.shape().get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        _ => (1, 1),
    }
}

fn dense_matrix_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.is_empty(),
        Value::SparseTensor(t) => t.rows == 0 || t.cols == 0,
        Value::ComplexTensor(t) => tensor::complex_tensor_element_len(t) == 0,
        Value::LogicalArray(a) => a.data.is_empty(),
        Value::StringArray(s) => s.data().is_empty(),
        Value::CharArray(c) => c.shape.contains(&0),
        Value::Cell(c) => c.data().is_empty(),
        Value::StructArray(array) => array.is_empty(),
        Value::SymbolicArray(array) => array.data().is_empty(),
        Value::ObjectArray(array) => array.is_empty(),
        Value::GpuTensor(handle) => handle.shape().contains(&0),
        _ => false,
    }
}

pub fn value_is_scalar_or_empty(value: &Value) -> bool {
    let shape: &[usize] = match value {
        Value::Tensor(value) => value.shape(),
        Value::ComplexTensor(value) => value.shape(),
        Value::LogicalArray(value) => &value.shape,
        Value::StringArray(value) => &value.shape,
        Value::Cell(value) => &value.shape,
        Value::StructArray(value) => value.shape(),
        Value::SymbolicArray(value) => &value.shape,
        Value::ObjectArray(value) => value.shape(),
        Value::CharArray(value) => &value.shape,
        Value::GpuTensor(handle) => handle.shape(),
        _ => {
            let (rows, cols) = value_shape_2d(value);
            return (rows == 1 && cols == 1) || rows == 0 || cols == 0;
        }
    };
    shape.contains(&0) || shape.iter().all(|extent| *extent == 1)
}

pub fn value_is_column(value: &Value) -> bool {
    let shape: &[usize] = match value {
        Value::Tensor(value) => value.shape(),
        Value::ComplexTensor(value) => value.shape(),
        Value::LogicalArray(value) => &value.shape,
        Value::StringArray(value) => &value.shape,
        Value::Cell(value) => &value.shape,
        Value::StructArray(value) => value.shape(),
        Value::SymbolicArray(value) => &value.shape,
        Value::ObjectArray(value) => value.shape(),
        Value::CharArray(value) => &value.shape,
        Value::GpuTensor(handle) => handle.shape(),
        _ => {
            let (_, cols) = value_shape_2d(value);
            return cols == 1;
        }
    };
    shape.get(1).copied().unwrap_or(1) == 1 && shape.iter().skip(2).all(|extent| *extent == 1)
}

pub fn value_is_vector(value: &Value) -> Result<bool, RuntimeError> {
    fn shape_is_vector(shape: &[usize]) -> bool {
        if crate::builtins::common::shape::effective_rank(shape) > 2 {
            return false;
        }
        let rows = shape.first().copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(1);
        rows == 1 || cols == 1
    }

    Ok(match value {
        Value::Tensor(value) => shape_is_vector(value.shape()),
        Value::ComplexTensor(value) => shape_is_vector(value.shape()),
        Value::LogicalArray(value) => shape_is_vector(&value.shape),
        Value::StringArray(value) => shape_is_vector(&value.shape),
        Value::Cell(value) => shape_is_vector(&value.shape),
        Value::StructArray(value) => shape_is_vector(value.shape()),
        Value::SymbolicArray(value) => shape_is_vector(&value.shape),
        Value::ObjectArray(value) => shape_is_vector(value.shape()),
        Value::CharArray(value) => shape_is_vector(&value.shape),
        Value::GpuTensor(handle) => shape_is_vector(handle.shape()),
        Value::SparseTensor(value) => value.rows == 1 || value.cols == 1,
        _ => true,
    })
}

pub fn value_satisfies_vector_validator(
    value: &Value,
    allow_all_empties: bool,
) -> Result<bool, RuntimeError> {
    Ok(value_is_vector(value)? || (allow_all_empties && value_is_empty(value)))
}

/// Shared callable/arguments-block sparse admission, without builtin dispatch.
pub fn value_satisfies_sparse_validator(value: &Value) -> Result<bool, RuntimeError> {
    super::validate_resident_metadata(value)?;
    Ok(value_is_empty(value) || matches!(value, Value::SparseTensor(_)))
}

pub fn value_satisfies_nonsparse_validator(value: &Value) -> Result<bool, RuntimeError> {
    super::validate_resident_metadata(value)?;
    Ok(!matches!(value, Value::SparseTensor(_)))
}
