use runmat_value::{CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, Tensor, Value};

use super::{UniformScalarCollector, UniformScalarError};

pub(super) fn value(
    collector: UniformScalarCollector<'_>,
    shape: &[usize],
) -> Result<Value, UniformScalarError> {
    match collector {
        UniformScalarCollector::Managed(owner) => owner.finish(shape),
        UniformScalarCollector::Pending => tensor(vec![0.0; shape.iter().product()], shape),
        UniformScalarCollector::Logical(values) => LogicalArray::new(values, shape.to_vec())
            .map(Value::LogicalArray)
            .map_err(UniformScalarError::materialization),
        UniformScalarCollector::F64(values) => tensor(values, shape),
        UniformScalarCollector::F32(values) => Tensor::from_f32(values, shape.to_vec())
            .map(Value::Tensor)
            .map_err(UniformScalarError::materialization),
        UniformScalarCollector::Integer { prototype, values } => {
            let storage = prototype
                .from_same_class_values(values)
                .map_err(UniformScalarError::materialization)?;
            Tensor::new_integer(storage, shape.to_vec())
                .map(Value::Tensor)
                .map_err(UniformScalarError::materialization)
        }
        UniformScalarCollector::ComplexF64(values) => ComplexTensor::new(values, shape.to_vec())
            .map(Value::ComplexTensor)
            .map_err(UniformScalarError::materialization),
        UniformScalarCollector::ComplexF32(values) => {
            ComplexTensor::from_f32(values, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(UniformScalarError::materialization)
        }
        UniformScalarCollector::IntegerComplex {
            prototype,
            real,
            imaginary,
        } => {
            let real = prototype
                .from_same_class_values(real)
                .map_err(UniformScalarError::materialization)?;
            let imaginary = prototype
                .from_same_class_values(imaginary)
                .map_err(UniformScalarError::materialization)?;
            let storage = IntegerComplexStorage::new(real, imaginary)
                .map_err(UniformScalarError::materialization)?;
            ComplexTensor::new_integer(storage, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(UniformScalarError::materialization)
        }
        UniformScalarCollector::Char(values) => character(values, shape),
    }
}

fn tensor(values: Vec<f64>, shape: &[usize]) -> Result<Value, UniformScalarError> {
    Tensor::new(values, shape.to_vec())
        .map(Value::Tensor)
        .map_err(UniformScalarError::materialization)
}

fn character(
    values: Vec<runmat_value::Utf16CodeUnit>,
    shape: &[usize],
) -> Result<Value, UniformScalarError> {
    let normalized = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    if normalized.len() > 2 {
        return Err(UniformScalarError::CHARACTER_RANK);
    }
    let rows = normalized.first().copied().unwrap_or(1);
    let columns = normalized.get(1).copied().unwrap_or(1);
    let expected = rows
        .checked_mul(columns)
        .ok_or(UniformScalarError::SIZE_OVERFLOW)?;
    if expected != values.len() {
        return Err(UniformScalarError::CHARACTER_COUNT);
    }
    let mut row_major = vec![runmat_value::Utf16CodeUnit::from_raw(0); expected];
    for column in 0..columns {
        for row in 0..rows {
            row_major[row * columns + column] = values[row + column * rows];
        }
    }
    CharArray::from_code_units(row_major, rows, columns)
        .map(Value::CharArray)
        .map_err(UniformScalarError::materialization)
}
