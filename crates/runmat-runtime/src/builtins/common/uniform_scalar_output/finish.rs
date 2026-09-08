use runmat_value::{CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, Tensor, Value};

use super::{UniformScalarCollector, UniformScalarError};

pub(super) fn value(
    collector: UniformScalarCollector,
    shape: &[usize],
) -> Result<Value, UniformScalarError> {
    match collector {
        UniformScalarCollector::Pending => tensor(vec![0.0; shape.iter().product()], shape),
        UniformScalarCollector::Logical(values) => LogicalArray::new(values, shape.to_vec())
            .map(Value::LogicalArray)
            .map_err(UniformScalarError::Materialization),
        UniformScalarCollector::F64(values) => tensor(values, shape),
        UniformScalarCollector::F32(values) => Tensor::from_f32(values, shape.to_vec())
            .map(Value::Tensor)
            .map_err(UniformScalarError::Materialization),
        UniformScalarCollector::Integer { prototype, values } => {
            let storage = prototype
                .from_same_class_values(values)
                .map_err(UniformScalarError::Materialization)?;
            Tensor::new_integer(storage, shape.to_vec())
                .map(Value::Tensor)
                .map_err(UniformScalarError::Materialization)
        }
        UniformScalarCollector::ComplexF64(values) => ComplexTensor::new(values, shape.to_vec())
            .map(Value::ComplexTensor)
            .map_err(UniformScalarError::Materialization),
        UniformScalarCollector::ComplexF32(values) => {
            ComplexTensor::from_f32(values, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(UniformScalarError::Materialization)
        }
        UniformScalarCollector::IntegerComplex {
            prototype,
            real,
            imaginary,
        } => {
            let real = prototype
                .from_same_class_values(real)
                .map_err(UniformScalarError::Materialization)?;
            let imaginary = prototype
                .from_same_class_values(imaginary)
                .map_err(UniformScalarError::Materialization)?;
            let storage = IntegerComplexStorage::new(real, imaginary)
                .map_err(UniformScalarError::Materialization)?;
            ComplexTensor::new_integer(storage, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(UniformScalarError::Materialization)
        }
        UniformScalarCollector::Char(values) => character(values, shape),
    }
}

fn tensor(values: Vec<f64>, shape: &[usize]) -> Result<Value, UniformScalarError> {
    Tensor::new(values, shape.to_vec())
        .map(Value::Tensor)
        .map_err(UniformScalarError::Materialization)
}

fn character(values: Vec<char>, shape: &[usize]) -> Result<Value, UniformScalarError> {
    let normalized = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    if normalized.len() > 2 {
        return Err(UniformScalarError::CharacterRank);
    }
    let rows = normalized.first().copied().unwrap_or(1);
    let columns = normalized.get(1).copied().unwrap_or(1);
    let expected = rows
        .checked_mul(columns)
        .ok_or(UniformScalarError::SizeOverflow)?;
    if expected != values.len() {
        return Err(UniformScalarError::CharacterCount);
    }
    let mut row_major = vec!['\0'; expected];
    for column in 0..columns {
        for row in 0..rows {
            row_major[row * columns + column] = values[row + column * rows];
        }
    }
    CharArray::new(row_major, rows, columns)
        .map(Value::CharArray)
        .map_err(UniformScalarError::Materialization)
}
