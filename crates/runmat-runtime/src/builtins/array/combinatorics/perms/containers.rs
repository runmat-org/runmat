use runmat_builtins::{PERMS_ERROR_INTERNAL, PERMS_ERROR_TOO_LARGE};
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericStorage, StringArray, Tensor, Value,
};

use crate::BuiltinResult;

use super::super::enumeration::{self, EnumerationError};
use super::{cardinality, error, shape};

pub(super) fn numeric(tensor: Tensor) -> BuiltinResult<Value> {
    let elements = shape::vector_len(&tensor.shape)?;
    let rows = cardinality::output_rows(elements)?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| internal(detail))?;
    let storage = permute_numeric_storage(storage, rows)?;
    Tensor::from_numeric_storage(storage, vec![rows, elements])
        .map(Value::Tensor)
        .map_err(internal)
}

fn permute_numeric_storage(storage: NumericStorage, rows: usize) -> BuiltinResult<NumericStorage> {
    macro_rules! permute {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(permuted_columns(&$values, rows)?)
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => permute!(values, F64),
        NumericStorage::F32(values) => permute!(values, F32),
        NumericStorage::I8(values) => permute!(values, I8),
        NumericStorage::I16(values) => permute!(values, I16),
        NumericStorage::I32(values) => permute!(values, I32),
        NumericStorage::I64(values) => permute!(values, I64),
        NumericStorage::U8(values) => permute!(values, U8),
        NumericStorage::U16(values) => permute!(values, U16),
        NumericStorage::U32(values) => permute!(values, U32),
        NumericStorage::U64(values) => permute!(values, U64),
    })
}

pub(super) fn complex(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let elements = shape::vector_len(&tensor.shape)?;
    let rows = cardinality::output_rows(elements)?;
    if let Some(storage) = tensor.integer_storage() {
        let storage = storage
            .reorder(|values| {
                enumeration::permuted_columns(values, rows)
                    .map_err(|failure| format!("{failure:?}"))
            })
            .map_err(internal)?;
        return ComplexTensor::new_integer(storage, vec![rows, elements])
            .map(Value::ComplexTensor)
            .map_err(internal);
    }
    let data = permuted_columns(&tensor.materialize_f64(), rows)?;
    ComplexTensor::new(data, vec![rows, elements])
        .map(Value::ComplexTensor)
        .map_err(internal)
}

pub(super) fn logical(array: LogicalArray) -> BuiltinResult<Value> {
    let elements = shape::vector_len(&array.shape)?;
    let rows = cardinality::output_rows(elements)?;
    LogicalArray::new(permuted_columns(&array.data, rows)?, vec![rows, elements])
        .map(Value::LogicalArray)
        .map_err(internal)
}

pub(super) fn characters(chars: CharArray) -> BuiltinResult<Value> {
    let elements = shape::vector_len(&[chars.rows, chars.cols])?;
    let rows = cardinality::output_rows(elements)?;
    CharArray::new(permuted_rows(&chars.data, rows)?, rows, elements)
        .map(Value::CharArray)
        .map_err(internal)
}

pub(super) fn strings(array: StringArray) -> BuiltinResult<Value> {
    let elements = shape::vector_len(&array.shape)?;
    let rows = cardinality::output_rows(elements)?;
    StringArray::new(permuted_columns(&array.data, rows)?, vec![rows, elements])
        .map(Value::StringArray)
        .map_err(internal)
}

pub(super) fn cells(cell: CellArray) -> BuiltinResult<Value> {
    let elements = shape::vector_len(&cell.shape)?;
    let rows = cardinality::output_rows(elements)?;
    CellArray::new(permuted_rows(&cell.data, rows)?, rows, elements)
        .map(Value::Cell)
        .map_err(internal)
}

fn permuted_columns<T: Clone>(values: &[T], rows: usize) -> BuiltinResult<Vec<T>> {
    enumeration::permuted_columns(values, rows).map_err(enumeration_error)
}

fn permuted_rows<T: Clone>(values: &[T], rows: usize) -> BuiltinResult<Vec<T>> {
    enumeration::permuted_rows(values, rows).map_err(enumeration_error)
}

fn enumeration_error(failure: EnumerationError) -> crate::RuntimeError {
    let descriptor = match failure {
        EnumerationError::CardinalityOverflow | EnumerationError::ElementLimitExceeded => {
            &PERMS_ERROR_TOO_LARGE
        }
        EnumerationError::SequenceInvariant => &PERMS_ERROR_INTERNAL,
    };
    error::with_message(descriptor, format!("perms: {failure:?}"))
}

fn internal(detail: impl std::fmt::Display) -> crate::RuntimeError {
    error::with_message(&PERMS_ERROR_INTERNAL, format!("perms: {detail}"))
}
