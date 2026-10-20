use runmat_value::{
    ComplexElement, ComplexStorage, IntegerStorage, LogicalArray, NumericStorage, Value,
};

use super::{CellEntry, ElementKind, EntryData};
use crate::builtins::cells::core::cell2mat::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL, CELL2MAT_ERROR_INVALID_CONTENTS,
};
use crate::BuiltinResult;

pub(super) fn value(value: Value) -> BuiltinResult<CellEntry> {
    let entry = match value {
        Value::Tensor(tensor) => {
            let shape = normalized_shape(tensor.shape().to_vec());
            let storage = tensor.into_numeric_storage().map_err(|error| {
                cell2mat_error_with_message(format!("cell2mat: {error}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            CellEntry {
                kind: ElementKind::Numeric,
                shape,
                data: EntryData::Numeric(storage),
            }
        }
        Value::Num(number) => numeric_scalar(NumericStorage::F64(vec![number])),
        Value::Int(integer) => {
            numeric_scalar(NumericStorage::from(IntegerStorage::from_scalar(integer)))
        }
        Value::Bool(value) => CellEntry {
            kind: ElementKind::Logical,
            shape: vec![1, 1],
            data: EntryData::Logical(vec![u8::from(value)]),
        },
        Value::LogicalArray(array) => logical(array),
        Value::Complex(real, imaginary) => CellEntry {
            kind: ElementKind::Complex,
            shape: vec![1, 1],
            data: EntryData::Complex(ComplexStorage::F64(
                vec![ComplexElement(real, imaginary)].into(),
            )),
        },
        Value::ComplexTensor(tensor) => {
            let (storage, shape) = tensor.into_complex_storage_and_shape();
            let shape = normalized_shape(shape.to_vec());
            match storage {
                ComplexStorage::Integer(storage) => CellEntry {
                    kind: ElementKind::TypedComplexInteger,
                    shape,
                    data: EntryData::TypedComplexInteger(storage),
                },
                storage @ (ComplexStorage::F32(_) | ComplexStorage::F64(_)) => CellEntry {
                    kind: ElementKind::Complex,
                    shape,
                    data: EntryData::Complex(storage),
                },
            }
        }
        Value::CharArray(array) => CellEntry {
            kind: ElementKind::Character,
            shape: array.shape().to_vec(),
            data: EntryData::Character(array.into_parts().0),
        },
        Value::Cell(_) => return Err(invalid("nested cell arrays are not supported")),
        Value::String(_) | Value::StringArray(_) => {
            return Err(invalid(
                "string inputs are not supported; convert them to character arrays first",
            ))
        }
        Value::GpuTensor(_) => return Err(invalid("a resident value remained after gather")),
        other => return Err(invalid(format!("unsupported cell element type: {other:?}"))),
    };
    Ok(entry)
}

fn numeric_scalar(storage: NumericStorage) -> CellEntry {
    CellEntry {
        kind: ElementKind::Numeric,
        shape: vec![1, 1],
        data: EntryData::Numeric(storage),
    }
}

fn logical(array: LogicalArray) -> CellEntry {
    CellEntry {
        kind: ElementKind::Logical,
        shape: normalized_shape(array.shape.to_vec()),
        data: EntryData::Logical(array.data.to_vec()),
    }
}

fn normalized_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    }
    shape
}

fn invalid(detail: impl Into<String>) -> crate::RuntimeError {
    cell2mat_error_with_message(
        format!("cell2mat: {}", detail.into()),
        &CELL2MAT_ERROR_INVALID_CONTENTS,
    )
}
