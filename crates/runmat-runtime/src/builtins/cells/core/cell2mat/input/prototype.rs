use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericStorage};

use super::{CellEntry, EntryData};
use crate::builtins::cells::core::cell2mat::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL, CELL2MAT_ERROR_INVALID_CONTENTS,
};
use crate::BuiltinResult;

pub(in crate::builtins::cells::core::cell2mat) fn typed_complex_integer(
    entries: &[CellEntry],
) -> BuiltinResult<&IntegerComplexStorage> {
    let prototype = entries
        .iter()
        .find_map(|entry| match &entry.data {
            EntryData::TypedComplexInteger(storage) => Some(storage),
            _ => None,
        })
        .ok_or_else(|| internal("typed complex integer contents have no storage"))?;
    if entries.iter().any(|entry| matches!(&entry.data, EntryData::TypedComplexInteger(storage) if storage.numeric_dtype() != prototype.numeric_dtype())) {
        return Err(invalid(
            "typed complex integer contents must share the same integer class",
        ));
    }
    Ok(prototype)
}

pub(in crate::builtins::cells::core::cell2mat) fn numeric(
    entries: &[CellEntry],
) -> BuiltinResult<&NumericStorage> {
    let integer = entries.iter().find_map(|entry| match &entry.data {
        EntryData::Numeric(storage)
            if !storage.is_empty() && storage.clone().into_integer_storage().is_ok() =>
        {
            Some(storage)
        }
        _ => None,
    });
    let prototype = integer
        .or_else(|| nonempty_numeric(entries))
        .or_else(|| any_numeric(entries))
        .ok_or_else(|| internal("numeric contents have no storage"))?;
    validate_numeric_classes(entries, prototype)?;
    Ok(prototype)
}

fn validate_numeric_classes(
    entries: &[CellEntry],
    prototype: &NumericStorage,
) -> BuiltinResult<()> {
    let integer = prototype.clone().into_integer_storage().ok();
    let scalar_double = entries.iter().any(|entry| matches!(&entry.data, EntryData::Numeric(NumericStorage::F64(values)) if values.len() == 1));
    let compatible = entries.iter().all(|entry| match &entry.data {
        EntryData::Numeric(storage) if storage.clone().into_integer_storage().is_ok() => true,
        EntryData::Numeric(NumericStorage::F64(values)) => values.len() == 1,
        _ => true,
    });
    let scalar_allowed = integer.as_ref().is_some_and(|storage| {
        !scalar_double || !matches!(storage, IntegerStorage::I64(_) | IntegerStorage::U64(_))
    });
    let mixed_class = entries.iter().any(|entry| matches!(&entry.data, EntryData::Numeric(storage) if storage.numeric_dtype() != prototype.numeric_dtype()));
    if !(integer.is_some() && compatible && scalar_allowed) && mixed_class {
        return Err(invalid("floating-point contents must share a class; scalar doubles cannot concatenate with int64 or uint64"));
    }
    Ok(())
}

fn nonempty_numeric(entries: &[CellEntry]) -> Option<&NumericStorage> {
    entries.iter().find_map(|entry| match &entry.data {
        EntryData::Numeric(storage) if !storage.is_empty() => Some(storage),
        _ => None,
    })
}

fn any_numeric(entries: &[CellEntry]) -> Option<&NumericStorage> {
    entries.iter().find_map(|entry| match &entry.data {
        EntryData::Numeric(storage) => Some(storage),
        _ => None,
    })
}

fn invalid(detail: &'static str) -> crate::RuntimeError {
    cell2mat_error_with_message(
        format!("cell2mat: {detail}"),
        &CELL2MAT_ERROR_INVALID_CONTENTS,
    )
}

fn internal(detail: &'static str) -> crate::RuntimeError {
    cell2mat_error_with_message(format!("cell2mat: {detail}"), &CELL2MAT_ERROR_INTERNAL)
}
