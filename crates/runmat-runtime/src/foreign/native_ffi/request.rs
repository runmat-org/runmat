use std::path::PathBuf;

use runmat_native_ffi::{NativeScalar, NativeType};
use runmat_value::Value;

use super::super::{foreign_error, ForeignErrorKind};
use crate::RuntimeError;

pub(super) fn string_argument(
    arguments: &[Value],
    index: usize,
    operation: &str,
) -> Result<String, RuntimeError> {
    let value = arguments.get(index).ok_or_else(|| {
        invalid_call(format!(
            "native FFI operation {operation} requires argument {}",
            index + 1
        ))
    })?;
    String::try_from(value).map_err(|_| {
        invalid_call(format!(
            "argument {} to native FFI operation {operation} must be text",
            index + 1
        ))
    })
}

pub(super) fn optional_string_argument(
    arguments: &[Value],
    index: usize,
    operation: &str,
) -> Result<Option<String>, RuntimeError> {
    arguments
        .get(index)
        .map(|_| string_argument(arguments, index, operation))
        .transpose()
}

pub(super) fn default_alias(path: &str) -> Result<String, RuntimeError> {
    PathBuf::from(path)
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| invalid_call("native library path has no valid file name"))
}

pub(super) fn legacy_pointer_type(name: &str) -> Result<NativeType, RuntimeError> {
    let normalized = name.trim().to_ascii_lowercase();
    let scalar = match normalized.as_str() {
        "logicalptr" | "boolptr" => NativeScalar::Bool,
        "charptr" | "int8ptr" => NativeScalar::I8,
        "uint8ptr" => NativeScalar::U8,
        "int16ptr" | "shortptr" => NativeScalar::I16,
        "uint16ptr" | "ushortptr" => NativeScalar::U16,
        "int32ptr" | "intptr" => NativeScalar::I32,
        "uint32ptr" | "uintptr" => NativeScalar::U32,
        "int64ptr" | "longlongptr" => NativeScalar::I64,
        "uint64ptr" | "ulonglongptr" => NativeScalar::U64,
        "singleptr" | "floatptr" => NativeScalar::F32,
        "doubleptr" => NativeScalar::F64,
        _ => {
            return Err(invalid_call(format!(
                "unsupported libpointer type `{name}`; use a normalized scalar pointer type"
            )))
        }
    };
    Ok(NativeType::Scalar { scalar })
}

pub(super) fn invalid_call(message: impl Into<String>) -> RuntimeError {
    foreign_error(ForeignErrorKind::InvalidCall, message)
}
