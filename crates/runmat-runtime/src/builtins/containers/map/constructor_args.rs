//! Constructor argument decoding and exact key/value type selection.

use super::*;
use crate::builtins::common::random_args::keyword_of;
use runmat_value::{IntegerStorage, NumericDType};

pub(super) struct ConstructorArgs {
    pub(super) key_type: KeyType,
    pub(super) value_type: ValueType,
    pub(super) uniform_values: bool,
    pub(super) keys: Vec<KeyCandidate>,
    pub(super) values: Vec<Value>,
}

pub(super) struct KeyCandidate {
    pub(super) normalized: NormalizedKey,
    pub(super) canonical: Value,
}

pub(super) async fn parse_constructor_args(
    args: Vec<Value>,
    builtin: &'static str,
) -> BuiltinResult<ConstructorArgs> {
    let mut index = 0usize;
    let mut keys_input: Option<Value> = None;
    let mut values_input: Option<Value> = None;

    if index < args.len() && keyword_of(&args[index])?.is_none() {
        if args.len() < 2 {
            return Err(map_error(
                "containers.Map: constructor requires both keys and values when either is provided.",
                builtin,
            ));
        }
        keys_input = Some(args[index].clone());
        values_input = Some(args[index + 1].clone());
        index += 2;
    }

    let has_data = keys_input.is_some();
    let mut key_type = KeyType::Char;
    let mut value_type = ValueType::Any;
    let mut uniform_values = has_data;
    let mut key_type_explicit = false;
    let mut value_type_explicit = false;
    while index < args.len() {
        let keyword = keyword_of(&args[index])?.ok_or_else(|| {
            map_error(
                "containers.Map: expected option name (e.g. 'KeyType')",
                builtin,
            )
        })?;
        index += 1;
        let Some(value) = args.get(index) else {
            return Err(map_error(
                format!("containers.Map: missing value for option '{keyword}'"),
                builtin,
            ));
        };
        index += 1;
        match keyword.as_str() {
            "keytype" => {
                if has_data {
                    return Err(map_error(
                        "containers.Map: KeyType is only valid for an empty Map constructor",
                        builtin,
                    ));
                }
                key_type = KeyType::parse(value, builtin)?;
                key_type_explicit = true;
            }
            "valuetype" => {
                if has_data {
                    return Err(map_error(
                        "containers.Map: ValueType is only valid for an empty Map constructor",
                        builtin,
                    ));
                }
                value_type = ValueType::parse(value, builtin)?;
                value_type_explicit = true;
            }
            "uniformvalues" => {
                if !has_data {
                    return Err(map_error(
                        "containers.Map: UniformValues requires keySet and valueSet",
                        builtin,
                    ));
                }
                uniform_values = bool_from_value(
                    value,
                    "containers.Map: UniformValues must be logical",
                    builtin,
                )?
            }
            other => {
                return Err(map_error(
                    format!("containers.Map: unrecognised option '{other}'"),
                    builtin,
                ));
            }
        }
    }

    if !has_data && (key_type_explicit != value_type_explicit) {
        return Err(map_error(
            "containers.Map: KeyType and ValueType must both be specified for an empty typed Map",
            builtin,
        ));
    }
    if let Some(keys) = &keys_input {
        key_type = infer_key_type(keys, builtin)?;
    }
    if let Some(values) = &values_input {
        value_type = if uniform_values {
            infer_value_type(values)
        } else {
            ValueType::Any
        };
    }

    let keys = match keys_input {
        Some(value) => prepare_keys(value, key_type, builtin).await?,
        None => Vec::new(),
    };

    let values = match values_input {
        Some(value) => prepare_values(value, builtin).await?,
        None => Vec::new(),
    };

    if keys.len() != values.len() {
        return Err(map_error(
            format!(
                "containers.Map: number of keys ({}) must match number of values ({})",
                keys.len(),
                values.len()
            ),
            builtin,
        ));
    }

    Ok(ConstructorArgs {
        key_type,
        value_type,
        uniform_values,
        keys,
        values,
    })
}

pub(super) fn infer_key_type(value: &Value, builtin: &'static str) -> BuiltinResult<KeyType> {
    match value {
        Value::CharArray(_) | Value::StringArray(_) | Value::String(_) => Ok(KeyType::Char),
        Value::Cell(cell)
            if cell.data().iter().all(|value| {
                matches!(value, Value::CharArray(_) | Value::String(_) | Value::StringArray(_))
            }) =>
        {
            Ok(KeyType::Char)
        }
        Value::Num(_) => Ok(KeyType::Double),
        Value::Tensor(tensor) => key_type_from_dtype(tensor.numeric_dtype()).ok_or_else(|| {
            map_error(
                "containers.Map: numeric key arrays must be double, single, int32, uint32, int64, or uint64",
                builtin,
            )
        }),
        Value::Int(value) => key_type_from_dtype(int_value_dtype(value)).ok_or_else(|| {
            map_error(
                "containers.Map: integer keys must be int32, uint32, int64, or uint64",
                builtin,
            )
        }),
        Value::GpuTensor(handle) => {
            if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
                match integer_type {
                    runmat_accelerate_api::IntegerElementType::I32 => Ok(KeyType::Int32),
                    runmat_accelerate_api::IntegerElementType::U32 => Ok(KeyType::UInt32),
                    runmat_accelerate_api::IntegerElementType::I64 => Ok(KeyType::Int64),
                    runmat_accelerate_api::IntegerElementType::U64 => Ok(KeyType::UInt64),
                    _ => Err(map_error(
                        "containers.Map: resident integer keys must be int32, uint32, int64, or uint64",
                        builtin,
                    )),
                }
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                Ok(KeyType::Single)
            } else {
                Ok(KeyType::Double)
            }
        }
        _ => Err(map_error(
            "containers.Map: keys must be a numeric array, cell array of character vectors, or string array",
            builtin,
        )),
    }
}

fn key_type_from_dtype(dtype: NumericDType) -> Option<KeyType> {
    match dtype {
        NumericDType::F64 => Some(KeyType::Double),
        NumericDType::F32 => Some(KeyType::Single),
        NumericDType::I32 => Some(KeyType::Int32),
        NumericDType::U32 => Some(KeyType::UInt32),
        NumericDType::I64 => Some(KeyType::Int64),
        NumericDType::U64 => Some(KeyType::UInt64),
        _ => None,
    }
}

pub(super) fn int_value_dtype(value: &IntValue) -> NumericDType {
    IntegerStorage::from_scalar(value.clone()).numeric_dtype()
}

pub(super) fn infer_value_type(value: &Value) -> ValueType {
    match value {
        Value::Num(_) => ValueType::Double,
        Value::Tensor(tensor) => value_type_from_dtype(tensor.numeric_dtype()),
        Value::Int(value) => value_type_from_dtype(int_value_dtype(value)),
        Value::Bool(_) | Value::LogicalArray(_) => ValueType::Logical,
        Value::CharArray(_) => ValueType::Char,
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                ValueType::Logical
            } else if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
                match integer_type {
                    runmat_accelerate_api::IntegerElementType::I8 => ValueType::Int8,
                    runmat_accelerate_api::IntegerElementType::U8 => ValueType::UInt8,
                    runmat_accelerate_api::IntegerElementType::I16 => ValueType::Int16,
                    runmat_accelerate_api::IntegerElementType::U16 => ValueType::UInt16,
                    runmat_accelerate_api::IntegerElementType::I32 => ValueType::Int32,
                    runmat_accelerate_api::IntegerElementType::U32 => ValueType::UInt32,
                    runmat_accelerate_api::IntegerElementType::I64 => ValueType::Int64,
                    runmat_accelerate_api::IntegerElementType::U64 => ValueType::UInt64,
                }
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                ValueType::Single
            } else {
                ValueType::Double
            }
        }
        Value::Cell(cell) => {
            let mut iter = cell.data().iter();
            let Some(first) = iter.next() else {
                return ValueType::Any;
            };
            let inferred = infer_value_type(first);
            if inferred != ValueType::Any && iter.all(|value| infer_value_type(value) == inferred) {
                inferred
            } else {
                ValueType::Any
            }
        }
        _ => ValueType::Any,
    }
}

fn value_type_from_dtype(dtype: NumericDType) -> ValueType {
    match dtype {
        NumericDType::F64 => ValueType::Double,
        NumericDType::F32 => ValueType::Single,
        NumericDType::I8 => ValueType::Int8,
        NumericDType::U8 => ValueType::UInt8,
        NumericDType::I16 => ValueType::Int16,
        NumericDType::U16 => ValueType::UInt16,
        NumericDType::I32 => ValueType::Int32,
        NumericDType::U32 => ValueType::UInt32,
        NumericDType::I64 => ValueType::Int64,
        NumericDType::U64 => ValueType::UInt64,
    }
}
