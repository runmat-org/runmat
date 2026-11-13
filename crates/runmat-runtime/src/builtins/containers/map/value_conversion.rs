use super::*;
use runmat_value::{IntegerStorage, NumericDType};

pub(super) fn normalize_numeric_value(
    value: Value,
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) if dtype == NumericDType::F64 => Ok(Value::Num(value)),
        Value::Num(value) => Tensor::from_f32(vec![value as f32], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| map_error(format!("containers.Map: {err}"), builtin)),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype)))
        }
        Value::Int(value) => normalize_numeric_value(Value::Num(value.to_f64()), dtype, builtin),
        Value::Bool(value) => {
            normalize_numeric_value(Value::Num(if value { 1.0 } else { 0.0 }), dtype, builtin)
        }
        Value::LogicalArray(arr) if arr.data.len() == 1 => {
            let data: Vec<f64> = arr
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, arr.shape.clone())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
            Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype)))
        }
        Value::Cell(_)
        | Value::SparseTensor(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::GpuTensor(_)
        | Value::OutputList(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => Err(map_error(
            "containers.Map: values must be numeric scalars when ValueType is 'double' or 'single'",
            builtin,
        )),
    }
}

pub(super) fn normalize_logical_value(value: Value, builtin: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) => Ok(value),
        Value::LogicalArray(ref array) if array.data.len() == 1 => Ok(value),
        Value::Int(i) => Ok(Value::Bool(!i.is_zero())),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Tensor(t) if t.len() == 1 => {
            let flags: Vec<u8> = if let Some(storage) = t.integer_storage() {
                storage
                    .exact_values()
                    .into_iter()
                    .map(|value| if value.is_zero() { 0 } else { 1 })
                    .collect()
            } else {
                (0..t.len())
                    .map(|index| {
                        if tensor::tensor_value_f64(&t, index) != 0.0 {
                            1
                        } else {
                            0
                        }
                    })
                    .collect()
            };
            let logical = LogicalArray::new(flags, t.shape().to_vec())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
            Ok(Value::LogicalArray(logical))
        }
        Value::CharArray(_)
        | Value::SparseTensor(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Cell(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::GpuTensor(_)
        | Value::OutputList(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => Err(map_error(
            "containers.Map: values must be logical scalars when ValueType is 'logical'",
            builtin,
        )),
    }
}

pub(super) fn integer_dtype_for_value_type(value_type: ValueType) -> Option<NumericDType> {
    match value_type {
        ValueType::Int8 => Some(NumericDType::I8),
        ValueType::UInt8 => Some(NumericDType::U8),
        ValueType::Int16 => Some(NumericDType::I16),
        ValueType::UInt16 => Some(NumericDType::U16),
        ValueType::Int32 => Some(NumericDType::I32),
        ValueType::UInt32 => Some(NumericDType::U32),
        ValueType::Int64 => Some(NumericDType::I64),
        ValueType::UInt64 => Some(NumericDType::U64),
        _ => None,
    }
}

pub(super) fn normalize_integer_value(
    value: Value,
    value_type: ValueType,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let dtype = integer_dtype_for_value_type(value_type)
        .ok_or_else(|| map_internal("containers.Map: invalid integer ValueType", builtin))?;
    let (tensor, scalar) = match value {
        Value::Tensor(tensor) if tensor.len() == 1 => (tensor, true),
        Value::Int(value) => (
            Tensor::new_integer(integer_storage_from_scalar(value), vec![1, 1])
                .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        Value::Num(value) => (
            Tensor::new(vec![value], vec![1, 1])
                .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        Value::Bool(value) => (
            Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        Value::LogicalArray(array) if array.data.len() == 1 => (
            Tensor::new(
                array
                    .data
                    .iter()
                    .map(|value| f64::from(*value != 0))
                    .collect(),
                array.shape,
            )
            .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
            true,
        ),
        _ => {
            return Err(map_error(
                "containers.Map: value cannot be converted to the declared integer ValueType",
                builtin,
            ))
        }
    };
    let converted = tensor::coerce_tensor_dtype(tensor, dtype);
    if scalar {
        let value = converted
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
            .ok_or_else(|| {
                map_internal("containers.Map: integer scalar conversion failed", builtin)
            })?;
        Ok(Value::Int(value))
    } else {
        Ok(Value::Tensor(converted))
    }
}

pub(super) fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

pub(super) fn numeric_from_value(
    value: &Value,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(arr) if arr.data.len() == 1 => {
            Ok(if arr.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(map_error(context, builtin)),
    }
}

pub(super) fn integer_from_value(
    value: &Value,
    min: i64,
    max: i64,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<i64> {
    match value {
        Value::Int(i) => {
            let Some(v) = i.try_to_i64() else {
                return Err(map_error(context, builtin));
            };
            if v < min || v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let Some(v) = storage.value_at(0).and_then(|value| value.try_to_i64()) else {
                    return Err(map_error(context, builtin));
                };
                if v < min || v > max {
                    return Err(map_error(context, builtin));
                }
                Ok(v)
            } else {
                integer_from_value(
                    &Value::Num(tensor::tensor_value_f64(t, 0)),
                    min,
                    max,
                    context,
                    builtin,
                )
            }
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(map_error(context, builtin));
            }
            if (*n < min as f64) || (*n > max as f64) {
                return Err(map_error(context, builtin));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(map_error(context, builtin));
            }
            Ok(n.round() as i64)
        }
        Value::Bool(b) => {
            let v = if *b { 1 } else { 0 };
            if v < min || v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        _ => Err(map_error(context, builtin)),
    }
}

pub(super) fn unsigned_from_value(
    value: &Value,
    max: u64,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<u64> {
    match value {
        Value::Int(i) => {
            let Some(v) = i.try_to_u64() else {
                return Err(map_error(context, builtin));
            };
            if v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let Some(v) = storage.value_at(0).and_then(|value| value.try_to_u64()) else {
                    return Err(map_error(context, builtin));
                };
                if v > max {
                    return Err(map_error(context, builtin));
                }
                Ok(v)
            } else {
                unsigned_from_value(
                    &Value::Num(tensor::tensor_value_f64(t, 0)),
                    max,
                    context,
                    builtin,
                )
            }
        }
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 || *n > max as f64 {
                return Err(map_error(context, builtin));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(map_error(context, builtin));
            }
            Ok(n.round() as u64)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(map_error(context, builtin)),
    }
}

pub(super) fn bool_from_value(
    value: &Value,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::LogicalArray(arr) if arr.data.len() == 1 => Ok(arr.data[0] != 0),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                Ok(!storage
                    .value_at(0)
                    .ok_or_else(|| map_error(context, builtin))?
                    .is_zero())
            } else {
                Ok(tensor::tensor_value_f64(t, 0) != 0.0)
            }
        }
        _ => Err(map_error(context, builtin)),
    }
}
