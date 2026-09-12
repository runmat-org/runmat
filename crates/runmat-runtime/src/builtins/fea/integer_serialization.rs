use crate::builtins::fea::errors::{builtin_error, builtin_error_with_source};
use crate::builtins::fea::registry::ensure_fea_classes_registered;
use crate::builtins::io::json::jsondecode::value_from_json;
use crate::BuiltinResult;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::IntValue;
use runmat_value::IntegerStorage;
use runmat_value::ObjectInstance;
use runmat_value::Tensor;
use runmat_value::Value;
use serde::Serialize;

pub(in crate::builtins::fea) fn serializable_to_object<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: runmat_types::StaticClassIdentity,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    serializable_to_object_value(builtin, error, class_name, value, hidden_json_property)
        .map(Value::Object)
}

pub(in crate::builtins::fea) fn serializable_to_object_preserving_integers<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: runmat_types::StaticClassIdentity,
    value: &T,
    hidden_json_property: Option<&'static str>,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let object =
        serializable_to_object_value(builtin, error, class_name, value, hidden_json_property)?;
    let mut wrapped = Value::Object(object);
    promote_named_integer_fields(
        builtin,
        error,
        &mut wrapped,
        &json,
        signed_fields,
        unsigned_fields,
    )?;
    Ok(wrapped)
}

pub(in crate::builtins::fea) fn promote_named_integer_fields(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
    json: &serde_json::Value,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<()> {
    match (value, json) {
        (Value::Object(object), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                let Some(target) = object.properties.get_mut(name) else {
                    continue;
                };
                promote_named_integer_field(
                    builtin,
                    error,
                    name,
                    target,
                    child,
                    signed_fields,
                    unsigned_fields,
                )?;
            }
        }
        (Value::Struct(value), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                let Some(target) = value.fields.get_mut(name) else {
                    continue;
                };
                promote_named_integer_field(
                    builtin,
                    error,
                    name,
                    target,
                    child,
                    signed_fields,
                    unsigned_fields,
                )?;
            }
        }
        (Value::Struct(value), serde_json::Value::Array(items)) if items.len() == 1 => {
            if let serde_json::Value::Object(map) = &items[0] {
                for (name, child) in map {
                    let Some(target) = value.fields.get_mut(name) else {
                        continue;
                    };
                    promote_named_integer_field(
                        builtin,
                        error,
                        name,
                        target,
                        child,
                        signed_fields,
                        unsigned_fields,
                    )?;
                }
            }
        }
        (Value::StructArray(array), serde_json::Value::Array(items)) => {
            let Some((json_shape, leaves)) =
                crate::builtins::io::json::layout::rectangular_leaves(items)
            else {
                return Ok(());
            };
            let shape = if json_shape.len() == 1 {
                vec![json_shape[0], 1]
            } else {
                json_shape
            };
            if shape != array.shape() {
                return Ok(());
            }
            let leaves = crate::builtins::io::json::layout::row_to_column_major(leaves, &shape)
                .map_err(|message| builtin_error(builtin, error, message))?;
            array.try_for_each_indexed_value_mut(|name, index, target| -> BuiltinResult<()> {
                let Some(serde_json::Value::Object(map)) = leaves.get(index).copied() else {
                    return Ok(());
                };
                let Some(child) = map.get(name) else {
                    return Ok(());
                };
                promote_named_integer_field(
                    builtin,
                    error,
                    name,
                    target,
                    child,
                    signed_fields,
                    unsigned_fields,
                )?;
                Ok(())
            })?;
        }
        (Value::Cell(cell), serde_json::Value::Array(items)) => {
            for (target, child) in cell.data.iter_mut().zip(items) {
                promote_named_integer_fields(
                    builtin,
                    error,
                    target,
                    child,
                    signed_fields,
                    unsigned_fields,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(in crate::builtins::fea) fn promote_named_integer_field(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    name: &str,
    target: &mut Value,
    child: &serde_json::Value,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<()> {
    let exact = if signed_fields.contains(&name) {
        exact_integer_json_value(builtin, error, child, true)?
    } else if unsigned_fields.contains(&name) {
        exact_integer_json_value(builtin, error, child, false)?
    } else {
        return promote_named_integer_fields(
            builtin,
            error,
            target,
            child,
            signed_fields,
            unsigned_fields,
        );
    };
    if let Some(exact) = exact {
        *target = exact;
    }
    Ok(())
}

pub(in crate::builtins::fea) fn exact_integer_json_value(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    json: &serde_json::Value,
    signed: bool,
) -> BuiltinResult<Option<Value>> {
    match json {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(number) if signed => number
            .as_i64()
            .map(|value| Some(Value::Int(IntValue::I64(value))))
            .ok_or_else(|| {
                builtin_error(builtin, error, "signed structural integer is out of range")
            }),
        serde_json::Value::Number(number) => number
            .as_u64()
            .map(|value| Some(Value::Int(IntValue::U64(value))))
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    error,
                    "unsigned structural integer is out of range",
                )
            }),
        serde_json::Value::Array(items) => {
            if signed {
                let values = items
                    .iter()
                    .map(|item| {
                        item.as_i64().ok_or_else(|| {
                            builtin_error(
                                builtin,
                                error,
                                "signed structural integer array is out of range",
                            )
                        })
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?;
                Tensor::new_integer(IntegerStorage::I64(values), vec![1, items.len()])
                    .map(Value::Tensor)
                    .map(Some)
                    .map_err(|message| builtin_error(builtin, error, message))
            } else {
                let values = items
                    .iter()
                    .map(|item| {
                        item.as_u64().ok_or_else(|| {
                            builtin_error(
                                builtin,
                                error,
                                "unsigned structural integer array is out of range",
                            )
                        })
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?;
                Tensor::new_integer(IntegerStorage::U64(values), vec![1, items.len()])
                    .map(Value::Tensor)
                    .map(Some)
                    .map_err(|message| builtin_error(builtin, error, message))
            }
        }
        _ => Err(builtin_error(
            builtin,
            error,
            "structural integer field has a noninteger representation",
        )),
    }
}

pub(in crate::builtins::fea) fn serializable_to_value_preserving_integers<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &T,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let mut converted = value_from_json_preserving_integer_kinds(builtin, error, &json)?;
    promote_named_integer_fields(
        builtin,
        error,
        &mut converted,
        &json,
        signed_fields,
        unsigned_fields,
    )?;
    Ok(converted)
}

pub(in crate::builtins::fea) fn value_from_json_preserving_integer_kinds(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    json: &serde_json::Value,
) -> BuiltinResult<Value> {
    let mut converted = value_from_json(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.message().to_string(), err))?;
    promote_json_integer_kinds(builtin, error, &mut converted, json)?;
    Ok(converted)
}

pub(in crate::builtins::fea) fn promote_json_integer_kinds(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
    json: &serde_json::Value,
) -> BuiltinResult<()> {
    match (value, json) {
        (target, serde_json::Value::Number(number)) if number.is_u64() => {
            *target = Value::Int(IntValue::U64(
                number.as_u64().expect("checked unsigned number"),
            ));
        }
        (target, serde_json::Value::Number(number)) if number.is_i64() => {
            *target = Value::Int(IntValue::I64(
                number.as_i64().expect("checked signed number"),
            ));
        }
        (Value::Tensor(tensor), serde_json::Value::Array(_)) => {
            if let Some(storage) = exact_json_integer_array(json, &tensor.shape) {
                *tensor = Tensor::new_integer(storage, tensor.shape.clone())
                    .map_err(|message| builtin_error(builtin, error, message))?;
            }
        }
        (Value::Struct(structure), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if let Some(target) = structure.fields.get_mut(name) {
                    promote_json_integer_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Object(object), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if let Some(target) = object.properties.get_mut(name) {
                    promote_json_integer_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Cell(cell), serde_json::Value::Array(items)) => {
            for (target, child) in cell.data.iter_mut().zip(items) {
                promote_json_integer_kinds(builtin, error, target, child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(in crate::builtins::fea) fn exact_json_integer_array(
    json: &serde_json::Value,
    shape: &[usize],
) -> Option<IntegerStorage> {
    fn collect<'a>(
        json: &'a serde_json::Value,
        numbers: &mut Vec<&'a serde_json::Number>,
    ) -> Option<()> {
        match json {
            serde_json::Value::Number(number) if number.is_i64() || number.is_u64() => {
                numbers.push(number);
                Some(())
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    collect(item, numbers)?;
                }
                Some(())
            }
            _ => None,
        }
    }

    let mut numbers = Vec::new();
    collect(json, &mut numbers)?;
    if numbers.is_empty() || numbers.len() != shape.iter().product::<usize>() {
        return None;
    }
    let row_major_index = |column_major_index: usize| {
        let mut row_major_index = 0;
        let mut column_stride = 1;
        for (dimension_index, &dimension) in shape.iter().enumerate() {
            let coordinate = (column_major_index / column_stride) % dimension;
            let row_stride = shape[dimension_index + 1..].iter().product::<usize>();
            row_major_index += coordinate * row_stride;
            column_stride *= dimension;
        }
        row_major_index
    };
    if numbers.iter().all(|number| number.is_u64()) {
        return Some(IntegerStorage::U64(
            (0..numbers.len())
                .map(|index| {
                    numbers[row_major_index(index)]
                        .as_u64()
                        .expect("checked unsigned number")
                })
                .collect(),
        ));
    }
    if numbers.iter().all(|number| number.is_i64()) {
        return Some(IntegerStorage::I64(
            (0..numbers.len())
                .map(|index| {
                    numbers[row_major_index(index)]
                        .as_i64()
                        .expect("checked signed number")
                })
                .collect(),
        ));
    }
    None
}

pub(in crate::builtins::fea) fn serializable_to_object_value<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: runmat_types::StaticClassIdentity,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<ObjectInstance> {
    ensure_fea_classes_registered();
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let converted = value_from_json_preserving_integer_kinds(builtin, error, &json)?;
    let mut object = ObjectInstance::new(class_name);
    if let Value::Struct(fields) = converted {
        object.properties = fields.fields.into_iter().collect();
    } else {
        object.properties.insert("value".to_string(), converted);
    }
    if let Some(property) = hidden_json_property {
        object
            .properties
            .insert(property.to_string(), Value::String(json.to_string()));
    }
    Ok(object)
}
