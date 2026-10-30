use super::*;
use runmat_value::NumericDType;

pub(super) async fn prepare_keys(
    value: Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<Vec<KeyCandidate>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    let flattened = flatten_keys(&host, key_type, builtin).await?;
    let mut out = Vec::with_capacity(flattened.len());
    for raw_key in flattened {
        let (normalized, canonical) = canonicalize_key(raw_key, key_type, builtin)?;
        out.push(KeyCandidate {
            normalized,
            canonical,
        });
    }
    Ok(out)
}

pub(super) async fn prepare_values(
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    flatten_values(&host, builtin).await
}

pub(super) async fn flatten_keys(
    value: &Value,
    _key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data().len());
            for ptr in cell.data() {
                let element = ptr;
                if matches!(element, Value::Cell(_)) {
                    return Err(map_error(
                        "containers.Map: nested cell arrays are not supported for keys",
                        builtin,
                    ));
                }
                out.push(
                    gather_if_needed_async(element)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data()
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca, builtin)?),
        Value::LogicalArray(_) => Err(map_error(
            "containers.Map: logical arrays are not supported as Map keys",
            builtin,
        )),
        Value::Tensor(t) => {
            if !t.shape().is_empty()
                && tensor::tensor_element_len(t) != 1
                && !is_vector_shape(t.shape())
            {
                return Err(map_error(
                    "containers.Map: numeric keys must be scalar or vector shaped",
                    builtin,
                ));
            }
            Ok(tensor_elements_to_values(t))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => {
            Ok(vec![value.clone()])
        }
        Value::GpuTensor(_) => Err(map_error(
            "containers.Map: GPU keys must be gathered to the host before construction",
            builtin,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported key container {other:?}"),
            builtin,
        )),
    }
}

pub(super) async fn flatten_values(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data().len());
            for ptr in cell.data() {
                out.push(
                    gather_if_needed_async(ptr)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data()
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca, builtin)?),
        Value::LogicalArray(arr) => Ok(arr.data.iter().map(|&b| Value::Bool(b != 0)).collect()),
        Value::Tensor(t) => {
            if !t.shape().is_empty()
                && !is_vector_shape(t.shape())
                && tensor::tensor_element_len(t) != 1
            {
                return Err(map_error(
                    "containers.Map: numeric values must be scalar or vector shaped",
                    builtin,
                ));
            }
            Ok(tensor_elements_to_values(t))
        }
        _ => Ok(vec![value.clone()]),
    }
}

pub(super) fn char_array_rows(ca: &CharArray, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    if ca.rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let start = row * ca.cols;
        let text = runmat_value::decode_utf16_code_units(&ca.data()[start..start + ca.cols])
            .map_err(|source| map_error(format!("containers.Map: {source}"), builtin))?;
        let chars: Vec<char> = text.chars().collect();
        let array = CharArray::from_unicode_scalars(chars, 1, text.encode_utf16().count())
            .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
        out.push(Value::CharArray(array));
    }
    Ok(out)
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

pub(super) fn tensor_elements_to_values(tensor: &Tensor) -> Vec<Value> {
    if let Some(storage) = tensor.integer_storage() {
        return storage.exact_values().into_iter().map(Value::Int).collect();
    }
    match tensor.numeric_dtype() {
        NumericDType::F64 => tensor
            .as_f64_slice()
            .expect("double tensor has native double storage")
            .iter()
            .copied()
            .map(Value::Num)
            .collect(),
        NumericDType::F32 => (0..tensor.len())
            .map(|index| {
                let value = match tensor.numeric_value_at(index) {
                    Some(runmat_value::NumericScalar::F32(value)) => value,
                    _ => unreachable!("single tensor has native single storage"),
                };
                Value::Tensor(
                    Tensor::from_f32(vec![value], vec![1, 1])
                        .expect("single scalar tensor construction"),
                )
            })
            .collect(),
        _ => unreachable!("integer tensor returned through exact key/value path"),
    }
}

pub(super) fn canonicalize_key(
    value: Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<(NormalizedKey, Value)> {
    let normalized = normalize_key(&value, key_type, builtin)?;
    let canonical = match key_type {
        KeyType::Char => Value::CharArray(char_array_from_value(&value, builtin)?),
        KeyType::Double => Value::Num(numeric_from_value(
            &value,
            "containers.Map: keys must be numeric scalars",
            builtin,
        )?),
        KeyType::Single => Value::Tensor(
            Tensor::from_f32(
                vec![numeric_from_value(
                    &value,
                    "containers.Map: keys must be numeric scalars",
                    builtin,
                )? as f32],
                vec![1, 1],
            )
            .map_err(|err| map_error(format!("containers.Map: {err}"), builtin))?,
        ),
        KeyType::Int32 => Value::Int(IntValue::I32(integer_from_value(
            &value,
            i32::MIN as i64,
            i32::MAX as i64,
            "containers.Map: int32 keys must be integers",
            builtin,
        )? as i32)),
        KeyType::UInt32 => Value::Int(IntValue::U32(unsigned_from_value(
            &value,
            u32::MAX as u64,
            "containers.Map: uint32 keys must be unsigned integers",
            builtin,
        )? as u32)),
        KeyType::Int64 => Value::Int(IntValue::I64(integer_from_value(
            &value,
            i64::MIN,
            i64::MAX,
            "containers.Map: int64 keys must be integers",
            builtin,
        )?)),
        KeyType::UInt64 => Value::Int(IntValue::U64(unsigned_from_value(
            &value,
            u64::MAX,
            "containers.Map: uint64 keys must be unsigned integers",
            builtin,
        )?)),
    };
    Ok((normalized, canonical))
}

pub(super) fn normalize_key(
    value: &Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<NormalizedKey> {
    if !value_matches_key_type(value, key_type) {
        return Err(map_error(
            format!(
                "containers.Map: key class must match the map KeyType '{}'",
                key_type.matlab_name()
            ),
            builtin,
        ));
    }
    match key_type {
        KeyType::Char => {
            let text =
                string_from_value(value, "containers.Map: keys must be text scalars", builtin)?;
            Ok(NormalizedKey::String(text))
        }
        KeyType::Double | KeyType::Single => {
            let numeric = numeric_from_value(
                value,
                "containers.Map: keys must be numeric scalars",
                builtin,
            )?;
            if !numeric.is_finite() {
                return Err(map_error(
                    "containers.Map: keys must be finite numeric scalars",
                    builtin,
                ));
            }
            let numeric = if key_type == KeyType::Single {
                f64::from(numeric as f32)
            } else {
                numeric
            };
            let canonical = if numeric == 0.0 { 0.0 } else { numeric };
            Ok(NormalizedKey::Float(canonical.to_bits()))
        }
        KeyType::Int32 | KeyType::Int64 => {
            let bounds = if key_type == KeyType::Int32 {
                (i32::MIN as i64, i32::MAX as i64)
            } else {
                (i64::MIN, i64::MAX)
            };
            let value = integer_from_value(
                value,
                bounds.0,
                bounds.1,
                "containers.Map: integer keys must be whole numbers",
                builtin,
            )?;
            Ok(NormalizedKey::Int(value))
        }
        KeyType::UInt32 | KeyType::UInt64 => {
            let limit = if key_type == KeyType::UInt32 {
                u32::MAX as u64
            } else {
                u64::MAX
            };
            let value = unsigned_from_value(
                value,
                limit,
                "containers.Map: unsigned keys must be non-negative integers",
                builtin,
            )?;
            Ok(NormalizedKey::UInt(value))
        }
    }
}

pub(super) fn value_matches_key_type(value: &Value, key_type: KeyType) -> bool {
    match key_type {
        KeyType::Char => {
            matches!(
                value,
                Value::CharArray(chars) if chars.rows == 1
            ) || matches!(value, Value::String(_))
                || matches!(value, Value::StringArray(strings) if strings.data().len() == 1)
        }
        KeyType::Double => {
            matches!(value, Value::Num(_))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64)
        }
        KeyType::Single => {
            matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F32)
        }
        KeyType::Int32 => {
            matches!(value, Value::Int(IntValue::I32(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::I32)
        }
        KeyType::UInt32 => {
            matches!(value, Value::Int(IntValue::U32(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::U32)
        }
        KeyType::Int64 => {
            matches!(value, Value::Int(IntValue::I64(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::I64)
        }
        KeyType::UInt64 => {
            matches!(value, Value::Int(IntValue::U64(_)))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::U64)
        }
    }
}

pub(super) fn string_from_value(
    value: &Value,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.to_string()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => runmat_value::decode_utf16_code_units(ca.data())
            .map_err(|source| map_error(format!("containers.Map: {source}"), builtin)),
        _ => Err(map_error(context, builtin)),
    }
}

pub(super) fn char_array_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<CharArray> {
    match value {
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.clone()),
        Value::String(runmat_value::StringElement::Text(s)) => {
            let chars: Vec<char> = s.chars().collect();
            CharArray::from_unicode_scalars(chars, 1, s.encode_utf16().count())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            let chars: Vec<char> = sa.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .chars()
                .collect();
            CharArray::from_unicode_scalars(
                chars,
                1,
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .encode_utf16()
                    .count(),
            )
            .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
        }
        _ => Err(map_error(
            "containers.Map: keys must be character vectors",
            builtin,
        )),
    }
}

pub(super) fn char_array_value(text: &str, builtin: &'static str) -> BuiltinResult<Value> {
    let chars: Vec<char> = text.chars().collect();
    CharArray::from_unicode_scalars(chars, 1, text.encode_utf16().count())
        .map(Value::CharArray)
        .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
}
