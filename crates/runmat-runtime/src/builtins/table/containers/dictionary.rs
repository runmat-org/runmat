use super::*;
use runmat_builtins::{IntValue, IntegerStorage, NumericScalar};

pub(in crate::builtins::table) fn dictionary_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    let (mut keys, mut values, keys_are_cells, values_are_cells) = match args.as_slice() {
        [] => (Vec::new(), Vec::new(), false, false),
        [keys, values] => {
            let key_shape = dictionary_element_shape(keys);
            let value_shape = dictionary_element_shape(values);
            let keys_are_cells = matches!(keys, Value::Cell(_));
            let values_are_cells = matches!(values, Value::Cell(_));
            let keys = value_elements(keys)?;
            let mut values = value_elements(values)?;
            if values.len() == 1 && keys.len() != 1 {
                values.resize(keys.len(), values[0].clone());
            } else if keys.len() != values.len() || key_shape != value_shape {
                return Err(invalid_argument(
                    "dictionary: keys and values must have the same size unless values is scalar",
                ));
            }
            (keys, values, keys_are_cells, values_are_cells)
        }
        _ if args.len().is_multiple_of(2) => {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            let mut keys_are_cells = true;
            let mut values_are_cells = true;
            let mut idx = 0usize;
            while idx < args.len() {
                let key_arg = &args[idx];
                let value_arg = &args[idx + 1];
                keys_are_cells &= matches!(key_arg, Value::Cell(_));
                values_are_cells &= matches!(value_arg, Value::Cell(_));
                let pair_keys = value_elements(key_arg)?;
                let mut pair_values = value_elements(value_arg)?;
                if pair_values.len() == 1 && pair_keys.len() != 1 {
                    pair_values.resize(pair_keys.len(), pair_values[0].clone());
                } else if pair_keys.len() != pair_values.len()
                    || dictionary_element_shape(key_arg) != dictionary_element_shape(value_arg)
                {
                    return Err(invalid_argument(
                        "dictionary: each key/value pair must have the same size unless its value is scalar",
                    ));
                }
                keys.extend(pair_keys);
                values.extend(pair_values);
                idx += 2;
            }
            (keys, values, keys_are_cells, values_are_cells)
        }
        _ => {
            return Err(invalid_argument(
                "dictionary: expected keys and values, or key/value pairs",
            ))
        }
    };
    normalize_dictionary_elements(&mut keys, keys_are_cells)?;
    normalize_dictionary_elements(&mut values, values_are_cells)?;
    let mut unique_keys = Vec::new();
    let mut unique_values = Vec::new();
    for (key, value) in keys.into_iter().zip(values) {
        if let Some(index) = unique_keys
            .iter()
            .position(|existing| dictionary_keys_equal(existing, &key))
        {
            unique_values[index] = value;
        } else {
            unique_keys.push(key);
            unique_values.push(value);
        }
    }
    let keys = unique_keys;
    let values = unique_values;
    let key_count = keys.len();
    let value_count = values.len();
    let mut object = ObjectInstance::new(DICTIONARY_CLASS.to_string());
    object.properties.insert(
        "Keys".to_string(),
        Value::Cell(CellArray::new(keys, 1, key_count).map_err(invalid_variable)?),
    );
    object.properties.insert(
        "Values".to_string(),
        Value::Cell(CellArray::new(values, 1, value_count).map_err(invalid_variable)?),
    );
    Ok(Value::Object(object))
}

pub(in crate::builtins::table) fn into_dictionary_object(
    value: Value,
    context: &str,
) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(DICTIONARY_CLASS) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected dictionary, got {other:?}"
        ))),
    }
}

pub(in crate::builtins::table) fn dictionary_cells<'a>(
    object: &'a ObjectInstance,
    field: &str,
) -> BuiltinResult<&'a CellArray> {
    match object.properties.get(field) {
        Some(Value::Cell(cell)) => Ok(cell),
        Some(other) => Err(invalid_variable(format!(
            "dictionary: {field} storage must be a cell array, got {other:?}"
        ))),
        None => Err(invalid_variable(format!(
            "dictionary: missing {field} storage"
        ))),
    }
}

pub(in crate::builtins::table) fn dictionary_lookup(
    object: &ObjectInstance,
    payload: &Value,
) -> BuiltinResult<Value> {
    let (query_keys, query_shape) = dictionary_selector_values(payload)?;
    let keys = dictionary_cells(object, "Keys")?;
    let values = dictionary_cells(object, "Values")?;
    let mut out = Vec::with_capacity(query_keys.len());
    let query_keys = convert_queries_to_key_class(keys, query_keys)?;
    for query in &query_keys {
        let idx = keys
            .data
            .iter()
            .position(|key| dictionary_keys_equal(key, query))
            .ok_or_else(|| invalid_index(format!("dictionary: key {query:?} not found")))?;
        out.push(
            values
                .data
                .get(idx)
                .cloned()
                .ok_or_else(|| invalid_index("dictionary: value index out of bounds"))?,
        );
    }
    dictionary_lookup_output(out, query_shape)
}

pub(in crate::builtins::table) fn dictionary_assign(
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let (query_keys, query_shape) = dictionary_selector_values(payload)?;
    let mut keys = dictionary_cells(&object, "Keys")?.clone();
    let mut values = dictionary_cells(&object, "Values")?.clone();
    let query_keys = convert_queries_to_key_class(&keys, query_keys)?;
    if option_value_is_empty(&rhs) {
        for query in query_keys {
            if let Some(index) = keys
                .data
                .iter()
                .position(|key| dictionary_keys_equal(key, &query))
            {
                keys.data.remove(index);
                values.data.remove(index);
            }
        }
    } else {
        let mut rhs_values = value_elements(&rhs)?;
        if rhs_values.len() == 1 && query_keys.len() != 1 {
            rhs_values.resize(query_keys.len(), rhs_values[0].clone());
        } else if rhs_values.len() != query_keys.len() {
            return Err(invalid_index(
                "dictionary: assignment keys and values must have the same size unless values is scalar",
            ));
        } else if rhs_values.len() > 1 && dictionary_element_shape(&rhs) != query_shape {
            return Err(invalid_index(
                "dictionary: non-scalar assignment values must have the same size as the keys",
            ));
        }
        normalize_to_existing_class(&values.data, &mut rhs_values)?;
        for (query, value) in query_keys.into_iter().zip(rhs_values) {
            if let Some(index) = keys
                .data
                .iter()
                .position(|key| dictionary_keys_equal(key, &query))
            {
                values.data[index] = value;
            } else {
                keys.data.push(query);
                values.data.push(value);
            }
        }
    }
    keys.cols = keys.data.len();
    keys.rows = usize::from(!keys.data.is_empty());
    values.cols = values.data.len();
    values.rows = usize::from(!values.data.is_empty());
    object
        .properties
        .insert("Keys".to_string(), Value::Cell(keys));
    object
        .properties
        .insert("Values".to_string(), Value::Cell(values));
    Ok(Value::Object(object))
}

fn dictionary_element_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::LogicalArray(array) => array.shape.clone(),
        Value::StringArray(array) => array.shape.clone(),
        Value::Cell(array) => vec![array.rows, array.cols],
        Value::CharArray(array) => vec![array.rows, 1],
        _ => vec![1, 1],
    }
}

fn normalize_dictionary_elements(values: &mut [Value], cells: bool) -> BuiltinResult<()> {
    if cells || values.is_empty() {
        return Ok(());
    }
    let existing = vec![values[0].clone()];
    normalize_to_existing_class(&existing, values)
}

fn normalize_to_existing_class(existing: &[Value], values: &mut [Value]) -> BuiltinResult<()> {
    let Some(first) = existing.first() else {
        return Ok(());
    };
    match first {
        Value::Int(first) => {
            let target =
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::from_int_value(
                    first,
                );
            for value in values {
                *value = match value {
                    Value::Int(value) => Value::Int(target.cast_int(value)),
                    Value::Num(value) => Value::Int(target.cast_scalar(*value)),
                    other => {
                        return Err(invalid_argument(format!(
                            "dictionary: value {other:?} is not convertible to the configured integer type"
                        )))
                    }
                };
            }
        }
        Value::Num(_) => {
            for value in values {
                *value = match value {
                    Value::Num(value) => Value::Num(*value),
                    Value::Int(value) => Value::Num(value.to_f64()),
                    other => {
                        return Err(invalid_argument(format!(
                            "dictionary: value {other:?} is not convertible to double"
                        )))
                    }
                };
            }
        }
        Value::String(_) => {
            for value in values {
                *value = match value {
                    Value::String(value) => Value::String(value.clone()),
                    Value::CharArray(array) if array.rows == 1 => {
                        Value::String(array.data.iter().collect())
                    }
                    other => {
                        return Err(invalid_argument(format!(
                            "dictionary: value {other:?} is not convertible to string"
                        )))
                    }
                };
            }
        }
        _ if values
            .iter()
            .all(|value| std::mem::discriminant(value) == std::mem::discriminant(first)) => {}
        _ => {
            return Err(invalid_argument(
                "dictionary: entries must use one compatible configured type",
            ))
        }
    }
    Ok(())
}

fn convert_queries_to_key_class(
    keys: &CellArray,
    mut queries: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    normalize_to_existing_class(&keys.data, &mut queries)?;
    Ok(queries)
}

fn dictionary_selector_values(payload: &Value) -> BuiltinResult<(Vec<Value>, Vec<usize>)> {
    let selectors = selector_values(payload)?;
    let candidate_shape = if selectors.len() == 1 {
        dictionary_element_shape(&selectors[0])
    } else {
        dictionary_element_shape(payload)
    };
    let mut values = Vec::new();
    for selector in selectors {
        values.extend(value_elements(&selector)?);
    }
    let shape = if candidate_shape.iter().product::<usize>() == values.len() {
        candidate_shape
    } else {
        vec![1, values.len()]
    };
    Ok((values, shape))
}

fn dictionary_lookup_output(mut values: Vec<Value>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if values.len() == 1 {
        return Ok(values.remove(0));
    }
    let Some(first) = values.first().cloned() else {
        return crate::make_cell_with_shape(values, shape).map_err(invalid_variable);
    };
    match &first {
        Value::Int(first)
            if values.iter().all(
                |value| matches!(value, Value::Int(value) if std::mem::discriminant(value) == std::mem::discriminant(first)),
            ) =>
        {
            let storage = integer_values_to_storage(first, values)?;
            Tensor::new_integer(storage, shape)
                .map(Value::Tensor)
                .map_err(invalid_variable)
        }
        Value::Num(_) if values.iter().all(|value| matches!(value, Value::Num(_))) => {
            let data = values
                .into_iter()
                .map(|value| match value {
                    Value::Num(value) => value,
                    _ => unreachable!("validated double dictionary values"),
                })
                .collect();
            Tensor::new(data, shape)
                .map(Value::Tensor)
                .map_err(invalid_variable)
        }
        Value::Bool(_) if values.iter().all(|value| matches!(value, Value::Bool(_))) => {
            let data = values
                .into_iter()
                .map(|value| match value {
                    Value::Bool(value) => u8::from(value),
                    _ => unreachable!("validated logical dictionary values"),
                })
                .collect();
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(invalid_variable)
        }
        Value::String(_) if values.iter().all(|value| matches!(value, Value::String(_))) => {
            let data = values
                .into_iter()
                .map(|value| match value {
                    Value::String(value) => value,
                    _ => unreachable!("validated string dictionary values"),
                })
                .collect();
            StringArray::new(data, shape)
                .map(Value::StringArray)
                .map_err(invalid_variable)
        }
        _ => crate::make_cell_with_shape(values, shape).map_err(invalid_variable),
    }
}

fn integer_values_to_storage(
    first: &IntValue,
    values: Vec<Value>,
) -> BuiltinResult<IntegerStorage> {
    macro_rules! collect_variant {
        ($variant:ident) => {{
            let data = values
                .into_iter()
                .map(|value| match value {
                    Value::Int(IntValue::$variant(value)) => Ok(value),
                    other => Err(invalid_variable(format!(
                        "dictionary: inconsistent integer value class {other:?}"
                    ))),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            IntegerStorage::$variant(data)
        }};
    }
    Ok(match first {
        IntValue::I8(_) => collect_variant!(I8),
        IntValue::I16(_) => collect_variant!(I16),
        IntValue::I32(_) => collect_variant!(I32),
        IntValue::I64(_) => collect_variant!(I64),
        IntValue::U8(_) => collect_variant!(U8),
        IntValue::U16(_) => collect_variant!(U16),
        IntValue::U32(_) => collect_variant!(U32),
        IntValue::U64(_) => collect_variant!(U64),
    })
}

pub(in crate::builtins::table) fn dictionary_keys_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::CharArray(a), Value::CharArray(b)) => a.data == b.data,
        (Value::String(a), Value::CharArray(b)) | (Value::CharArray(b), Value::String(a))
            if b.rows == 1 =>
        {
            b.data.iter().collect::<String>() == *a
        }
        (Value::Num(a), Value::Num(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => left == right,
    }
}

pub(in crate::builtins::table) fn value_elements(value: &Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => Ok(cell.data.clone()),
        Value::StringArray(array) => Ok(array.data.iter().cloned().map(Value::String).collect()),
        Value::Tensor(tensor) => Ok((0..tensor.len())
            .map(|index| {
                numeric_scalar_value(
                    tensor
                        .numeric_value_at(index)
                        .expect("validated dictionary tensor storage"),
                )
            })
            .collect()),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| Value::Bool(*flag != 0))
            .collect()),
        Value::CharArray(array) => Ok(char_rows(array).into_iter().map(Value::String).collect()),
        other => Ok(vec![other.clone()]),
    }
}

fn numeric_scalar_value(value: NumericScalar) -> Value {
    match value {
        NumericScalar::F64(value) => Value::Num(value),
        NumericScalar::F32(value) => Value::Num(f64::from(value)),
        value => Value::Int(
            value
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage};

    #[test]
    fn dictionary_exact_integer_keys_do_not_saturate_wide_uint64_values() {
        let first = Value::Int(IntValue::U64(i64::MAX as u64 + 1));
        let second = Value::Int(IntValue::U64(i64::MAX as u64 + 2));
        let repeated = Value::Int(IntValue::U64(i64::MAX as u64 + 1));

        assert!(dictionary_keys_equal(&first, &repeated));
        assert!(!dictionary_keys_equal(&first, &second));
    }

    #[test]
    fn dictionary_value_elements_preserve_exact_integer_tensor_keys() {
        let wide = i64::MAX as u64 + 1;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
            .expect("integer tensor");

        let elements = value_elements(&Value::Tensor(tensor)).expect("elements");

        assert_eq!(
            elements,
            vec![
                Value::Int(IntValue::U64(wide)),
                Value::Int(IntValue::U64(u64::MAX))
            ]
        );
    }

    #[test]
    fn dictionary_lookup_distinguishes_wide_uint64_tensor_keys() {
        let first = i64::MAX as u64 + 1;
        let second = i64::MAX as u64 + 2;
        let keys = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![first, second]), vec![1, 2])
                .expect("integer keys"),
        );
        let values = Value::Cell(
            CellArray::new(
                vec![
                    Value::String("first".to_string()),
                    Value::String("second".to_string()),
                ],
                1,
                2,
            )
            .expect("values"),
        );
        let dictionary = dictionary_from_args(vec![keys, values]).expect("dictionary");
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary object");
        };

        let found = dictionary_lookup(&object, &Value::Int(IntValue::U64(second))).expect("lookup");

        assert_eq!(found, Value::String("second".to_string()));
    }

    #[test]
    fn dictionary_scalar_values_expand_and_duplicate_keys_keep_last_in_place() {
        let keys = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 1]), vec![1, 3]).unwrap(),
        );
        let dictionary = dictionary_from_args(vec![keys, Value::Int(IntValue::I8(7))]).unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let keys = dictionary_cells(&object, "Keys").unwrap();
        let values = dictionary_cells(&object, "Values").unwrap();
        assert_eq!(
            keys.data,
            vec![Value::Int(IntValue::U16(1)), Value::Int(IntValue::U16(2))]
        );
        assert_eq!(
            values.data,
            vec![Value::Int(IntValue::I8(7)), Value::Int(IntValue::I8(7))]
        );
    }

    #[test]
    fn dictionary_array_pairs_expand_and_convert_to_first_integer_class() {
        let dictionary = dictionary_from_args(vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
            Value::Int(IntValue::U8(3)),
            Value::Int(IntValue::I32(30)),
        ])
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        assert_eq!(
            dictionary_cells(&object, "Keys").unwrap().data,
            vec![
                Value::Int(IntValue::I16(1)),
                Value::Int(IntValue::I16(2)),
                Value::Int(IntValue::I16(3))
            ]
        );
        assert_eq!(
            dictionary_lookup(&object, &Value::Int(IntValue::U64(3))).unwrap(),
            Value::Int(IntValue::U64(30))
        );
    }

    #[test]
    fn dictionary_vector_assignment_and_removal_preserve_exact_integer_keys() {
        let dictionary = dictionary_from_args(vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
        ])
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let assigned = dictionary_assign(
            object,
            &Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![2, 3]), vec![1, 2]).unwrap(),
            ),
            Value::Int(IntValue::I32(99)),
        )
        .unwrap();
        let Value::Object(object) = assigned else {
            panic!("expected dictionary");
        };
        assert_eq!(
            dictionary_lookup(&object, &Value::Int(IntValue::U64(3))).unwrap(),
            Value::Int(IntValue::I16(99))
        );
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let removed = dictionary_assign(object, &Value::Int(IntValue::U64(2)), empty).unwrap();
        let Value::Object(object) = removed else {
            panic!("expected dictionary");
        };
        assert!(dictionary_lookup(&object, &Value::Int(IntValue::U64(2))).is_err());
    }

    #[test]
    fn dictionary_vector_lookup_preserves_integer_class_and_query_shape() {
        let dictionary = dictionary_from_args(vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
        ])
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let query =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![2, 1]), vec![2, 1]).unwrap());
        let Value::Tensor(found) = dictionary_lookup(&object, &query).unwrap() else {
            panic!("expected configured integer tensor output");
        };
        assert_eq!(found.shape, vec![2, 1]);
        assert_eq!(
            found.integer_storage(),
            Some(&IntegerStorage::I16(vec![20, 10]))
        );
    }

    #[test]
    fn dictionary_vector_assignment_requires_matching_non_scalar_shape() {
        let dictionary = dictionary_from_args(vec![
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).unwrap()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
        ])
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let keys =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap());
        let values = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![30, 40]), vec![1, 2]).unwrap(),
        );
        assert!(dictionary_assign(object, &keys, values).is_err());
    }
}
