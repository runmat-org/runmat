use super::*;

pub(in crate::builtins::table) fn dictionary_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    let (keys, values) = match args.as_slice() {
        [] => (Vec::new(), Vec::new()),
        [keys, values] => {
            let keys = value_elements(keys)?;
            let values = value_elements(values)?;
            if keys.len() != values.len() {
                return Err(invalid_argument(
                    "dictionary: keys and values must have the same number of elements",
                ));
            }
            (keys, values)
        }
        _ if args.len().is_multiple_of(2) => {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            let mut idx = 0usize;
            while idx < args.len() {
                keys.push(args[idx].clone());
                values.push(args[idx + 1].clone());
                idx += 2;
            }
            (keys, values)
        }
        _ => {
            return Err(invalid_argument(
                "dictionary: expected keys and values, or key/value pairs",
            ))
        }
    };
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
    let query_keys = selector_values(payload)?;
    let keys = dictionary_cells(object, "Keys")?;
    let values = dictionary_cells(object, "Values")?;
    let mut out = Vec::with_capacity(query_keys.len());
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
    if out.len() == 1 {
        Ok(out.remove(0))
    } else {
        let len = out.len();
        CellArray::new(out, 1, len)
            .map(Value::Cell)
            .map_err(invalid_variable)
    }
}

pub(in crate::builtins::table) fn dictionary_assign(
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let query_keys = selector_values(payload)?;
    if query_keys.len() != 1 {
        return Err(invalid_index(
            "dictionary: assignment currently expects one key",
        ));
    }
    let mut keys = dictionary_cells(&object, "Keys")?.clone();
    let mut values = dictionary_cells(&object, "Values")?.clone();
    if let Some(idx) = keys
        .data
        .iter()
        .position(|key| dictionary_keys_equal(key, &query_keys[0]))
    {
        values.data[idx] = rhs;
    } else {
        keys.data.push(query_keys[0].clone());
        values.data.push(rhs);
        keys.cols = keys.data.len();
        keys.rows = usize::from(!keys.data.is_empty());
        values.cols = values.data.len();
        values.rows = usize::from(!values.data.is_empty());
    }
    object
        .properties
        .insert("Keys".to_string(), Value::Cell(keys));
    object
        .properties
        .insert("Values".to_string(), Value::Cell(values));
    Ok(Value::Object(object))
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
        (Value::Int(a), Value::Int(b)) => a.to_i64() == b.to_i64(),
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => left == right,
    }
}

pub(in crate::builtins::table) fn value_elements(value: &Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => Ok(cell.data.clone()),
        Value::StringArray(array) => Ok(array.data.iter().cloned().map(Value::String).collect()),
        Value::Tensor(tensor) => Ok(tensor.data.iter().copied().map(Value::Num).collect()),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| Value::Bool(*flag != 0))
            .collect()),
        Value::CharArray(array) => Ok(char_rows(array).into_iter().map(Value::String).collect()),
        other => Ok(vec![other.clone()]),
    }
}
