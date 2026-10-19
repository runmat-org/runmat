use runmat_value::{CellArray, ObjectInstance, Value};

const KEYS: &str = "Keys";
const VALUES: &str = "Values";

pub(super) fn from_pairs(pairs: Vec<(String, String)>) -> Result<Value, String> {
    crate::builtins::table::ensure_table_class_registered();
    let mut object = ObjectInstance::new(runmat_types::standard::DICTIONARY.to_string());
    let count = pairs.len();
    let keys = pairs
        .iter()
        .map(|(key, _)| Value::String(key.clone()))
        .collect();
    let values = pairs
        .into_iter()
        .map(|(_, value)| Value::String(value))
        .collect();
    object
        .properties
        .insert(KEYS.into(), Value::Cell(CellArray::new(keys, 1, count)?));
    object.properties.insert(
        VALUES.into(),
        Value::Cell(CellArray::new(values, 1, count)?),
    );
    Ok(Value::Object(object))
}

pub(super) fn entries(object: &ObjectInstance) -> Result<Vec<(Value, Value)>, ()> {
    if !object.is_class(runmat_types::standard::DICTIONARY) {
        return Err(());
    }
    let Some(Value::Cell(keys)) = object.properties.get(KEYS) else {
        return Err(());
    };
    let Some(Value::Cell(values)) = object.properties.get(VALUES) else {
        return Err(());
    };
    if keys.data.len() != values.data.len() {
        return Err(());
    }
    Ok(keys
        .data
        .iter()
        .cloned()
        .zip(values.data.iter().cloned())
        .collect())
}
