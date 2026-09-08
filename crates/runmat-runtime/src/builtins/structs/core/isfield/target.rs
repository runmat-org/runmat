use runmat_value::Value;
use std::collections::HashSet;

pub(super) fn common_fields(value: &Value) -> Option<HashSet<&str>> {
    match value {
        Value::Struct(structure) => Some(structure.fields.keys().map(String::as_str).collect()),
        Value::Cell(array) if !array.data.is_empty() => {
            let mut structures = array.data.iter().map(|value| match value {
                Value::Struct(structure) => Some(structure),
                _ => None,
            });
            let first = structures.next().flatten()?;
            let mut fields: HashSet<&str> = first.fields.keys().map(String::as_str).collect();
            for structure in structures {
                let structure = structure?;
                fields.retain(|name| structure.fields.contains_key(*name));
            }
            Some(fields)
        }
        _ => None,
    }
}
