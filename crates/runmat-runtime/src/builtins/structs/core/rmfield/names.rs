use crate::builtins::structs::field_name;
use runmat_value::{CellArray, StringArray, Value};
use std::collections::HashSet;

pub(super) struct FieldNames(Vec<String>);

impl FieldNames {
    pub(super) fn as_slice(&self) -> &[String] {
        &self.0
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

pub(super) fn parse(first: &Value, rest: &[Value]) -> crate::BuiltinResult<FieldNames> {
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::RMFIELD_VARIADIC_EXTENSION,
            super::BUILTIN_NAME,
        )?;
    }
    let mut names = collect(first, None)?;
    for value in rest {
        names.extend(collect(value, None)?);
    }
    Ok(FieldNames(deduplicate(names)))
}

fn collect(value: &Value, context: Option<&str>) -> crate::BuiltinResult<Vec<String>> {
    match value {
        Value::StringArray(array) if array.data.len() != 1 => string_array(array),
        Value::Cell(array) => cell_array(array),
        _ => scalar(value, context).map(|name| vec![name]),
    }
}

fn string_array(array: &StringArray) -> crate::BuiltinResult<Vec<String>> {
    array
        .data
        .iter()
        .enumerate()
        .map(|(index, name)| {
            require_nonempty(
                name.clone(),
                Some(format!("string array element {}", index + 1)),
            )
        })
        .collect()
}

fn cell_array(array: &CellArray) -> crate::BuiltinResult<Vec<String>> {
    array
        .iter_column_major()
        .enumerate()
        .map(|(index, value)| scalar(value, Some(&format!("cell element {}", index + 1))))
        .collect()
}

fn scalar(value: &Value, context: Option<&str>) -> crate::BuiltinResult<String> {
    let name = field_name::decode(value).map_err(|_| super::error::field_name_type(context))?;
    require_nonempty(name, context.map(str::to_string))
}

fn require_nonempty(name: String, context: Option<String>) -> crate::BuiltinResult<String> {
    if name.is_empty() {
        return Err(super::error::empty_field_name(context.as_deref()));
    }
    Ok(name)
}

fn deduplicate(names: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}
