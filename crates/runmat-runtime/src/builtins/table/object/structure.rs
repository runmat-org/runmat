use super::*;

pub fn table_from_columns(names: Vec<String>, columns: Vec<Value>) -> BuiltinResult<Value> {
    table_from_columns_with_properties(names, columns, None)
}

pub(crate) fn table_from_columns_with_properties(
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    table_from_columns_with_class(TABLE_CLASS, names, columns, row_names)
}

pub(crate) fn table_from_columns_like(
    source: &ObjectInstance,
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
    selected_rows: Option<&[usize]>,
) -> BuiltinResult<Value> {
    let mut out =
        table_from_columns_with_class(source.class_name.as_str(), names, columns, row_names)?;
    if source.is_class(TIMETABLE_CLASS) {
        if let Value::Object(object) = &mut out {
            let row_times = if let Some(rows) = selected_rows {
                selected_row_times(source, rows)?
            } else {
                timetable_row_times(source)?
            };
            set_timetable_row_times(object, row_times)?;
        }
    }
    Ok(out)
}

pub(in crate::builtins::table) fn table_from_columns_with_class(
    class_name: &str,
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if names.len() != columns.len() {
        return Err(invalid_variable(
            "table: number of variable names must match number of variables",
        ));
    }
    let names = make_unique_names(names);
    let height = validate_column_heights(&names, &columns)?;
    if let Some(row_names) = &row_names {
        if row_names.len() != height {
            return Err(invalid_variable(
                "table: number of row names must match table height",
            ));
        }
    }
    let mut variables = StructValue::new();
    for (name, value) in names.iter().cloned().zip(columns) {
        variables.insert(name, value);
    }
    let props = default_properties_for_class(class_name, names, row_names);
    let mut object = ObjectInstance::new(class_name.to_string());
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    object.properties.insert(
        TABLE_PROPERTIES_FIELD.to_string(),
        Value::Struct(props.clone()),
    );
    object
        .properties
        .insert(PROPERTIES_MEMBER.to_string(), Value::Struct(props));
    Ok(Value::Object(object))
}

pub(super) fn validate_column_heights(names: &[String], columns: &[Value]) -> BuiltinResult<usize> {
    if columns.is_empty() {
        return Ok(0);
    }
    let height = value_row_count(&columns[0])?;
    for (name, value) in names.iter().zip(columns) {
        let rows = value_row_count(value)?;
        if rows != height {
            return Err(invalid_variable(format!(
                "table: variable '{name}' has {rows} rows but expected {height}"
            )));
        }
    }
    Ok(height)
}

pub fn is_table_value(value: &Value) -> bool {
    table_object(value).is_some()
}

pub fn is_tabular_object(object: &ObjectInstance) -> bool {
    is_tabular_class(object)
}

pub(in crate::builtins::table) fn table_object(value: &Value) -> Option<&ObjectInstance> {
    match value {
        Value::Object(object) if is_tabular_class(object) => Some(object),
        _ => None,
    }
}

pub(in crate::builtins::table) fn into_table_object(
    value: Value,
    context: &str,
) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if is_tabular_class(&object) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected table, got {other:?}"
        ))),
    }
}

pub(in crate::builtins::table) fn into_timetable_object(
    value: Value,
    context: &str,
) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TIMETABLE_CLASS) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected timetable, got {other:?}"
        ))),
    }
}

pub(in crate::builtins::table) fn is_tabular_class(object: &ObjectInstance) -> bool {
    object.is_class(TABLE_CLASS) || object.is_class(TIMETABLE_CLASS)
}

pub fn table_variables(object: &ObjectInstance) -> BuiltinResult<StructValue> {
    match object.properties.get(TABLE_VARIABLES_FIELD) {
        Some(Value::Struct(st)) => Ok(st.clone()),
        Some(other) => Err(invalid_variable(format!(
            "table: invalid internal variable storage {other:?}"
        ))),
        None => Ok(StructValue::new()),
    }
}

pub fn table_variable_names_from_object(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    let variables = table_variables(object)?;
    Ok(variables.fields.keys().cloned().collect())
}

pub fn table_height(object: &ObjectInstance) -> BuiltinResult<usize> {
    let variables = table_variables(object)?;
    match variables.fields.values().next() {
        Some(value) => value_row_count(value),
        None => Ok(0),
    }
}

pub fn table_width(object: &ObjectInstance) -> BuiltinResult<usize> {
    table_variables(object).map(|vars| vars.fields.len())
}
