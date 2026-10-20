use super::*;

pub fn table_from_columns(names: Vec<String>, columns: Vec<Value>) -> BuiltinResult<Value> {
    table_from_columns_with_properties(names, columns, None)
}

pub fn table_replace_variables_like(
    source: &ObjectInstance,
    variables: StructValue,
) -> BuiltinResult<Value> {
    let names = table_variable_names_from_object(source)?;
    if variables.fields.len() != names.len()
        || names
            .iter()
            .any(|name| !variables.fields.contains_key(name))
    {
        return Err(invalid_variable(
            "table: replacement variables must preserve variable names and order",
        ));
    }
    let columns = names
        .iter()
        .filter_map(|name| variables.fields.get(name).cloned())
        .collect::<Vec<_>>();
    let height = validate_column_heights(&names, &columns)?;
    let source_height = table_height(source)?;
    if height != source_height {
        return Err(invalid_variable(format!(
            "table: replacement variables have {height} rows but expected {source_height}"
        )));
    }
    let mut object = source.clone();
    crate::execution::object::replacement::replace_property(
        None,
        &mut object,
        TABLE_VARIABLES_FIELD,
        Value::Struct(variables),
        |_, source| invalid_variable(format!("table: replacement admission failed: {source}")),
        invalid_variable,
    )?;
    Ok(Value::Object(object))
}

/// Retain each completed variable through validation and replacement Object
/// publication. The input batch and names are constructed in the same table
/// traversal; no raw column may escape an awaited producer before admission.
pub(crate) fn table_replace_variables_like_owned(
    source: &ObjectInstance,
    names: Vec<String>,
    columns: crate::execution::RootedValues,
) -> BuiltinResult<crate::execution::RootedValue> {
    columns.with_owned_output_storage(|values| {
        if names.len() != values.len() {
            return Err(crate::RuntimeError::new(
                "table: owned column count differs from variable names",
            ));
        }
        let mut variables = runmat_value::UnboundedStructValueBuilder::new();
        let values = values.try_into_owned().map_err(|_| {
            crate::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            )
        })?;
        for (name, value) in names.into_iter().zip(values) {
            variables.insert(name, value);
        }
        let variables = variables.finish();
        let replacement = table_replace_variables_like(source, variables)?;
        crate::execution::RootedValue::new(replacement).map_err(|error| error.into_runtime_error())
    })
}

pub(crate) fn table_from_columns_with_properties(
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    table_from_columns_with_class(TABLE_CLASS, names, columns, row_names)
}

pub(crate) fn table_from_columns_like(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &ObjectInstance,
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
    selected_rows: Option<&[usize]>,
) -> BuiltinResult<Value> {
    let mut out =
        table_from_columns_with_class(source.class_name().clone(), names, columns, row_names)?;
    if source.is_class(TIMETABLE_CLASS) {
        if let Value::Object(object) = &mut out {
            let row_times = if let Some(rows) = selected_rows {
                selected_row_times(runtime, source, rows)?
            } else {
                timetable_row_times(source)?
            };
            set_timetable_row_times(runtime, object, row_times)?;
        }
    }
    Ok(out)
}

pub(in crate::builtins::table) fn table_from_columns_with_class(
    class_name: impl Into<runmat_types::ClassIdentity>,
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    let class_name = class_name.into();
    ensure_table_class_registered();
    if names.len() != columns.len() {
        return Err(invalid_variable(
            "table: number of variable names must match number of variables",
        ));
    }
    validate_unique_names(&names)?;
    let height = validate_column_heights(&names, &columns)?;
    if let Some(row_names) = &row_names {
        if row_names.len() != height {
            return Err(invalid_variable(
                "table: number of row names must match table height",
            ));
        }
    }
    let mut variables = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in names.iter().cloned().zip(columns) {
        variables.insert(name, value);
    }
    let props = default_properties_for_class(class_name.clone(), names, row_names);
    let mut object = runmat_value::UnboundedObjectBuilder::with_property_capacity(class_name, 3);
    object.insert(
        TABLE_VARIABLES_FIELD.to_string(),
        Value::Struct(variables.finish()),
    );
    object.insert(
        TABLE_PROPERTIES_FIELD.to_string(),
        Value::Struct(props.clone()),
    );
    object.insert(PROPERTIES_MEMBER.to_string(), Value::Struct(props));
    Ok(Value::Object(object.finish()))
}

pub(in crate::builtins::table) fn table_from_retained_columns_with_class<'a, I>(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: runmat_types::StaticClassIdentity,
    names: Vec<String>,
    columns: I,
    row_names: Option<Vec<String>>,
) -> BuiltinResult<Value>
where
    I: Iterator<Item = &'a Value> + Clone,
{
    ensure_table_class_registered();
    if names.len() != columns.clone().count() {
        return Err(invalid_variable(
            "table: number of variable names must match number of variables",
        ));
    }
    validate_unique_names(&names)?;
    let height = validate_column_heights_iter(&names, columns.clone())?;
    if row_names.as_ref().is_some_and(|rows| rows.len() != height) {
        return Err(invalid_variable(
            "table: number of row names must match table height",
        ));
    }
    let variables = crate::execution::structure::construct_struct_from_fields_in_optional_context(
        runtime,
        names.iter().map(String::as_str).zip(columns),
    )?;
    // Default property construction retains its existing domain metadata
    // algorithm. Its complete finite-allocation contract is separate from
    // this admitted argument/column generation boundary.
    let properties = Value::Struct(default_properties_for_class(class_name, names, row_names));
    let fields = [
        (TABLE_VARIABLES_FIELD, variables.value()),
        (TABLE_PROPERTIES_FIELD, &properties),
        (PROPERTIES_MEMBER, &properties),
    ];
    let result = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class_name.display_name(),
        fields.into_iter(),
    )?;
    Ok(result.into_value_and_lease().0)
}

pub(super) fn validate_column_heights(names: &[String], columns: &[Value]) -> BuiltinResult<usize> {
    validate_column_heights_iter(names, columns.iter())
}

fn validate_column_heights_iter<'a>(
    names: &[String],
    columns: impl Iterator<Item = &'a Value> + Clone,
) -> BuiltinResult<usize> {
    let Some(first) = columns.clone().next() else {
        return Ok(0);
    };
    let height = value_row_count(first)?;
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
    object.class_name().is(TABLE_CLASS) || object.class_name().is(TIMETABLE_CLASS)
}

pub fn table_variables(object: &ObjectInstance) -> BuiltinResult<StructValue> {
    Ok(table_variables_ref(object)?.cloned().unwrap_or_default())
}

pub(crate) fn table_variables_ref(object: &ObjectInstance) -> BuiltinResult<Option<&StructValue>> {
    match object.properties().get(TABLE_VARIABLES_FIELD) {
        Some(Value::Struct(st)) => Ok(Some(st)),
        Some(other) => Err(invalid_variable(format!(
            "table: invalid internal variable storage {other:?}"
        ))),
        None => Ok(None),
    }
}

pub fn table_variable_names_from_object(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    let variables = table_variables_ref(object)?;
    let properties = object
        .properties()
        .get(TABLE_PROPERTIES_FIELD)
        .or_else(|| object.properties().get(PROPERTIES_MEMBER));
    let Some(properties) = properties else {
        // Only legacy table objects with no Properties payload lack an order
        // receipt. Their storage-key order remains the supported fallback.
        return Ok(variables
            .map(|variables| variables.fields.keys().map(ToOwned::to_owned).collect())
            .unwrap_or_default());
    };
    let Value::Struct(properties) = properties else {
        return Err(invalid_variable("table: invalid Properties storage"));
    };
    let source = properties
        .fields
        .get(VARIABLE_NAMES)
        .ok_or_else(|| invalid_variable("table: Properties.VariableNames is missing"))?;
    let names = match source {
        Value::StringArray(array) if array.data().is_empty() => Vec::new(),
        _ => variable_name_list(source)?,
    };
    let actual = variables.ok_or_else(|| invalid_variable("table: variable storage is missing"))?;
    if names.len() != actual.fields.len()
        || names.iter().any(|name| !actual.fields.contains_key(name))
    {
        return Err(invalid_variable(
            "table: Properties.VariableNames disagrees with variable storage",
        ));
    }
    Ok(names)
}

pub fn table_height(object: &ObjectInstance) -> BuiltinResult<usize> {
    let variables = table_variables(object)?;
    let height = match variables.fields.values().next() {
        Some(value) => value_row_count(value),
        None => Ok(0),
    }?;
    Ok(height)
}

pub fn table_width(object: &ObjectInstance) -> BuiltinResult<usize> {
    table_variables(object).map(|vars| vars.fields.len())
}
