use super::selectors::parse_row_selector_for_object;
use super::*;

pub(in crate::builtins::table) fn table_member_get(
    object: &ObjectInstance,
    payload: &Value,
) -> BuiltinResult<Value> {
    let name = scalar_text(payload, "table member")?;
    if name == PROPERTIES_MEMBER {
        return Ok(Value::Struct(table_public_properties(object)?));
    }
    if object.is_class(TIMETABLE_CLASS) {
        let props = table_public_properties(object)?;
        let time_dimension_name = match props.fields.get(DIMENSION_NAMES) {
            Some(Value::StringArray(array)) => match array.data().first() {
                Some(text) => text
                    .as_str()
                    .ok_or_else(|| invalid_variable("table: dimension name must not be missing"))?
                    .to_owned(),
                None => "Time".to_owned(),
            },
            Some(Value::Cell(cell)) => match cell.data().first() {
                Some(value) => scalar_text(value, "dimension name")?,
                None => "Time".to_owned(),
            },
            _ => "Time".to_owned(),
        };
        if name == ROW_TIMES || name == time_dimension_name {
            if let Some(row_times) = props.fields.get(ROW_TIMES).cloned() {
                return Ok(row_times);
            }
        }
    }
    let variables = table_variables(object)?;
    variables
        .fields
        .get(&name)
        .cloned()
        .ok_or_else(|| invalid_variable(format!("table: unrecognized variable '{name}'")))
}

pub(in crate::builtins::table) fn table_member_set(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    field: &str,
    rhs: Value,
) -> BuiltinResult<()> {
    if field == PROPERTIES_MEMBER {
        let Value::Struct(props) = rhs else {
            return Err(invalid_variable(
                "table: Properties assignment expects a scalar struct",
            ));
        };
        apply_properties(runtime, object, props)?;
        return Ok(());
    }
    let variables = table_variables(object)?;
    let mut names = table_variable_names_from_object(object)?;
    let height = table_height(object)?;
    let rhs_rows = value_row_count(&rhs)?;
    if !variables.fields.is_empty() && rhs_rows != height {
        return Err(invalid_variable(format!(
            "table: variable '{field}' has {rhs_rows} rows but table has {height}"
        )));
    }
    if !variables.fields.contains_key(field) {
        names.push(field.to_string());
    }
    let (variables, _variable_roots) = replace_struct_fields(runtime, &variables, &[(field, rhs)])?;
    let props = table_public_properties(object)?;
    let (props, _property_roots) = updated_variable_metadata_names(runtime, &props, names)?;
    let updates = [
        (TABLE_VARIABLES_FIELD, Value::Struct(variables)),
        (TABLE_PROPERTIES_FIELD, Value::Struct(props.clone())),
        (PROPERTIES_MEMBER, Value::Struct(props)),
    ];
    replace_table_properties(
        runtime,
        object,
        updates.iter().map(|(name, value)| (*name, value)),
    )
}

pub(in crate::builtins::table) fn table_paren_get(
    object: &ObjectInstance,
    payload: &Value,
) -> BuiltinResult<Value> {
    table_paren_get_in_context(None, object, payload)
}

pub(in crate::builtins::table) fn table_paren_get_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &ObjectInstance,
    payload: &Value,
) -> BuiltinResult<Value> {
    let selectors = selector_values(payload)?;
    let rows = parse_row_selector_for_object(selectors.first(), object)?;
    let variable_names = table_variable_names_from_object(object)?;
    let selected_names =
        parse_variable_selector_for_object(selectors.get(1), object, &variable_names)?;
    let variables = table_variables(object)?;
    let mut out = Vec::with_capacity(selected_names.len());
    for name in &selected_names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{name}'")))?;
        out.push(select_rows(value, &rows)?);
    }
    subset_tabular_object_in_context(runtime, object, selected_names, out, &rows)
}

pub(in crate::builtins::table) fn subset_tabular_object_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &ObjectInstance,
    names: Vec<String>,
    columns: Vec<Value>,
    rows: &[usize],
) -> BuiltinResult<Value> {
    let row_names = selected_row_names(source, rows)?;
    table_from_columns_like(runtime, source, names, columns, row_names, Some(rows))
}

pub(in crate::builtins::table) fn table_brace_get(
    object: &ObjectInstance,
    payload: &Value,
) -> BuiltinResult<Value> {
    let subset = table_paren_get(object, payload)?;
    let object = into_table_object(subset, "table brace indexing")?;
    let variables = table_variables(&object)?;
    if variables.fields.len() == 1 {
        return variables
            .fields
            .values()
            .next()
            .cloned()
            .ok_or_else(|| invalid_variable("table: missing selected variable"));
    }
    let values = variables.fields.values().collect::<Vec<_>>();
    if values.iter().all(|value| matches!(value, Value::Tensor(_))) {
        return concatenate_numeric_columns(&values);
    }
    CellArray::new(
        values.into_iter().cloned().collect(),
        1,
        variables.fields.len(),
    )
    .map(Value::Cell)
    .map_err(|err| invalid_variable(format!("table: {err}")))
}

pub(in crate::builtins::table) fn table_paren_assign(
    runtime: Option<&crate::context::RuntimeContext>,
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let rhs_table = into_table_object(rhs, "table paren assignment")?;
    let selectors = selector_values(payload)?;
    let rows = parse_row_selector_for_object(selectors.first(), &object)?;
    let variable_names = table_variable_names_from_object(&object)?;
    let selected_names =
        parse_variable_selector_for_object(selectors.get(1), &object, &variable_names)?;
    let rhs_names = table_variable_names_from_object(&rhs_table)?;
    if selected_names.len() != rhs_names.len() {
        return Err(invalid_variable(
            "table: assignment variable count must match selected variables",
        ));
    }
    let mut variables = table_variables(&object)?;
    let rhs_variables = table_variables(&rhs_table)?;
    for (target_name, rhs_name) in selected_names.iter().zip(rhs_names.iter()) {
        let current =
            variables.fields.get(target_name).cloned().ok_or_else(|| {
                invalid_variable(format!("table: missing variable '{target_name}'"))
            })?;
        let rhs_col =
            rhs_variables.fields.get(rhs_name).cloned().ok_or_else(|| {
                invalid_variable(format!("table: missing rhs variable '{rhs_name}'"))
            })?;
        let replacement = assign_rows(current, &rows, rhs_col)?;
        let (next, _roots) =
            replace_struct_fields(runtime, &variables, &[(target_name.as_str(), replacement)])?;
        variables = next;
    }
    replace_table_property(
        runtime,
        &mut object,
        TABLE_VARIABLES_FIELD,
        Value::Struct(variables),
    )?;
    Ok(Value::Object(object))
}

pub(in crate::builtins::table) fn table_brace_assign(
    runtime: Option<&crate::context::RuntimeContext>,
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let selectors = selector_values(payload)?;
    let rows = parse_row_selector_for_object(selectors.first(), &object)?;
    let variable_names = table_variable_names_from_object(&object)?;
    let selected_names =
        parse_variable_selector_for_object(selectors.get(1), &object, &variable_names)?;
    if selected_names.len() != 1 {
        return Err(invalid_variable(
            "table: brace assignment supports one variable at a time",
        ));
    }
    let variables = table_variables(&object)?;
    let target = selected_names[0].clone();
    let current = variables
        .fields
        .get(&target)
        .cloned()
        .ok_or_else(|| invalid_variable(format!("table: missing variable '{target}'")))?;
    let replacement = assign_rows(current, &rows, rhs)?;
    let (variables, _roots) =
        replace_struct_fields(runtime, &variables, &[(target.as_str(), replacement)])?;
    replace_table_property(
        runtime,
        &mut object,
        TABLE_VARIABLES_FIELD,
        Value::Struct(variables),
    )?;
    Ok(Value::Object(object))
}

pub(in crate::builtins::table) fn selector_values(payload: &Value) -> BuiltinResult<Vec<Value>> {
    match payload {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data().len());
            for handle in cell.data() {
                out.push(handle.clone());
            }
            Ok(out)
        }
        other => Ok(vec![other.clone()]),
    }
}
