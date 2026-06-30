use super::*;

pub(super) fn default_properties(
    variable_names: Vec<String>,
    row_names: Option<Vec<String>>,
) -> StructValue {
    default_properties_for_class(TABLE_CLASS, variable_names, row_names)
}

pub(super) fn default_properties_for_class(
    class_name: &str,
    variable_names: Vec<String>,
    row_names: Option<Vec<String>>,
) -> StructValue {
    let mut props = StructValue::new();
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(variable_names.clone(), vec![1, variable_names.len()])
                .expect("VariableNames shape is valid"),
        ),
    );
    props.insert(
        ROW_NAMES,
        row_names
            .map(|names| {
                Value::StringArray(
                    StringArray::new(names.clone(), vec![names.len(), 1])
                        .expect("RowNames shape is valid"),
                )
            })
            .unwrap_or_else(|| {
                Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap())
            }),
    );
    props.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(
                vec![
                    if class_name == TIMETABLE_CLASS {
                        "Time".to_string()
                    } else {
                        DEFAULT_ROW_DIM_NAME.to_string()
                    },
                    DEFAULT_VARIABLE_DIM_NAME.to_string(),
                ],
                vec![1, 2],
            )
            .expect("DimensionNames shape is valid"),
        ),
    );
    props.insert(
        VARIABLE_UNITS,
        Value::StringArray(
            StringArray::new(
                vec![String::new(); variable_names.len()],
                vec![1, variable_names.len()],
            )
            .expect("VariableUnits shape is valid"),
        ),
    );
    props.insert(
        VARIABLE_DESCRIPTIONS,
        Value::StringArray(
            StringArray::new(
                vec![String::new(); variable_names.len()],
                vec![1, variable_names.len()],
            )
            .expect("VariableDescriptions shape is valid"),
        ),
    );
    props.insert(DESCRIPTION, Value::String(String::new()));
    props.insert(USER_DATA, Value::Tensor(Tensor::zeros(vec![0, 0])));
    props
}

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

pub(super) fn table_from_columns_with_class(
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

pub(super) fn table_object(value: &Value) -> Option<&ObjectInstance> {
    match value {
        Value::Object(object) if is_tabular_class(object) => Some(object),
        _ => None,
    }
}

pub(super) fn into_table_object(value: Value, context: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if is_tabular_class(&object) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected table, got {other:?}"
        ))),
    }
}

pub(super) fn into_timetable_object(value: Value, context: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TIMETABLE_CLASS) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected timetable, got {other:?}"
        ))),
    }
}

pub(super) fn is_tabular_class(object: &ObjectInstance) -> bool {
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

pub(super) fn table_public_properties(object: &ObjectInstance) -> BuiltinResult<StructValue> {
    match object
        .properties
        .get(TABLE_PROPERTIES_FIELD)
        .or_else(|| object.properties.get(PROPERTIES_MEMBER))
    {
        Some(Value::Struct(st)) => Ok(st.clone()),
        Some(other) => Err(invalid_variable(format!(
            "table: invalid Properties storage {other:?}"
        ))),
        None => Ok(default_properties(
            table_variable_names_from_object(object)?,
            None,
        )),
    }
}

pub(super) fn sync_table_properties(object: &mut ObjectInstance, props: StructValue) {
    object.properties.insert(
        TABLE_PROPERTIES_FIELD.to_string(),
        Value::Struct(props.clone()),
    );
    object
        .properties
        .insert(PROPERTIES_MEMBER.to_string(), Value::Struct(props));
}

pub(super) fn table_member_get(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
    let name = scalar_text(payload, "table member")?;
    if name == PROPERTIES_MEMBER {
        return Ok(Value::Struct(table_public_properties(object)?));
    }
    if object.is_class(TIMETABLE_CLASS) {
        let props = table_public_properties(object)?;
        let time_dimension_name = props
            .fields
            .get(DIMENSION_NAMES)
            .and_then(|value| match value {
                Value::StringArray(array) => array.data.first().cloned(),
                Value::Cell(cell) => cell
                    .data
                    .first()
                    .and_then(|value| scalar_text(value, "dimension name").ok()),
                _ => None,
            })
            .unwrap_or_else(|| "Time".to_string());
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

pub(super) fn table_member_set(
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
        apply_properties(object, props)?;
        return Ok(());
    }
    let mut variables = table_variables(object)?;
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
    variables.insert(field.to_string(), rhs);
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    let mut props = table_public_properties(object)?;
    update_variable_metadata_names(&mut props, names)?;
    sync_table_properties(object, props);
    Ok(())
}

pub(super) fn apply_properties(
    object: &mut ObjectInstance,
    mut props: StructValue,
) -> BuiltinResult<()> {
    if let Some(value) = props.fields.get(VARIABLE_NAMES) {
        let names = variable_name_list(value)?;
        rename_table_variables(object, names.clone())?;
        update_variable_metadata_names(&mut props, names)?;
    }
    sync_table_properties(object, props);
    Ok(())
}

pub(super) fn rename_table_variables(
    object: &mut ObjectInstance,
    new_names: Vec<String>,
) -> BuiltinResult<()> {
    let old_names = table_variable_names_from_object(object)?;
    if old_names.len() != new_names.len() {
        return Err(invalid_variable(
            "table: VariableNames assignment must preserve variable count",
        ));
    }
    let new_names = make_unique_variable_names(new_names);
    let variables = table_variables(object)?;
    let mut renamed = StructValue::new();
    for (old, new) in old_names.iter().zip(new_names.iter()) {
        let value = variables
            .fields
            .get(old)
            .cloned()
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{old}'")))?;
        renamed.insert(new.clone(), value);
    }
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(renamed));
    Ok(())
}

pub(super) fn update_variable_metadata_names(
    props: &mut StructValue,
    names: Vec<String>,
) -> BuiltinResult<()> {
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(names.clone(), vec![1, names.len()])
                .map_err(|err| invalid_variable(format!("table: {err}")))?,
        ),
    );
    for field in [VARIABLE_UNITS, VARIABLE_DESCRIPTIONS] {
        let existing = props.fields.get(field).cloned();
        let values = match existing {
            Some(Value::StringArray(mut array)) => {
                array.data.resize(names.len(), String::new());
                array.data.truncate(names.len());
                array.data
            }
            _ => vec![String::new(); names.len()],
        };
        props.insert(
            field,
            Value::StringArray(
                StringArray::new(values, vec![1, names.len()])
                    .map_err(|err| invalid_variable(format!("table: {err}")))?,
            ),
        );
    }
    Ok(())
}

pub(super) fn table_paren_get(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
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
    subset_tabular_object(object, selected_names, out, &rows)
}

pub(super) fn subset_tabular_object(
    source: &ObjectInstance,
    names: Vec<String>,
    columns: Vec<Value>,
    rows: &[usize],
) -> BuiltinResult<Value> {
    let row_names = selected_row_names(source, rows)?;
    table_from_columns_like(source, names, columns, row_names, Some(rows))
}

pub(super) fn table_brace_get(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
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

pub(super) fn table_paren_assign(
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
        variables.insert(target_name.clone(), assign_rows(current, &rows, rhs_col)?);
    }
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    Ok(Value::Object(object))
}

pub(super) fn table_brace_assign(
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
    let mut variables = table_variables(&object)?;
    let target = selected_names[0].clone();
    let current = variables
        .fields
        .get(&target)
        .cloned()
        .ok_or_else(|| invalid_variable(format!("table: missing variable '{target}'")))?;
    variables.insert(target, assign_rows(current, &rows, rhs)?);
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    Ok(Value::Object(object))
}

pub(super) fn selector_values(payload: &Value) -> BuiltinResult<Vec<Value>> {
    match payload {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for handle in &cell.data {
                out.push(handle.clone());
            }
            Ok(out)
        }
        other => Ok(vec![other.clone()]),
    }
}

pub(super) fn parse_row_selector(
    selector: Option<&Value>,
    height: usize,
) -> BuiltinResult<Vec<usize>> {
    let Some(selector) = selector else {
        return Ok((0..height).collect());
    };
    if is_colon_selector(selector) {
        return Ok((0..height).collect());
    }
    if is_end_selector(selector) {
        return if height == 0 {
            Err(invalid_index(
                "table: end row index is invalid for empty table",
            ))
        } else {
            Ok(vec![height - 1])
        };
    }
    match selector {
        Value::Num(n) => Ok(vec![one_based_to_zero(*n, height, "row")?]),
        Value::Int(i) => Ok(vec![one_based_to_zero(i.to_f64(), height, "row")?]),
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|value| one_based_to_zero(*value, height, "row"))
            .collect(),
        Value::LogicalArray(array) => {
            if array.data.len() != height {
                return Err(invalid_index(
                    "table: logical row selector length must match table height",
                ));
            }
            Ok(array
                .data
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| (*value != 0).then_some(idx))
                .collect())
        }
        other => Err(invalid_index(format!(
            "table: unsupported row selector {other:?}"
        ))),
    }
}

pub(super) fn parse_row_selector_for_object(
    selector: Option<&Value>,
    object: &ObjectInstance,
) -> BuiltinResult<Vec<usize>> {
    let height = table_height(object)?;
    let Some(selector) = selector else {
        return Ok((0..height).collect());
    };
    if let Value::Object(selector_object) = selector {
        if selector_object.is_class(TIMERANGE_CLASS) {
            return parse_timerange_selector(selector_object, object);
        }
        if selector_object.is_class(ROWFILTER_CLASS) {
            return parse_rowfilter_selector(selector_object, object);
        }
    }
    parse_row_selector(Some(selector), height)
}

pub(super) fn parse_variable_selector(
    selector: Option<&Value>,
    names: &[String],
) -> BuiltinResult<Vec<String>> {
    let Some(selector) = selector else {
        return Ok(names.to_vec());
    };
    if is_colon_selector(selector) {
        return Ok(names.to_vec());
    }
    match selector {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) | Value::Cell(_) => {
            let selected = string_list(selector)?;
            for name in &selected {
                if !names.contains(name) {
                    return Err(invalid_variable(format!(
                        "table: unrecognized variable '{name}'"
                    )));
                }
            }
            Ok(selected)
        }
        Value::Num(n) => Ok(vec![name_at_index(names, *n)?]),
        Value::Int(i) => Ok(vec![name_at_index(names, i.to_f64())?]),
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|value| name_at_index(names, *value))
            .collect(),
        Value::LogicalArray(array) => {
            if array.data.len() != names.len() {
                return Err(invalid_index(
                    "table: logical variable selector length must match table width",
                ));
            }
            Ok(array
                .data
                .iter()
                .zip(names.iter())
                .filter_map(|(flag, name)| (*flag != 0).then_some(name.clone()))
                .collect())
        }
        other => Err(invalid_index(format!(
            "table: unsupported variable selector {other:?}"
        ))),
    }
}

pub(super) fn parse_variable_selector_for_object(
    selector: Option<&Value>,
    object: &ObjectInstance,
    names: &[String],
) -> BuiltinResult<Vec<String>> {
    let Some(selector) = selector else {
        return Ok(names.to_vec());
    };
    if let Value::Object(selector_object) = selector {
        if selector_object.is_class(VARTYPE_CLASS) {
            return parse_vartype_selector(selector_object, object, names);
        }
    }
    parse_variable_selector(Some(selector), names)
}

pub(super) fn parse_vartype_selector(
    selector: &ObjectInstance,
    object: &ObjectInstance,
    names: &[String],
) -> BuiltinResult<Vec<String>> {
    let kind = selector
        .properties
        .get("Type")
        .map(|value| scalar_text(value, "vartype type"))
        .transpose()?
        .unwrap_or_else(|| "all".to_string())
        .to_ascii_lowercase();
    let variables = table_variables(object)?;
    Ok(names
        .iter()
        .filter(|name| {
            variables
                .fields
                .get(*name)
                .map(|value| vartype_matches(value, &kind))
                .unwrap_or(false)
        })
        .cloned()
        .collect())
}

pub(super) fn vartype_matches(value: &Value, kind: &str) -> bool {
    match kind {
        "all" => true,
        "numeric" | "float" | "floating" => matches!(
            value,
            Value::Tensor(_) | Value::ComplexTensor(_) | Value::Num(_) | Value::Complex(_, _)
        ),
        "logical" => matches!(value, Value::LogicalArray(_) | Value::Bool(_)),
        "string" | "text" => matches!(
            value,
            Value::StringArray(_) | Value::String(_) | Value::CharArray(_)
        ),
        "cell" => matches!(value, Value::Cell(_)),
        "datetime" => matches!(value, Value::Object(obj) if obj.is_class("datetime")),
        "duration" => matches!(value, Value::Object(obj) if obj.is_class("duration")),
        "categorical" => matches!(value, Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS)),
        "table" => matches!(value, Value::Object(obj) if is_tabular_class(obj)),
        _ => false,
    }
}

pub(super) fn parse_timerange_selector(
    selector: &ObjectInstance,
    object: &ObjectInstance,
) -> BuiltinResult<Vec<usize>> {
    if !object.is_class(TIMETABLE_CLASS) {
        return Err(invalid_index("timerange selector requires a timetable"));
    }
    let row_times = timetable_row_times(object)?
        .ok_or_else(|| invalid_index("timerange selector requires timetable RowTimes"))?;
    let serials = selector_numeric_values(&row_times)?;
    let start = selector
        .properties
        .get("Start")
        .map(selector_bound_value)
        .transpose()?;
    let end = selector
        .properties
        .get("End")
        .map(selector_bound_value)
        .transpose()?;
    let inclusivity = selector
        .properties
        .get("Inclusivity")
        .map(|value| scalar_text(value, "timerange inclusivity"))
        .transpose()?
        .unwrap_or_else(|| "closed".to_string())
        .to_ascii_lowercase();
    let (include_start, include_end) = match inclusivity.as_str() {
        "closed" => (true, true),
        "open" => (false, false),
        "openleft" => (false, true),
        "openright" => (true, false),
        other => {
            return Err(invalid_index(format!(
                "timerange: unsupported inclusivity '{other}'"
            )))
        }
    };
    Ok(serials
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let after_start = start
                .map(|start| {
                    if include_start {
                        *value >= start
                    } else {
                        *value > start
                    }
                })
                .unwrap_or(true);
            let before_end = end
                .map(|end| {
                    if include_end {
                        *value <= end
                    } else {
                        *value < end
                    }
                })
                .unwrap_or(true);
            (after_start && before_end).then_some(idx)
        })
        .collect())
}

pub(super) fn parse_rowfilter_selector(
    selector: &ObjectInstance,
    object: &ObjectInstance,
) -> BuiltinResult<Vec<usize>> {
    let height = table_height(object)?;
    let predicate = selector.properties.get("Predicate");
    if let Some(predicate) = predicate {
        match predicate {
            Value::LogicalArray(mask) => {
                return logical_array_mask_for_table_rows(mask, height);
            }
            Value::Bool(flag) => {
                return Ok(if *flag {
                    (0..height).collect()
                } else {
                    Vec::new()
                });
            }
            Value::String(text) if text.is_empty() => {}
            Value::String(text) => {
                return evaluate_named_rowfilter(text, selector, object);
            }
            Value::CharArray(chars) if chars.rows == 1 => {
                let text = chars.data.iter().collect::<String>();
                return evaluate_named_rowfilter(&text, selector, object);
            }
            _ => {}
        }
    }
    Ok((0..height).collect())
}

pub(super) fn logical_array_mask_for_table_rows(
    mask: &LogicalArray,
    height: usize,
) -> BuiltinResult<Vec<usize>> {
    let rows = mask.shape.first().copied().unwrap_or(mask.data.len());
    if rows != height && mask.data.len() != height {
        return Err(invalid_index(
            "rowfilter: logical predicate length must match table height",
        ));
    }
    Ok(mask
        .data
        .iter()
        .take(height)
        .enumerate()
        .filter_map(|(idx, flag)| (*flag != 0).then_some(idx))
        .collect())
}

pub(super) fn evaluate_named_rowfilter(
    predicate: &str,
    selector: &ObjectInstance,
    object: &ObjectInstance,
) -> BuiltinResult<Vec<usize>> {
    let variable_names = selector
        .properties
        .get("Variables")
        .map(string_list)
        .transpose()?
        .unwrap_or_else(|| table_variable_names_from_object(object).unwrap_or_default());
    if variable_names.is_empty() {
        return Ok(Vec::new());
    }
    let variables = table_variables(object)?;
    let selected_values = variable_names
        .iter()
        .map(|name| {
            variables
                .fields
                .get(name)
                .ok_or_else(|| invalid_variable(format!("rowfilter: missing variable '{name}'")))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let normalized = predicate
        .trim()
        .trim_start_matches('@')
        .to_ascii_lowercase();
    let height = table_height(object)?;
    let mut rows = Vec::new();
    for row in 0..height {
        let keep = match normalized.as_str() {
            "gt0" | ">0" | "positive" => selected_values
                .iter()
                .map(|value| numeric_cell(value, row).map(|v| v > 0.0))
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .all(|flag| flag),
            "ge0" | ">=0" | "nonnegative" => selected_values
                .iter()
                .map(|value| numeric_cell(value, row).map(|v| v >= 0.0))
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .all(|flag| flag),
            "lt0" | "<0" | "negative" => selected_values
                .iter()
                .map(|value| numeric_cell(value, row).map(|v| v < 0.0))
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .all(|flag| flag),
            "nonmissing" | "notmissing" => selected_values.iter().all(|value| {
                !row_value(value, row)
                    .map(|value| value_is_missing_scalar(&value))
                    .unwrap_or(false)
            }),
            _ => {
                return Err(invalid_argument(format!(
                    "rowfilter: unsupported predicate '{predicate}'"
                )))
            }
        };
        if keep {
            rows.push(row);
        }
    }
    Ok(rows)
}

pub(super) fn selector_numeric_values(value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Object(obj) if obj.is_class("datetime") => {
            Ok(crate::builtins::datetime::serials_from_datetime_value(value)?.data)
        }
        Value::Object(obj) if obj.is_class("duration") => {
            Ok(crate::builtins::duration::duration_tensor_from_duration_value(value)?.data)
        }
        other => Err(invalid_argument(format!(
            "timerange: expected numeric, datetime, or duration row times, got {other:?}"
        ))),
    }
}

pub(super) fn selector_bound_value(value: &Value) -> BuiltinResult<f64> {
    selector_numeric_values(value)?
        .into_iter()
        .next()
        .ok_or_else(|| invalid_argument("timerange: boundary must not be empty"))
}

pub(super) fn numeric_cell(value: &Value, row: usize) -> BuiltinResult<f64> {
    match row_value(value, row)? {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if value { 1.0 } else { 0.0 }),
        other => Err(invalid_argument(format!(
            "rowfilter: expected numeric predicate variable, got {other:?}"
        ))),
    }
}

pub(super) fn value_is_missing_scalar(value: &Value) -> bool {
    match value {
        Value::Num(value) => value.is_nan(),
        Value::String(text) => text.is_empty() || text == "<missing>",
        Value::StringArray(array) => array
            .data
            .first()
            .map(|text| text.is_empty())
            .unwrap_or(true),
        Value::CharArray(array) => array.data.iter().all(|ch| ch.is_whitespace()),
        Value::Tensor(tensor) => tensor
            .data
            .first()
            .map(|value| value.is_nan())
            .unwrap_or(true),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.first().copied())
                .map(|serial| serial.is_nan())
                .unwrap_or(false)
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .ok()
                .and_then(|tensor| tensor.data.first().copied())
                .map(|days| days.is_nan())
                .unwrap_or(false)
        }
        _ => false,
    }
}

pub(super) fn is_colon_selector(value: &Value) -> bool {
    scalar_text(value, "selector")
        .map(|text| text == ":")
        .unwrap_or(false)
}

pub(super) fn is_end_selector(value: &Value) -> bool {
    scalar_text(value, "selector")
        .map(|text| text == "end")
        .unwrap_or(false)
}

pub(super) fn name_at_index(names: &[String], value: f64) -> BuiltinResult<String> {
    let idx = one_based_to_zero(value, names.len(), "variable")?;
    Ok(names[idx].clone())
}

pub(super) fn one_based_to_zero(value: f64, len: usize, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(invalid_index(format!(
            "table: {context} indices must be positive finite integers"
        )));
    }
    let idx = value.round() as usize - 1;
    if idx >= len {
        return Err(invalid_index(format!(
            "table: {context} index exceeds bounds"
        )));
    }
    Ok(idx)
}

pub(crate) fn selected_row_names(
    object: &ObjectInstance,
    rows: &[usize],
) -> BuiltinResult<Option<Vec<String>>> {
    let props = table_public_properties(object)?;
    let Some(value) = props.fields.get(ROW_NAMES) else {
        return Ok(None);
    };
    let names = string_list(value)?;
    if names.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        rows.iter()
            .filter_map(|row| names.get(*row).cloned())
            .collect(),
    ))
}

pub(super) fn selected_row_times(
    object: &ObjectInstance,
    rows: &[usize],
) -> BuiltinResult<Option<Value>> {
    let Some(row_times) = timetable_row_times(object)? else {
        return Ok(None);
    };
    select_rows(&row_times, rows).map(Some)
}

pub(crate) fn value_row_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.rows()),
        Value::ComplexTensor(tensor) => Ok(tensor.rows),
        Value::StringArray(array) => Ok(array.rows()),
        Value::LogicalArray(array) => Ok(array.shape.first().copied().unwrap_or(array.data.len())),
        Value::Cell(cell) => Ok(cell.rows),
        Value::CharArray(array) => Ok(array.rows),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .map(|tensor| tensor.rows())
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .map(|tensor| tensor.rows())
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => obj
            .properties
            .get("Codes")
            .map(value_row_count)
            .transpose()
            .map(|rows| rows.unwrap_or(0)),
        Value::Object(obj) if is_tabular_class(obj) => table_height(obj),
        _ => Ok(1),
    }
}

pub(crate) fn select_rows(value: &Value, rows: &[usize]) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let cols = tensor.cols();
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    data.push(tensor.get2(row, col).map_err(invalid_index)?);
                }
            }
            Tensor::new_with_dtype(data, vec![rows.len(), cols], tensor.dtype)
                .map(Value::Tensor)
                .map_err(invalid_variable)
        }
        Value::ComplexTensor(tensor) => {
            let mut data = Vec::with_capacity(rows.len() * tensor.cols);
            for col in 0..tensor.cols {
                for &row in rows {
                    let idx = row + col * tensor.rows;
                    data.push(*tensor.data.get(idx).ok_or_else(|| {
                        invalid_index("table: complex variable row index out of bounds")
                    })?);
                }
            }
            ComplexTensor::new(data, vec![rows.len(), tensor.cols])
                .map(Value::ComplexTensor)
                .map_err(invalid_variable)
        }
        Value::StringArray(array) => {
            let cols = array.cols();
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    let idx = row + col * array.rows();
                    data.push(array.data.get(idx).cloned().ok_or_else(|| {
                        invalid_index("table: string variable row index out of bounds")
                    })?);
                }
            }
            StringArray::new(data, vec![rows.len(), cols])
                .map(Value::StringArray)
                .map_err(invalid_variable)
        }
        Value::CharArray(array) => {
            let mut data = Vec::with_capacity(rows.len() * array.cols);
            for &row in rows {
                if row >= array.rows {
                    return Err(invalid_index(
                        "table: char variable row index out of bounds",
                    ));
                }
                let start = row * array.cols;
                data.extend_from_slice(&array.data[start..start + array.cols]);
            }
            CharArray::new(data, rows.len(), array.cols)
                .map(Value::CharArray)
                .map_err(invalid_variable)
        }
        Value::LogicalArray(array) => {
            let source_rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    let idx = row + col * source_rows;
                    data.push(*array.data.get(idx).ok_or_else(|| {
                        invalid_index("table: logical variable row index out of bounds")
                    })?);
                }
            }
            LogicalArray::new(data, vec![rows.len(), cols])
                .map(Value::LogicalArray)
                .map_err(invalid_variable)
        }
        Value::Cell(cell) => {
            let mut data = Vec::with_capacity(rows.len() * cell.cols);
            for col in 0..cell.cols {
                for &row in rows {
                    data.push(cell.get(row, col).map_err(invalid_index)?);
                }
            }
            CellArray::new(data, rows.len(), cell.cols)
                .map(Value::Cell)
                .map_err(invalid_variable)
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            let selected = select_rows(&Value::Tensor(tensor), rows)?;
            match selected {
                Value::Tensor(tensor) => {
                    crate::builtins::datetime::datetime_object_from_serial_tensor(
                        tensor,
                        crate::builtins::datetime::datetime_format_from_value(value),
                    )
                }
                _ => unreachable!("select_rows tensor branch returns tensor"),
            }
        }
        Value::Object(obj) if obj.is_class("duration") => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let selected = select_rows(&Value::Tensor(tensor), rows)?;
            match selected {
                Value::Tensor(tensor) => {
                    crate::builtins::duration::duration_object_from_days_tensor(
                        tensor,
                        crate::builtins::duration::duration_format_from_value(value),
                    )
                }
                _ => unreachable!("select_rows tensor branch returns tensor"),
            }
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => {
            let codes = obj
                .properties
                .get("Codes")
                .ok_or_else(|| invalid_variable("categorical: missing Codes property"))?;
            let selected_codes = select_rows(codes, rows)?;
            let mut out = obj.clone();
            out.properties.insert("Codes".to_string(), selected_codes);
            Ok(Value::Object(out))
        }
        _ if rows.len() == 1 && rows[0] == 0 => Ok(value.clone()),
        other => Err(invalid_variable(format!(
            "table: row selection unsupported for variable {other:?}"
        ))),
    }
}

pub(super) fn assign_rows(mut current: Value, rows: &[usize], rhs: Value) -> BuiltinResult<Value> {
    if value_row_count(&rhs)? != rows.len() {
        return Err(invalid_variable(
            "table: assignment row count must match selected row count",
        ));
    }
    let replacing_all_rows = rows.len() == value_row_count(&current)?;
    match (&mut current, rhs) {
        (Value::Tensor(target), Value::Tensor(source)) => {
            if target.cols() != source.cols() {
                return Err(invalid_variable(
                    "table: tensor assignment column count mismatch",
                ));
            }
            for col in 0..target.cols() {
                for (src_row, &dst_row) in rows.iter().enumerate() {
                    let value = source.get2(src_row, col).map_err(invalid_index)?;
                    target.set2(dst_row, col, value).map_err(invalid_index)?;
                }
            }
            Ok(current)
        }
        (_, source) if replacing_all_rows => Ok(source),
        _ => Err(invalid_variable(
            "table: assignment for this variable type requires replacing all rows",
        )),
    }
}

pub(super) fn concatenate_numeric_columns(values: &[&Value]) -> BuiltinResult<Value> {
    let rows = values
        .first()
        .and_then(|value| match value {
            Value::Tensor(t) => Some(t.rows()),
            _ => None,
        })
        .unwrap_or(0);
    let cols = values
        .iter()
        .map(|value| match value {
            Value::Tensor(t) => Ok(t.cols()),
            _ => Err(invalid_variable("table: expected numeric variable")),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let total_cols: usize = cols.iter().sum();
    let mut data = Vec::with_capacity(rows * total_cols);
    for value in values {
        let Value::Tensor(tensor) = value else {
            return Err(invalid_variable("table: expected numeric variable"));
        };
        for col in 0..tensor.cols() {
            for row in 0..rows {
                data.push(tensor.get2(row, col).map_err(invalid_index)?);
            }
        }
    }
    Tensor::new(data, vec![rows, total_cols])
        .map(Value::Tensor)
        .map_err(invalid_variable)
}

pub fn sortrows_table(value: Value, rest: &[Value]) -> BuiltinResult<(Value, Tensor)> {
    let object = into_table_object(value, "sortrows")?;
    let names = table_variable_names_from_object(&object)?;
    let sort_spec = SortSpec::parse(rest, &names)?;
    let height = table_height(&object)?;
    let variables = table_variables(&object)?;
    let mut indices: Vec<usize> = (0..height).collect();
    indices.sort_by(|&a, &b| {
        for key in &sort_spec.keys {
            let Some(value) = variables.fields.get(&key.name) else {
                continue;
            };
            let ord = compare_table_cells(value, a, b).unwrap_or(Ordering::Equal);
            let ord = if key.descending { ord.reverse() } else { ord };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a.cmp(&b)
    });
    let mut sorted_columns = Vec::with_capacity(names.len());
    for name in &names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{name}'")))?;
        sorted_columns.push(select_rows(value, &indices)?);
    }
    let row_names = selected_row_names(&object, &indices)?;
    let sorted = table_from_columns_with_properties(names, sorted_columns, row_names)?;
    let indices_tensor = Tensor::new(
        indices.iter().map(|idx| *idx as f64 + 1.0).collect(),
        vec![indices.len(), 1],
    )
    .map_err(invalid_variable)?;
    Ok((sorted, indices_tensor))
}

pub(super) struct SortSpec {
    keys: Vec<SortKey>,
}

pub(super) struct SortKey {
    name: String,
    descending: bool,
}

impl SortSpec {
    fn parse(rest: &[Value], names: &[String]) -> BuiltinResult<Self> {
        let mut keys = if rest.is_empty() {
            names
                .iter()
                .map(|name| SortKey {
                    name: name.clone(),
                    descending: false,
                })
                .collect::<Vec<_>>()
        } else {
            parse_variable_selector(rest.first(), names)?
                .into_iter()
                .map(|name| SortKey {
                    name,
                    descending: false,
                })
                .collect()
        };
        if let Some(direction) = rest.get(1) {
            let directions = string_list(direction)?;
            if directions.len() == 1 {
                let descending = directions[0].eq_ignore_ascii_case("descend")
                    || directions[0].eq_ignore_ascii_case("desc");
                for key in &mut keys {
                    key.descending = descending;
                }
            } else {
                for (key, direction) in keys.iter_mut().zip(directions.iter()) {
                    key.descending = direction.eq_ignore_ascii_case("descend")
                        || direction.eq_ignore_ascii_case("desc");
                }
            }
        }
        Ok(Self { keys })
    }
}

pub(super) fn compare_table_cells(value: &Value, a: usize, b: usize) -> BuiltinResult<Ordering> {
    match value {
        Value::Tensor(tensor) => Ok(tensor
            .get2(a, 0)
            .map_err(invalid_index)?
            .partial_cmp(&tensor.get2(b, 0).map_err(invalid_index)?)
            .unwrap_or(Ordering::Greater)),
        Value::StringArray(array) => {
            let av = array.data.get(a).cloned().unwrap_or_default();
            let bv = array.data.get(b).cloned().unwrap_or_default();
            Ok(av.cmp(&bv))
        }
        Value::LogicalArray(array) => {
            let av = *array.data.get(a).unwrap_or(&0);
            let bv = *array.data.get(b).unwrap_or(&0);
            Ok(av.cmp(&bv))
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            Ok(tensor
                .data
                .get(a)
                .copied()
                .unwrap_or(f64::NAN)
                .partial_cmp(&tensor.data.get(b).copied().unwrap_or(f64::NAN))
                .unwrap_or(Ordering::Greater))
        }
        other => Ok(cell_key_string(other, a).cmp(&cell_key_string(other, b))),
    }
}

#[derive(Clone, Debug)]
pub(super) enum GroupAtom {
    Number(f64),
    Text(String),
    Logical(bool),
    Missing,
}

impl GroupAtom {
    fn rank(&self) -> u8 {
        match self {
            Self::Missing => 0,
            Self::Logical(_) => 1,
            Self::Number(_) => 2,
            Self::Text(_) => 3,
        }
    }
}

impl PartialEq for GroupAtom {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for GroupAtom {}

impl PartialOrd for GroupAtom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GroupAtom {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank = self.rank().cmp(&other.rank());
        if rank != Ordering::Equal {
            return rank;
        }
        match (self, other) {
            (Self::Missing, Self::Missing) => Ordering::Equal,
            (Self::Logical(a), Self::Logical(b)) => a.cmp(b),
            (Self::Number(a), Self::Number(b)) => a.total_cmp(b),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

pub(super) fn cell_group_atom(value: &Value, row: usize) -> GroupAtom {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(GroupAtom::Number)
            .unwrap_or(GroupAtom::Missing),
        Value::StringArray(array) => array
            .data
            .get(row)
            .cloned()
            .map(GroupAtom::Text)
            .unwrap_or(GroupAtom::Missing),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| GroupAtom::Logical(*value != 0))
            .unwrap_or(GroupAtom::Missing),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(GroupAtom::Number)
                .unwrap_or(GroupAtom::Missing)
        }
        other => GroupAtom::Text(cell_key_string(other, row)),
    }
}

pub(super) fn pivot_impl(
    table: Value,
    rowvars: Value,
    colvars: Value,
    datavar: Value,
    method: &str,
) -> BuiltinResult<Value> {
    let object = into_table_object(table, "pivot")?;
    let names = table_variable_names_from_object(&object)?;
    let row_names = parse_variable_selector_for_object(Some(&rowvars), &object, &names)?;
    let col_names = parse_variable_selector_for_object(Some(&colvars), &object, &names)?;
    let data_names = parse_variable_selector_for_object(Some(&datavar), &object, &names)?;
    if row_names.is_empty() || col_names.is_empty() || data_names.is_empty() {
        return Err(invalid_argument(
            "pivot: rowvars, colvars, and datavar must select at least one variable",
        ));
    }
    if data_names.len() != 1 {
        return Err(invalid_argument(
            "pivot: exactly one data variable is currently supported",
        ));
    }
    let data_name = &data_names[0];
    let variables = table_variables(&object)?;
    let data_value = variables
        .fields
        .get(data_name)
        .ok_or_else(|| invalid_variable(format!("pivot: missing data variable '{data_name}'")))?;
    if !matches!(data_value, Value::Tensor(tensor) if tensor.cols() == 1) {
        return Err(invalid_variable(
            "pivot: data variable must be a numeric column vector",
        ));
    }

    let height = table_height(&object)?;
    let mut row_order = Vec::<Vec<GroupAtom>>::new();
    let mut row_first_index = BTreeMap::<Vec<GroupAtom>, usize>::new();
    let mut col_order = Vec::<Vec<GroupAtom>>::new();
    let mut col_seen = BTreeMap::<Vec<GroupAtom>, ()>::new();
    let mut buckets = BTreeMap::<(Vec<GroupAtom>, Vec<GroupAtom>), Vec<usize>>::new();
    for row in 0..height {
        let row_key = group_key_for_row(&variables, &row_names, row);
        let col_key = group_key_for_row(&variables, &col_names, row);
        if !row_first_index.contains_key(&row_key) {
            row_first_index.insert(row_key.clone(), row);
            row_order.push(row_key.clone());
        }
        if !col_seen.contains_key(&col_key) {
            col_seen.insert(col_key.clone(), ());
            col_order.push(col_key.clone());
        }
        buckets.entry((row_key, col_key)).or_default().push(row);
    }

    let mut out_names = row_names.clone();
    let mut out_columns = Vec::with_capacity(row_names.len() + col_order.len());
    for name in &row_names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("pivot: missing row variable '{name}'")))?;
        let rows = row_order
            .iter()
            .filter_map(|key| row_first_index.get(key).copied())
            .collect::<Vec<_>>();
        out_columns.push(select_rows(value, &rows)?);
    }
    for col_key in &col_order {
        let mut values = Vec::with_capacity(row_order.len());
        for row_key in &row_order {
            let summary_rows = buckets
                .get(&(row_key.clone(), col_key.clone()))
                .cloned()
                .unwrap_or_default();
            if summary_rows.is_empty() {
                values.push(f64::NAN);
            } else {
                values.push(
                    summarize_groups(data_value, std::iter::once(&summary_rows), method)?
                        .into_iter()
                        .next()
                        .unwrap_or(f64::NAN),
                );
            }
        }
        out_names.push(format!(
            "{}_{}",
            make_valid_variable_name(&group_key_label(col_key), out_names.len() + 1),
            data_name
        ));
        out_columns.push(Value::Tensor(
            Tensor::new(values, vec![row_order.len(), 1]).map_err(invalid_variable)?,
        ));
    }
    let out_names = make_unique_variable_names(out_names);
    table_from_columns(out_names, out_columns)
}

pub(super) fn group_key_for_row(
    variables: &StructValue,
    names: &[String],
    row: usize,
) -> Vec<GroupAtom> {
    names
        .iter()
        .map(|name| {
            variables
                .fields
                .get(name)
                .map(|value| cell_group_atom(value, row))
                .unwrap_or(GroupAtom::Missing)
        })
        .collect()
}

pub(super) fn group_key_label(key: &[GroupAtom]) -> String {
    if key.is_empty() {
        return "missing".to_string();
    }
    key.iter()
        .map(group_atom_label)
        .collect::<Vec<_>>()
        .join("_")
}

pub(super) fn group_atom_label(atom: &GroupAtom) -> String {
    match atom {
        GroupAtom::Number(value) => format_key_number(*value),
        GroupAtom::Text(text) => text.clone(),
        GroupAtom::Logical(flag) => flag.to_string(),
        GroupAtom::Missing => "missing".to_string(),
    }
}

pub(super) fn groupsummary_impl(
    table: Value,
    groupvars: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let object = into_table_object(table, "groupsummary")?;
    let names = table_variable_names_from_object(&object)?;
    let group_names = parse_variable_selector_for_object(Some(&groupvars), &object, &names)?;
    let methods = string_list(&method)?;
    if methods.is_empty() {
        return Err(invalid_argument(
            "groupsummary: method list must not be empty",
        ));
    }
    let data_names = if let Some(value) = rest.first() {
        parse_variable_selector_for_object(Some(value), &object, &names)?
    } else {
        names
            .iter()
            .filter(|name| !group_names.contains(name))
            .filter(|name| {
                table_variables(&object)
                    .ok()
                    .and_then(|vars| vars.fields.get(*name).cloned())
                    .map(|value| matches!(value, Value::Tensor(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    };
    let variables = table_variables(&object)?;
    let height = table_height(&object)?;
    let mut groups: BTreeMap<Vec<GroupAtom>, Vec<usize>> = BTreeMap::new();
    for row in 0..height {
        let key = group_names
            .iter()
            .map(|name| {
                variables
                    .fields
                    .get(name)
                    .map(|value| cell_group_atom(value, row))
                    .unwrap_or(GroupAtom::Missing)
            })
            .collect::<Vec<_>>();
        groups.entry(key).or_default().push(row);
    }
    let group_rows = groups
        .values()
        .filter_map(|rows| rows.first().copied())
        .collect::<Vec<_>>();
    let mut out_names = Vec::new();
    let mut out_columns = Vec::new();
    for name in &group_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            invalid_variable(format!("groupsummary: missing group variable '{name}'"))
        })?;
        out_names.push(name.clone());
        out_columns.push(select_rows(value, &group_rows)?);
    }
    out_names.push("GroupCount".to_string());
    out_columns.push(Value::Tensor(
        Tensor::new(
            groups.values().map(|rows| rows.len() as f64).collect(),
            vec![groups.len(), 1],
        )
        .map_err(invalid_variable)?,
    ));
    for method in &methods {
        for name in &data_names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("groupsummary: missing data variable '{name}'"))
            })?;
            let values = summarize_groups(value, groups.values(), method)?;
            out_names.push(format!("{}_{}", method.to_ascii_lowercase(), name));
            out_columns.push(Value::Tensor(
                Tensor::new(values, vec![groups.len(), 1]).map_err(invalid_variable)?,
            ));
        }
    }
    table_from_columns(out_names, out_columns)
}

pub(super) fn summarize_groups<'a>(
    value: &Value,
    groups: impl Iterator<Item = &'a Vec<usize>>,
    method: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => tensor,
        _ => {
            return Err(invalid_variable(
                "groupsummary: summary data variables must be numeric column vectors",
            ))
        }
    };
    groups
        .map(|rows| {
            let mut values = rows
                .iter()
                .map(|row| tensor.get2(*row, 0).map_err(invalid_index))
                .collect::<BuiltinResult<Vec<_>>>()?;
            values.retain(|value| !value.is_nan());
            let result = match method.to_ascii_lowercase().as_str() {
                "mean" => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                }
                "sum" => values.iter().sum(),
                "min" => values.into_iter().fold(f64::INFINITY, f64::min),
                "max" => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                "median" => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let mid = values.len() / 2;
                        if values.len() % 2 == 0 {
                            (values[mid - 1] + values[mid]) / 2.0
                        } else {
                            values[mid]
                        }
                    }
                }
                "count" | "numel" => values.len() as f64,
                other => {
                    return Err(invalid_argument(format!(
                        "groupsummary: unsupported method '{other}'"
                    )))
                }
            };
            Ok(result)
        })
        .collect()
}

pub(super) fn cell_key_string(value: &Value, row: usize) -> String {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(format_key_number)
            .unwrap_or_default(),
        Value::StringArray(array) => array.data.get(row).cloned().unwrap_or_default(),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(format_key_number)
                .unwrap_or_default()
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(format_key_number)
                .unwrap_or_default()
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => {
            categorical_label_at(obj, row).unwrap_or_default()
        }
        Value::Cell(cell) => cell
            .get(row, 0)
            .map(|item| cell_to_text(&item))
            .unwrap_or_default(),
        other => format!("{other}"),
    }
}
