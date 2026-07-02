use super::*;

mod analytics;
mod selectors;

pub use analytics::sortrows_table;
pub(super) use analytics::*;
pub(super) use selectors::*;

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
