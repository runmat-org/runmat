use super::*;

pub(super) async fn gather_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

#[derive(Default)]
pub(super) struct TableConstructorOptions {
    pub(super) variable_names: Option<Vec<String>>,
    pub(super) row_names: Option<Vec<String>>,
}

pub(super) struct Struct2TableOptions {
    pub(super) table: TableConstructorOptions,
    pub(super) as_array: bool,
}

pub(super) fn split_table_constructor_args(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, TableConstructorOptions)> {
    let mut variables = Vec::new();
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if let Ok(name) = scalar_text(&args[idx], "table option") {
            if idx + 1 < args.len() && is_table_constructor_option(&name) {
                let value = &args[idx + 1];
                if name.eq_ignore_ascii_case("VariableNames") {
                    options.variable_names = Some(variable_name_list(value)?);
                } else if name.eq_ignore_ascii_case("RowNames") {
                    options.row_names = Some(string_list(value)?);
                }
                idx += 2;
                continue;
            }
        }
        variables.push(args[idx].clone());
        idx += 1;
    }
    Ok((variables, options))
}

pub(super) fn is_table_constructor_option(name: &str) -> bool {
    name.eq_ignore_ascii_case("VariableNames") || name.eq_ignore_ascii_case("RowNames")
}

pub(super) fn parse_table_options(
    args: &[Value],
    context: &str,
) -> BuiltinResult<TableConstructorOptions> {
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let name = scalar_text(&args[idx], "table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            options.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            options.row_names = Some(string_list(&args[idx + 1])?);
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(options)
}

pub(super) fn parse_struct2table_options(args: &[Value]) -> BuiltinResult<Struct2TableOptions> {
    let mut table = TableConstructorOptions::default();
    let mut as_array = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "struct2table: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "struct2table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            table.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            table.row_names = Some(string_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("AsArray") {
            as_array = bool_scalar(&args[idx + 1], "AsArray")?;
        } else {
            return Err(invalid_argument(format!(
                "struct2table: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(Struct2TableOptions { table, as_array })
}

pub(super) fn parse_table2struct_to_scalar(args: &[Value]) -> BuiltinResult<bool> {
    let mut to_scalar = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "table2struct: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "table2struct option")?;
        if name.eq_ignore_ascii_case("ToScalar") {
            to_scalar = bool_scalar(&args[idx + 1], "ToScalar")?;
        } else {
            return Err(invalid_argument(format!(
                "table2struct: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(to_scalar)
}

pub(super) fn split_readtimetable_options(
    args: &[Value],
) -> BuiltinResult<(Vec<Value>, Vec<Value>)> {
    let mut readtable_args = Vec::new();
    let mut timetable_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "readtimetable: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "readtimetable option")?;
        if name.eq_ignore_ascii_case(ROW_TIMES) {
            timetable_args.push(args[idx].clone());
            timetable_args.push(args[idx + 1].clone());
        } else {
            readtable_args.push(args[idx].clone());
            readtable_args.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok((readtable_args, timetable_args))
}

pub(super) fn parse_named_option<'a>(args: &'a [Value], name: &str) -> Option<&'a Value> {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if scalar_text(&args[idx], "option name")
            .map(|text| text.eq_ignore_ascii_case(name))
            .unwrap_or(false)
        {
            return args.get(idx + 1);
        }
        idx += 2;
    }
    None
}

pub(super) fn parse_bool_option(
    args: &[Value],
    name: &str,
    default: bool,
    context: &str,
) -> BuiltinResult<bool> {
    let mut result = default;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if option_name.eq_ignore_ascii_case(name) {
            result = bool_scalar(&args[idx + 1], name)?;
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(result)
}

pub(super) fn parse_named_text_option(
    args: &[Value],
    name: &str,
    default: &str,
    context: &str,
) -> BuiltinResult<String> {
    let mut result = default.to_string();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if option_name.eq_ignore_ascii_case(name) {
            result = scalar_text(&args[idx + 1], name)?;
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(result)
}

pub(super) fn split_value_columns(value: Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(tensor.get2(row, col).map_err(invalid_index)?);
                }
                out.push(Value::Tensor(
                    Tensor::new_with_dtype(data, vec![rows, 1], tensor.dtype)
                        .map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::ComplexTensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.cols);
            for col in 0..tensor.cols {
                let mut data = Vec::with_capacity(tensor.rows);
                for row in 0..tensor.rows {
                    data.push(tensor.data[row + col * tensor.rows]);
                }
                out.push(Value::ComplexTensor(
                    ComplexTensor::new(data, vec![tensor.rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::StringArray(array) => {
            let rows = array.rows();
            let cols = array.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(array.data[row + col * rows].clone());
                }
                out.push(Value::StringArray(
                    StringArray::new(data, vec![rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::LogicalArray(array) => {
            let rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(*array.data.get(row + col * rows).ok_or_else(|| {
                        invalid_variable("array2table: logical array shape mismatch")
                    })?);
                }
                out.push(Value::LogicalArray(
                    LogicalArray::new(data, vec![rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.cols);
            for col in 0..cell.cols {
                let mut data = Vec::with_capacity(cell.rows);
                for row in 0..cell.rows {
                    data.push(cell.get(row, col).map_err(invalid_index)?);
                }
                out.push(Value::Cell(
                    CellArray::new(data, cell.rows, 1).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        other => Ok(vec![other]),
    }
}

pub(super) fn table_to_cell_array(object: &ObjectInstance) -> BuiltinResult<Value> {
    let height = table_height(object)?;
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let mut data = Vec::with_capacity(height * names.len());
    for col_name in &names {
        let value = variables.fields.get(col_name).ok_or_else(|| {
            invalid_variable(format!("table2cell: missing variable '{col_name}'"))
        })?;
        for row in 0..height {
            data.push(row_value(value, row)?);
        }
    }
    CellArray::new(data, height, names.len())
        .map(Value::Cell)
        .map_err(invalid_variable)
}

pub(super) fn row_value(value: &Value, row: usize) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => {
            Ok(Value::Num(tensor.get2(row, 0).map_err(invalid_index)?))
        }
        Value::Tensor(_) => select_rows(value, &[row]),
        Value::ComplexTensor(tensor) if tensor.cols == 1 => {
            let value = tensor
                .data
                .get(row)
                .copied()
                .ok_or_else(|| invalid_index("table2cell: complex row out of bounds"))?;
            Ok(Value::Complex(value.0, value.1))
        }
        Value::ComplexTensor(_) => select_rows(value, &[row]),
        Value::StringArray(array) if array.cols() == 1 => Ok(Value::String(
            array.data.get(row).cloned().unwrap_or_default(),
        )),
        Value::StringArray(_) => select_rows(value, &[row]),
        Value::LogicalArray(array) if array.shape.get(1).copied().unwrap_or(1) == 1 => {
            Ok(Value::Bool(*array.data.get(row).unwrap_or(&0) != 0))
        }
        Value::LogicalArray(_) => select_rows(value, &[row]),
        Value::Cell(cell) if cell.cols == 1 => cell.get(row, 0).map_err(invalid_index),
        Value::Cell(_) => select_rows(value, &[row]),
        Value::CharArray(array) => {
            if row >= array.rows {
                return Err(invalid_index("table2cell: char row out of bounds"));
            }
            let start = row * array.cols;
            CharArray::new(
                array.data[start..start + array.cols].to_vec(),
                1,
                array.cols,
            )
            .map(Value::CharArray)
            .map_err(invalid_variable)
        }
        Value::Object(obj) if obj.is_class("datetime") || obj.is_class("duration") => {
            select_rows(value, &[row])
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => select_rows(value, &[row]),
        other if row == 0 => Ok(other.clone()),
        _ => Err(invalid_index("table2cell: row out of bounds")),
    }
}

pub(super) fn colon_colon_payload() -> Value {
    Value::Cell(
        CellArray::new(vec![Value::from(":"), Value::from(":")], 1, 2)
            .expect("selector cell shape is valid"),
    )
}

pub(super) fn split_timetable_constructor_args(
    args: Vec<Value>,
) -> BuiltinResult<(Option<Value>, Vec<Value>, TableConstructorOptions)> {
    if args.is_empty() {
        return Ok((None, Vec::new(), TableConstructorOptions::default()));
    }
    let mut variables = Vec::new();
    let mut row_times = None;
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    let has_explicit_row_times = args.iter().any(|value| {
        scalar_text(value, "timetable option")
            .map(|name| name.eq_ignore_ascii_case("RowTimes"))
            .unwrap_or(false)
    });
    if !has_explicit_row_times && args.len() > 1 && !is_timetable_option_token(&args[0]) {
        row_times = Some(args[0].clone());
        idx = 1;
    }
    while idx < args.len() {
        if idx + 1 < args.len() {
            if let Ok(name) = scalar_text(&args[idx], "timetable option") {
                if name.eq_ignore_ascii_case("RowTimes") {
                    row_times = Some(args[idx + 1].clone());
                    idx += 2;
                    continue;
                }
                if name.eq_ignore_ascii_case("VariableNames") {
                    options.variable_names = Some(variable_name_list(&args[idx + 1])?);
                    idx += 2;
                    continue;
                }
                if name.eq_ignore_ascii_case("RowNames") {
                    options.row_names = Some(string_list(&args[idx + 1])?);
                    idx += 2;
                    continue;
                }
            }
        }
        variables.push(args[idx].clone());
        idx += 1;
    }
    Ok((row_times, variables, options))
}

pub(super) fn is_timetable_option_token(value: &Value) -> bool {
    scalar_text(value, "timetable option")
        .map(|name| {
            name.eq_ignore_ascii_case("RowTimes")
                || name.eq_ignore_ascii_case("VariableNames")
                || name.eq_ignore_ascii_case("RowNames")
        })
        .unwrap_or(false)
}

pub(super) fn is_time_like_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Object(obj) if obj.is_class("datetime") || obj.is_class("duration")
    )
}

pub(super) fn parse_timetable_options(
    args: &[Value],
    context: &str,
) -> BuiltinResult<(Option<Value>, TableConstructorOptions)> {
    let mut row_times = None;
    let mut table_options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let name = scalar_text(&args[idx], "timetable option")?;
        if name.eq_ignore_ascii_case("RowTimes") {
            row_times = Some(args[idx + 1].clone());
        } else if name.eq_ignore_ascii_case("VariableNames") {
            table_options.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            table_options.row_names = Some(string_list(&args[idx + 1])?);
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok((row_times, table_options))
}

pub(super) fn set_timetable_row_times(
    object: &mut ObjectInstance,
    row_times: Option<Value>,
) -> BuiltinResult<()> {
    let height = table_height(object)?;
    let times = row_times.unwrap_or_else(|| {
        Value::Tensor(
            Tensor::new(
                (1..=height).map(|idx| idx as f64).collect(),
                vec![height, 1],
            )
            .expect("generated row-time tensor shape is valid"),
        )
    });
    let rows = value_row_count(&times)?;
    if rows != height {
        return Err(invalid_variable(format!(
            "timetable: RowTimes has {rows} rows but timetable has {height}"
        )));
    }
    let mut props = table_public_properties(object)?;
    props.insert(ROW_TIMES, times);
    props.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(
                vec!["Time".to_string(), DEFAULT_VARIABLE_DIM_NAME.to_string()],
                vec![1, 2],
            )
            .map_err(invalid_variable)?,
        ),
    );
    sync_table_properties(object, props);
    Ok(())
}

pub(super) fn timetable_row_times(object: &ObjectInstance) -> BuiltinResult<Option<Value>> {
    if !object.is_class(TIMETABLE_CLASS) {
        return Ok(None);
    }
    let props = table_public_properties(object)?;
    Ok(props.fields.get(ROW_TIMES).cloned())
}

pub(super) fn categorical_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    let source = args
        .first()
        .cloned()
        .unwrap_or_else(|| Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()));
    let mut category_seed = None;
    let mut ordinal_args_start = 1usize;
    if args.len() > 1
        && !scalar_text(&args[1], "categorical option")
            .map(|name| name.eq_ignore_ascii_case("Ordinal"))
            .unwrap_or(false)
    {
        category_seed = Some(categorical_labels(&args[1])?);
        ordinal_args_start = 2;
    }
    let ordinal = parse_bool_option(&args[ordinal_args_start..], "Ordinal", false, "categorical")?;
    let labels = categorical_labels(&source)?;
    let has_explicit_categories = category_seed.is_some();
    let mut categories = category_seed.unwrap_or_default();
    let mut codes = Vec::with_capacity(labels.len());
    for label in labels {
        if label.is_empty() {
            codes.push(f64::NAN);
            continue;
        }
        let idx = if let Some(idx) = categories.iter().position(|existing| existing == &label) {
            idx
        } else {
            if has_explicit_categories {
                codes.push(f64::NAN);
                continue;
            }
            categories.push(label);
            categories.len() - 1
        };
        codes.push((idx + 1) as f64);
    }
    let mut object = ObjectInstance::new(CATEGORICAL_CLASS.to_string());
    object.properties.insert(
        "Codes".to_string(),
        Value::Tensor(
            Tensor::new(codes, value_shape_or_column(&source)?).map_err(invalid_variable)?,
        ),
    );
    object.properties.insert(
        "Categories".to_string(),
        Value::StringArray(
            StringArray::new(categories.clone(), vec![1, categories.len()])
                .map_err(invalid_variable)?,
        ),
    );
    object
        .properties
        .insert("Ordinal".to_string(), Value::Bool(ordinal));
    Ok(Value::Object(object))
}

pub(super) fn categorical_labels(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) => Ok(char_rows(array)),
        Value::Tensor(tensor) => Ok(tensor
            .data
            .iter()
            .map(|value| format_key_number(*value))
            .collect()),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| if *flag != 0 { "true" } else { "false" }.to_string())
            .collect()),
        Value::Cell(cell) => cell.data.iter().map(cell_scalar_label).collect(),
        other => Ok(vec![other.to_string()]),
    }
}

pub(super) fn cell_scalar_label(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::Num(value) => Ok(format_key_number(*value)),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.to_string()),
        other => Ok(other.to_string()),
    }
}

pub(super) fn value_shape_or_column(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::StringArray(array) => Ok(array.shape.clone()),
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::CharArray(array) => Ok(vec![array.rows, 1]),
        _ => Ok(vec![1, 1]),
    }
}

pub(super) fn dictionary_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
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

pub(super) fn into_dictionary_object(value: Value, context: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(DICTIONARY_CLASS) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected dictionary, got {other:?}"
        ))),
    }
}

pub(super) fn dictionary_cells<'a>(
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

pub(super) fn dictionary_lookup(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
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

pub(super) fn dictionary_assign(
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

pub(super) fn dictionary_keys_equal(left: &Value, right: &Value) -> bool {
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

pub(super) fn value_elements(value: &Value) -> BuiltinResult<Vec<Value>> {
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
