use super::*;

pub(in crate::builtins::table) fn categorical_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
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

pub(in crate::builtins::table) fn categorical_labels(value: &Value) -> BuiltinResult<Vec<String>> {
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

pub(in crate::builtins::table) fn cell_scalar_label(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::Num(value) => Ok(format_key_number(*value)),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.to_string()),
        other => Ok(other.to_string()),
    }
}

pub(in crate::builtins::table) fn value_shape_or_column(
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::StringArray(array) => Ok(array.shape.clone()),
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::CharArray(array) => Ok(vec![array.rows, 1]),
        _ => Ok(vec![1, 1]),
    }
}
