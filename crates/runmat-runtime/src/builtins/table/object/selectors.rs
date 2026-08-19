use super::*;
use crate::builtins::common::tensor as tensor_utils;
use runmat_value::NumericScalar;

pub(in crate::builtins::table) fn parse_row_selector(
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
        Value::Int(i) => Ok(vec![one_based_integer_to_zero(i, height, "row")?]),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .exact_values()
                    .iter()
                    .map(|value| one_based_integer_to_zero(value, height, "row"))
                    .collect();
            }
            tensor_utils::tensor_values_f64_cow(tensor)
                .iter()
                .map(|value| one_based_to_zero(*value, height, "row"))
                .collect()
        }
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

pub(in crate::builtins::table) fn parse_row_selector_for_object(
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
        if selector_object.is_class("datetime") || selector_object.is_class("duration") {
            return parse_row_time_selector(selector, object);
        }
    }
    if let Some(rows) = parse_row_name_selector(selector, object)? {
        return Ok(rows);
    }
    parse_row_selector(Some(selector), height)
}

fn parse_row_name_selector(
    selector: &Value,
    object: &ObjectInstance,
) -> BuiltinResult<Option<Vec<usize>>> {
    let Ok(selected) = string_list(selector) else {
        return Ok(None);
    };
    let props = table_public_properties(object)?;
    let Some(row_names_value) = props.fields.get(ROW_NAMES) else {
        return Ok(None);
    };
    let row_names = string_list(row_names_value)?;
    if row_names.is_empty() {
        return Ok(None);
    }
    let mut rows = Vec::with_capacity(selected.len());
    for name in selected {
        let idx = row_names
            .iter()
            .position(|row_name| row_name == &name)
            .ok_or_else(|| invalid_index(format!("table: unrecognized row name '{name}'")))?;
        rows.push(idx);
    }
    Ok(Some(rows))
}

fn parse_row_time_selector(selector: &Value, object: &ObjectInstance) -> BuiltinResult<Vec<usize>> {
    if !object.is_class(TIMETABLE_CLASS) {
        return Err(invalid_index(
            "table: datetime or duration row-time selectors require a timetable",
        ));
    }
    let row_times = timetable_row_times(object)?
        .ok_or_else(|| invalid_index("timetable row-time selector requires RowTimes"))?;
    let selector_kind = row_time_kind(selector)?;
    if row_time_kind(&row_times)? != selector_kind {
        return Err(invalid_index(
            "timetable row-time selector type must match timetable RowTimes",
        ));
    }
    let row_values = selector_numeric_values(&row_times)?;
    let selected_values = selector_numeric_values(selector)?;
    let mut rows = Vec::new();
    for selected in selected_values {
        let before = rows.len();
        for (idx, row_time) in row_values.iter().enumerate() {
            if (*row_time - selected).abs() <= 1e-12 {
                rows.push(idx);
            }
        }
        if rows.len() == before {
            return Err(invalid_index(
                "timetable: row-time selector did not match any rows",
            ));
        }
    }
    Ok(rows)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowTimeKind {
    Datetime,
    Duration,
    Numeric,
}

fn row_time_kind(value: &Value) -> BuiltinResult<RowTimeKind> {
    match value {
        Value::Object(obj) if obj.is_class("datetime") => Ok(RowTimeKind::Datetime),
        Value::Object(obj) if obj.is_class("duration") => Ok(RowTimeKind::Duration),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(RowTimeKind::Numeric),
        other => Err(invalid_index(format!(
            "timetable: unsupported row-time selector {other:?}"
        ))),
    }
}

pub(in crate::builtins::table) fn parse_variable_selector(
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
        Value::Int(i) => Ok(vec![name_at_integer_index(names, i)?]),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .exact_values()
                    .iter()
                    .map(|value| name_at_integer_index(names, value))
                    .collect();
            }
            tensor_utils::tensor_values_f64_cow(tensor)
                .iter()
                .map(|value| name_at_index(names, *value))
                .collect()
        }
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

pub(crate) fn parse_variable_selector_for_object(
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

pub(in crate::builtins::table) fn parse_vartype_selector(
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

pub(in crate::builtins::table) fn vartype_matches(value: &Value, kind: &str) -> bool {
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

pub(in crate::builtins::table) fn parse_timerange_selector(
    selector: &ObjectInstance,
    object: &ObjectInstance,
) -> BuiltinResult<Vec<usize>> {
    if !object.is_class(TIMETABLE_CLASS) {
        return Err(invalid_index("timerange selector requires a timetable"));
    }
    let row_times = timetable_row_times(object)?
        .ok_or_else(|| invalid_index("timerange selector requires timetable RowTimes"))?;
    let serials = timerange_numeric_values(&row_times)?;
    let start = selector
        .properties
        .get("Start")
        .map(timerange_bound_value)
        .transpose()?;
    let end = selector
        .properties
        .get("End")
        .map(timerange_bound_value)
        .transpose()?;
    let inclusivity = selector
        .properties
        .get("Inclusivity")
        .map(|value| scalar_text(value, "timerange inclusivity"))
        .transpose()?
        .unwrap_or_else(|| "openright".to_string())
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
                    let order = crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact(*value, start);
                    order.is_some_and(|order| if include_start { order != Ordering::Less } else { order == Ordering::Greater })
                })
                .unwrap_or(true);
            let before_end = end
                .map(|end| {
                    let order = crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact(*value, end);
                    order.is_some_and(|order| if include_end { order != Ordering::Greater } else { order == Ordering::Less })
                })
                .unwrap_or(true);
            (after_start && before_end).then_some(idx)
        })
        .collect())
}

fn timerange_numeric_values(value: &Value) -> BuiltinResult<Vec<NumericScalar>> {
    match value {
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid_argument("timerange: missing numeric row-time value"))
            })
            .collect(),
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        Value::Object(obj) if obj.is_class("datetime") => Ok(tensor_utils::tensor_into_values_f64(
            crate::builtins::datetime::serials_from_datetime_value(value)?,
        )
        .into_iter()
        .map(NumericScalar::F64)
        .collect()),
        Value::Object(obj) if obj.is_class("duration") => Ok(tensor_utils::tensor_into_values_f64(
            crate::builtins::duration::duration_tensor_from_duration_value(value)?,
        )
        .into_iter()
        .map(NumericScalar::F64)
        .collect()),
        other => Err(invalid_argument(format!(
            "timerange: expected numeric, datetime, or duration row times, got {other:?}"
        ))),
    }
}

fn timerange_bound_value(value: &Value) -> BuiltinResult<NumericScalar> {
    let values = timerange_numeric_values(value)?;
    if values.len() != 1 {
        return Err(invalid_argument("timerange: boundary must be scalar"));
    }
    Ok(values[0])
}

pub(in crate::builtins::table) fn parse_rowfilter_selector(
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

pub(in crate::builtins::table) fn logical_array_mask_for_table_rows(
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

pub(in crate::builtins::table) fn evaluate_named_rowfilter(
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
                .map(|value| {
                    numeric_cell_order_to_zero(value, row)
                        .map(|order| order == Some(Ordering::Greater))
                })
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .all(|flag| flag),
            "ge0" | ">=0" | "nonnegative" => selected_values
                .iter()
                .map(|value| {
                    numeric_cell_order_to_zero(value, row)
                        .map(|order| order.is_some_and(|order| order != Ordering::Less))
                })
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .all(|flag| flag),
            "lt0" | "<0" | "negative" => selected_values
                .iter()
                .map(|value| {
                    numeric_cell_order_to_zero(value, row)
                        .map(|order| order == Some(Ordering::Less))
                })
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

pub(in crate::builtins::table) fn selector_numeric_values(
    value: &Value,
) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid_argument("timerange: missing numeric value"))
                    .and_then(selector_scalar_to_f64)
            })
            .collect(),
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![selector_scalar_to_f64(NumericScalar::from(
            value.clone(),
        ))?]),
        Value::Object(obj) if obj.is_class("datetime") => Ok(tensor_utils::tensor_into_values_f64(
            crate::builtins::datetime::serials_from_datetime_value(value)?,
        )),
        Value::Object(obj) if obj.is_class("duration") => Ok(tensor_utils::tensor_into_values_f64(
            crate::builtins::duration::duration_tensor_from_duration_value(value)?,
        )),
        other => Err(invalid_argument(format!(
            "timerange: expected numeric, datetime, or duration row times, got {other:?}"
        ))),
    }
}

fn selector_scalar_to_f64(value: NumericScalar) -> BuiltinResult<f64> {
    match value {
        NumericScalar::F64(value) => Ok(value),
        NumericScalar::F32(value) => Ok(f64::from(value)),
        integer => {
            let integer = integer.into_int_value().expect("integer scalar");
            let exact = match integer {
                runmat_value::IntValue::I8(value) => i128::from(value),
                runmat_value::IntValue::I16(value) => i128::from(value),
                runmat_value::IntValue::I32(value) => i128::from(value),
                runmat_value::IntValue::I64(value) => i128::from(value),
                runmat_value::IntValue::U8(value) => i128::from(value),
                runmat_value::IntValue::U16(value) => i128::from(value),
                runmat_value::IntValue::U32(value) => i128::from(value),
                runmat_value::IntValue::U64(value) => i128::from(value),
            };
            if !(-(1_i128 << 53)..=(1_i128 << 53)).contains(&exact) {
                return Err(invalid_argument(
                    "timerange: integer time values must be exactly representable as double",
                ));
            }
            Ok(exact as f64)
        }
    }
}

pub(in crate::builtins::table) fn numeric_cell_order_to_zero(
    value: &Value,
    row: usize,
) -> BuiltinResult<Option<Ordering>> {
    match row_value(value, row)? {
        Value::Num(value) => Ok(value.partial_cmp(&0.0)),
        Value::Int(value) => Ok(Some(
            crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact(
                NumericScalar::from(value),
                NumericScalar::F64(0.0),
            )
            .expect("integer comparison with zero is ordered"),
        )),
        Value::Bool(value) => Ok(Some(if value {
            Ordering::Greater
        } else {
            Ordering::Equal
        })),
        other => Err(invalid_argument(format!(
            "rowfilter: expected numeric predicate variable, got {other:?}"
        ))),
    }
}

pub(in crate::builtins::table) fn value_is_missing_scalar(value: &Value) -> bool {
    match value {
        Value::Num(value) => value.is_nan(),
        Value::String(text) => text.is_empty() || text == "<missing>",
        Value::StringArray(array) => array
            .data
            .first()
            .map(|text| text.is_empty())
            .unwrap_or(true),
        Value::CharArray(array) => array.data.iter().all(|ch| ch.is_whitespace()),
        Value::Tensor(tensor) => {
            if tensor.integer_storage().is_some() {
                tensor.is_empty()
            } else {
                tensor_utils::tensor_values_f64_cow(tensor)
                    .first()
                    .map(|value| value.is_nan())
                    .unwrap_or(true)
            }
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| {
                    (!tensor.is_empty()).then(|| tensor_utils::tensor_value_f64(&tensor, 0))
                })
                .map(|serial| serial.is_nan())
                .unwrap_or(false)
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .ok()
                .and_then(|tensor| {
                    (!tensor.is_empty()).then(|| tensor_utils::tensor_value_f64(&tensor, 0))
                })
                .map(|days| days.is_nan())
                .unwrap_or(false)
        }
        _ => false,
    }
}

pub(in crate::builtins::table) fn is_colon_selector(value: &Value) -> bool {
    scalar_text(value, "selector")
        .map(|text| text == ":")
        .unwrap_or(false)
}

pub(in crate::builtins::table) fn is_end_selector(value: &Value) -> bool {
    scalar_text(value, "selector")
        .map(|text| text == "end")
        .unwrap_or(false)
}

pub(in crate::builtins::table) fn name_at_index(
    names: &[String],
    value: f64,
) -> BuiltinResult<String> {
    let idx = one_based_to_zero(value, names.len(), "variable")?;
    Ok(names[idx].clone())
}

fn name_at_integer_index(
    names: &[String],
    value: &runmat_value::IntValue,
) -> BuiltinResult<String> {
    let idx = one_based_integer_to_zero(value, names.len(), "variable")?;
    Ok(names[idx].clone())
}

fn one_based_integer_to_zero(
    value: &runmat_value::IntValue,
    len: usize,
    context: &str,
) -> BuiltinResult<usize> {
    let idx = value
        .try_to_usize()
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            invalid_index(format!(
                "table: {context} indices must be positive finite integers"
            ))
        })?;
    if idx >= len {
        return Err(invalid_index(format!(
            "table: {context} index exceeds bounds"
        )));
    }
    Ok(idx)
}

pub(in crate::builtins::table) fn one_based_to_zero(
    value: f64,
    len: usize,
    context: &str,
) -> BuiltinResult<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    fn integer_storages(values: &[u64]) -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(values.iter().map(|&value| value as i8).collect()),
            IntegerStorage::I16(values.iter().map(|&value| value as i16).collect()),
            IntegerStorage::I32(values.iter().map(|&value| value as i32).collect()),
            IntegerStorage::I64(values.iter().map(|&value| value as i64).collect()),
            IntegerStorage::U8(values.iter().map(|&value| value as u8).collect()),
            IntegerStorage::U16(values.iter().map(|&value| value as u16).collect()),
            IntegerStorage::U32(values.iter().map(|&value| value as u32).collect()),
            IntegerStorage::U64(values.to_vec()),
        ]
    }

    #[test]
    fn typed_row_and_variable_selectors_do_not_round_uint64() {
        let rows = parse_row_selector(
            Some(&Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap(),
            )),
            2,
        )
        .unwrap();
        assert_eq!(rows, vec![0, 1]);

        let names = vec!["left".to_string(), "right".to_string()];
        let variables =
            parse_variable_selector(Some(&Value::Int(runmat_value::IntValue::U64(2))), &names)
                .unwrap();
        assert_eq!(variables, vec!["right"]);

        let err = parse_row_selector(Some(&Value::Int(runmat_value::IntValue::U64(u64::MAX))), 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds bounds"), "unexpected error: {err}");
    }

    #[test]
    fn table_index_selectors_read_every_native_integer_class() {
        let names = vec!["left".to_string(), "right".to_string()];
        for storage in integer_storages(&[1, 2]) {
            let selector = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            assert_eq!(
                parse_row_selector(Some(&Value::Tensor(selector.clone())), 2).unwrap(),
                vec![0, 1]
            );
            assert_eq!(
                parse_variable_selector(Some(&Value::Tensor(selector)), &names).unwrap(),
                names
            );
        }
    }

    #[test]
    fn selector_numeric_values_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![7, 9]), vec![1, 2]).unwrap();

        assert_eq!(
            selector_numeric_values(&Value::Tensor(tensor)).unwrap(),
            vec![7.0, 9.0]
        );
    }

    #[test]
    fn value_is_missing_scalar_reads_typed_integer_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();

        assert!(!value_is_missing_scalar(&Value::Tensor(tensor)));

        let empty = Tensor::new_integer(IntegerStorage::I32(Vec::new()), vec![0, 1]).unwrap();
        assert!(value_is_missing_scalar(&Value::Tensor(empty)));
    }
}
