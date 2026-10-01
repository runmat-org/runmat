//! Group identity, sample grouping, and box statistics.
use super::plot::{color_for, positions_for, widths_for};
use super::statistics::stats_for_values;
use super::*;

pub(super) fn build_box_stats(
    x: Value,
    group: Option<Value>,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<BoxStats>> {
    let tensor =
        tensor::value_into_tensor_for(NAME, x).map_err(|err| invalid(format!("{NAME}: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid("boxplot: X must be a vector or matrix"));
    }
    let tensor_len = tensor::tensor_element_len(&tensor);
    if tensor_len == 0 {
        return Err(invalid("boxplot: X must not be empty"));
    }
    let mut groups = match group {
        Some(group) => grouped_series(&tensor, group, options)?,
        None => ungrouped_series(&tensor),
    };
    if let Some(color_group) = &options.color_group {
        validate_group_like(&tensor, color_group)?;
    }
    groups.retain(|(_, values)| values.iter().any(|value| !value.is_nan()));
    if groups.is_empty() {
        return Err(invalid(
            "boxplot: data must contain at least one non-NaN value",
        ));
    }
    apply_group_order(&mut groups, options)?;
    if let Some(labels) = &options.labels {
        if labels.len() != groups.len() {
            return Err(invalid(
                "boxplot: Labels length must match the number of plotted boxes",
            ));
        }
        for (idx, (label, _)) in groups.iter_mut().enumerate() {
            *label = labels[idx].clone();
        }
    }
    let positions = positions_for(groups.len(), options)?;
    let widths = widths_for(groups.len(), options);
    groups
        .into_iter()
        .enumerate()
        .map(|(idx, (label, values))| {
            let mut stats = stats_for_values(label, values, options.whisker)?;
            stats.position = positions[idx];
            stats.width = widths[idx];
            stats.color = color_for(idx, &options.box_colors);
            Ok(stats)
        })
        .collect()
}

fn ungrouped_series(tensor: &Tensor) -> Vec<(String, Vec<f64>)> {
    let values = tensor::tensor_values_f64_cow(tensor);
    if tensor.rows == 1 || tensor.cols == 1 {
        return vec![("1".to_string(), values.to_vec())];
    }
    (0..tensor.cols)
        .map(|col| {
            let values = (0..tensor.rows)
                .map(|row| values[col * tensor.rows + row])
                .collect::<Vec<_>>();
            ((col + 1).to_string(), values)
        })
        .collect()
}

fn grouped_series(
    tensor: &Tensor,
    group: Value,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<(String, Vec<f64>)>> {
    let values = tensor::tensor_values_f64_cow(tensor);
    let tensor_len = values.len();
    let labels = match group_labels(&group, tensor_len) {
        Ok(labels) => labels,
        Err(err) if tensor.rows > 1 && tensor.cols > 1 => {
            let row_labels = group_labels(&group, tensor.rows).map_err(|_| err)?;
            let mut labels = Vec::with_capacity(tensor_len);
            for _col in 0..tensor.cols {
                labels.extend(row_labels.iter().cloned());
            }
            labels
        }
        Err(err) => return Err(err),
    };
    if labels.len() != tensor_len {
        return Err(invalid(
            "boxplot: grouping variables must have the same number of elements as X",
        ));
    }
    let mut first_seen = Vec::<String>::new();
    let mut text_groups = HashMap::<String, Vec<f64>>::new();
    let mut floating_groups = BTreeMap::<OrderedF64, (String, Vec<f64>)>::new();
    let mut integer_groups = BTreeMap::<i128, (String, Vec<f64>)>::new();
    let floating_order = labels
        .iter()
        .flatten()
        .all(|label| matches!(label, GroupLabel::Floating(_)));
    let integer_order = labels
        .iter()
        .flatten()
        .all(|label| matches!(label, GroupLabel::Integer(_)));

    for (idx, maybe_label) in labels.into_iter().enumerate() {
        let Some(label) = maybe_label else {
            continue;
        };
        if floating_order {
            let GroupLabel::Floating(value) = label else {
                unreachable!("floating group classification checked above");
            };
            let text = number_label(&value);
            floating_groups
                .entry(OrderedF64(value))
                .or_insert_with(|| (text, Vec::new()))
                .1
                .push(values[idx]);
        } else if integer_order {
            let GroupLabel::Integer(value) = label else {
                unreachable!("integer group classification checked above");
            };
            integer_groups
                .entry(value)
                .or_insert_with(|| (value.to_string(), Vec::new()))
                .1
                .push(values[idx]);
        } else {
            let text = label.text();
            if !text_groups.contains_key(&text) {
                first_seen.push(text.clone());
            }
            text_groups.entry(text).or_default().push(values[idx]);
        }
    }

    let groups = if floating_order {
        floating_groups.into_values().collect::<Vec<_>>()
    } else if integer_order {
        integer_groups.into_values().collect::<Vec<_>>()
    } else {
        first_seen
            .into_iter()
            .filter_map(|label| text_groups.remove(&label).map(|values| (label, values)))
            .collect::<Vec<_>>()
    };

    if groups.is_empty() && options.group_order.is_some() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(groups)
}

fn apply_group_order(
    groups: &mut Vec<(String, Vec<f64>)>,
    options: &BoxplotOptions,
) -> BuiltinResult<()> {
    let Some(order) = &options.group_order else {
        return Ok(());
    };
    let mut by_label = groups.drain(..).collect::<HashMap<_, _>>();
    for label in order {
        if let Some(values) = by_label.remove(label) {
            groups.push((label.clone(), values));
        }
    }
    if groups.is_empty() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum GroupLabel {
    Floating(f64),
    Integer(i128),
    Text(String),
}

impl GroupLabel {
    fn text(&self) -> String {
        match self {
            Self::Floating(value) => number_label(value),
            Self::Integer(value) => value.to_string(),
            Self::Text(value) => value.clone(),
        }
    }
}

pub(super) fn int_group_key(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn numeric_group_label(value: NumericScalar) -> Option<GroupLabel> {
    match value {
        NumericScalar::F64(value) => value.is_finite().then_some(GroupLabel::Floating(value)),
        NumericScalar::F32(value) => value
            .is_finite()
            .then_some(GroupLabel::Floating(f64::from(value))),
        NumericScalar::I8(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I16(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I32(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I64(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U8(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U16(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U32(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U64(value) => Some(GroupLabel::Integer(i128::from(value))),
    }
}

fn group_labels(value: &Value, expected_len: usize) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    match value {
        Value::Cell(cell) => group_labels_from_cell(cell, expected_len),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => (0
            ..expected_len)
            .map(|row| {
                let label = crate::builtins::table::categorical_label_at(object, row)
                    .ok_or_else(|| invalid("boxplot: invalid categorical grouping variable"))?;
                Ok(missing_text_label(&label).then_some(GroupLabel::Text(label)))
            })
            .collect(),
        Value::String(text) if expected_len == 1 => Ok(vec![Some(GroupLabel::Text(text.clone()))]),
        Value::StringArray(array) if array.data.len() == expected_len => Ok(array
            .data
            .iter()
            .map(|text| missing_text_label(text).then(|| GroupLabel::Text(text.clone())))
            .collect()),
        Value::CharArray(array) if array.rows == expected_len => Ok(char_rows(array)
            .into_iter()
            .map(|text| missing_text_label(&text).then_some(GroupLabel::Text(text)))
            .collect()),
        Value::CharArray(array) if array.rows == 1 && expected_len == 1 => {
            let text: String = array.data.iter().collect();
            Ok(vec![
                missing_text_label(&text).then_some(GroupLabel::Text(text))
            ])
        }
        _ => {
            let tensor = tensor::value_into_tensor_for(NAME, value.clone()).map_err(|err| {
                invalid(format!(
                    "boxplot: grouping variable must be numeric, text, cell, or categorical ({err})"
                ))
            })?;
            if tensor::tensor_element_len(&tensor) != expected_len {
                return Err(invalid(
                    "boxplot: grouping variable length must match the number of X elements",
                ));
            }
            Ok((0..expected_len)
                .map(|index| tensor.numeric_value_at(index).and_then(numeric_group_label))
                .collect())
        }
    }
}

fn validate_group_like(tensor: &Tensor, value: &Value) -> BuiltinResult<()> {
    match group_labels(value, tensor::tensor_element_len(tensor)) {
        Ok(_) => Ok(()),
        Err(err) if tensor.rows > 1 && tensor.cols > 1 => group_labels(value, tensor.rows)
            .map(|_| ())
            .map_err(|_| err),
        Err(err) => Err(err),
    }
}

fn group_labels_from_cell(
    cell: &CellArray,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    if cell.data.len() == expected_len {
        let scalar_labels = cell
            .data
            .iter()
            .map(scalar_group_label)
            .collect::<BuiltinResult<Vec<_>>>();
        if let Ok(labels) = scalar_labels {
            return Ok(labels);
        }
    }
    let mut variables = Vec::with_capacity(cell.data.len());
    for value in &cell.data {
        let labels = group_labels(value, expected_len)?;
        if labels.len() != expected_len {
            return Err(invalid(
                "boxplot: grouping variables in a cell array must share the X length",
            ));
        }
        variables.push(labels);
    }
    let mut labels = Vec::with_capacity(expected_len);
    for row in 0..expected_len {
        let mut parts = Vec::with_capacity(variables.len());
        let mut missing = false;
        for variable in &variables {
            match &variable[row] {
                Some(label) => parts.push(label.text()),
                None => {
                    missing = true;
                    break;
                }
            }
        }
        labels.push((!missing).then(|| GroupLabel::Text(parts.join(","))));
    }
    Ok(labels)
}

fn scalar_group_label(value: &Value) -> BuiltinResult<Option<GroupLabel>> {
    match value {
        Value::String(text) => Ok(missing_text_label(text).then(|| GroupLabel::Text(text.clone()))),
        Value::StringArray(array) if array.data.len() == 1 => {
            let text = array.data[0].clone();
            Ok(missing_text_label(&text).then_some(GroupLabel::Text(text)))
        }
        Value::CharArray(array) if array.rows == 1 => {
            let text: String = array.data.iter().collect();
            Ok(missing_text_label(&text).then_some(GroupLabel::Text(text)))
        }
        Value::Num(value) => Ok(value.is_finite().then_some(GroupLabel::Floating(*value))),
        Value::Int(value) => Ok(Some(GroupLabel::Integer(int_group_key(value)))),
        Value::Bool(value) => Ok(Some(GroupLabel::Text(
            if *value { "true" } else { "false" }.to_string(),
        ))),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor.numeric_value_at(0).and_then(numeric_group_label))
        }
        _ => Err(invalid("boxplot: cell grouping labels must be scalar")),
    }
}

fn missing_text_label(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed != "<missing>"
        && trimmed != "<undefined>"
        && !trimmed.eq_ignore_ascii_case("missing")
}

pub(super) fn char_rows(array: &CharArray) -> Vec<String> {
    if array.rows <= 1 {
        return vec![array.data.iter().collect()];
    }
    (0..array.rows)
        .map(|row| {
            (0..array.cols)
                .filter_map(|col| array.data.get(row + col * array.rows).copied())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

pub(super) fn number_label(value: &f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        let text = format!("{value:.15}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub(super) fn numeric_scalar_label(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => number_label(&value),
        NumericScalar::F32(value) => number_label(&f64::from(value)),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}
