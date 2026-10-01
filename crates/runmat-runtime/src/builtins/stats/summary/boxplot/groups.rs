//! Group identity, sample grouping, and box statistics.
use super::plot::{color_for, positions_for, widths_for};
use super::statistics::stats_for_values;
use super::*;
pub(super) mod labels;
use crate::builtins::common::grouping_keys::{GroupIndex, KeyAtom, KeyOrder};
pub(super) use labels::{char_rows, number_label, numeric_scalar_label};
use labels::{group_label_text, group_labels, validate_group_like, GroupLabel};

pub(super) fn validate_known_host_group(
    group: &Value,
    x_shape: Option<&[usize]>,
    x_count: usize,
) -> BuiltinResult<()> {
    match group_labels(group, x_count) {
        Ok(_) => Ok(()),
        Err(first_error)
            if x_shape.is_some_and(|shape| shape.len() == 2 && shape[0] > 1 && shape[1] > 1) =>
        {
            let rows = x_shape.expect("matrix shape checked above")[0];
            group_labels(group, rows)
                .map(|_| ())
                .map_err(|_| first_error)
        }
        Err(error) => Err(error),
    }
}

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
    }
    let positions = positions_for(groups.len(), options)?;
    let widths = widths_for(groups.len(), options);
    groups
        .into_iter()
        .enumerate()
        .map(|(idx, (label, values))| {
            let display = options
                .labels
                .as_ref()
                .map_or_else(|| group_label_text(&label), |labels| labels[idx].clone());
            let mut stats = stats_for_values(display, values, options.whisker)?;
            stats.position = positions[idx];
            stats.width = widths[idx];
            stats.color = color_for(idx, &options.box_colors);
            Ok(stats)
        })
        .collect()
}

fn ungrouped_series(tensor: &Tensor) -> Vec<(GroupLabel, Vec<f64>)> {
    let values = tensor::tensor_values_f64_cow(tensor);
    if tensor.rows == 1 || tensor.cols == 1 {
        return vec![(vec![KeyAtom::Text("1".to_string())], values.to_vec())];
    }
    (0..tensor.cols)
        .map(|col| {
            let values = (0..tensor.rows)
                .map(|row| values[col * tensor.rows + row])
                .collect::<Vec<_>>();
            (vec![KeyAtom::Text((col + 1).to_string())], values)
        })
        .collect()
}

fn grouped_series(
    tensor: &Tensor,
    group: Value,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<(GroupLabel, Vec<f64>)>> {
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
    let float_order = labels
        .iter()
        .flatten()
        .all(|label| matches!(label.as_slice(), [KeyAtom::Number(_)]));
    let integer_order = labels
        .iter()
        .flatten()
        .all(|label| matches!(label.as_slice(), [KeyAtom::Integer(_)]));
    let order = if float_order || integer_order {
        KeyOrder::Sorted
    } else {
        KeyOrder::FirstAppearance
    };
    let grouped =
        GroupIndex::build(labels, order).map_err(|err| invalid(format!("boxplot: {err}")))?;
    let groups = grouped
        .keys
        .into_iter()
        .zip(grouped.row_groups)
        .map(|(key, rows)| (key, rows.into_iter().map(|row| values[row]).collect()))
        .collect::<Vec<_>>();

    if groups.is_empty() && options.group_order.is_some() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(groups)
}

fn apply_group_order(
    groups: &mut Vec<(GroupLabel, Vec<f64>)>,
    options: &BoxplotOptions,
) -> BuiltinResult<()> {
    let Some(order) = &options.group_order else {
        return Ok(());
    };
    // GroupOrder arrives as public text. Resolve it to one exact typed key here;
    // the presentation label must never become the internal grouping identity.
    let mut by_label = BTreeMap::<String, Vec<usize>>::new();
    for (index, (key, _)) in groups.iter().enumerate() {
        by_label
            .entry(group_label_text(key))
            .or_default()
            .push(index);
    }
    let mut entries = std::mem::take(groups)
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    for label in order {
        if let Some(indices) = by_label.get(label) {
            if indices.len() != 1 {
                return Err(invalid(
                    "boxplot: GroupOrder label is ambiguous across distinct typed groups",
                ));
            }
            if let Some(entry) = entries[indices[0]].take() {
                groups.push(entry);
            }
        }
    }
    if groups.is_empty() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(())
}
