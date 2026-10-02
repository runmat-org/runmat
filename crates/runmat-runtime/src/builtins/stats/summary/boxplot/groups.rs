//! Group identity, sample grouping, and box statistics.
use super::plot::color_for;
use super::statistics::stats_for_values_cooperative;
use super::*;
pub(super) mod labels;
#[cfg(test)]
mod tests;
use crate::builtins::common::grouping_keys::{
    group_key_work_units, GroupIndex, GroupIndexBuildError, KeyAtom, KeyOrder,
};
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

#[cfg(test)]
pub(super) fn build_box_stats(
    x: Value,
    group: Option<Value>,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<BoxStats>> {
    build_box_layout(x, group, options).map(|layout| layout.boxes)
}

#[cfg(test)]
pub(super) fn build_box_layout(
    x: Value,
    group: Option<Value>,
    options: &BoxplotOptions,
) -> BuiltinResult<super::factor_layout::BoxLayout> {
    futures::executor::block_on(build_box_layout_cooperative(
        x,
        group,
        options,
        &mut super::cooperation::BoxplotCooperation::new(None),
    ))
}

pub(super) async fn build_box_layout_cooperative(
    x: Value,
    group: Option<Value>,
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<super::factor_layout::BoxLayout> {
    let tensor =
        tensor::value_into_tensor_for(NAME, x).map_err(|err| invalid(format!("{NAME}: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid("boxplot: X must be a vector or matrix"));
    }
    let tensor_len = tensor::tensor_element_len(&tensor);
    if tensor_len == 0 {
        return Err(invalid("boxplot: X must not be empty"));
    }
    let groups = match group {
        Some(group) => grouped_series(&tensor, group, options, cooperation).await?,
        None => ungrouped_series(&tensor, cooperation).await?,
    };
    if let Some(color_group) = &options.color_group {
        validate_group_like(&tensor, color_group)?;
    }
    let mut nonmissing = Vec::new();
    nonmissing
        .try_reserve_exact(groups.len())
        .map_err(|_| invalid("boxplot: groups exceed available memory"))?;
    for group in groups {
        let mut has_observation = false;
        for value in &group.1 {
            has_observation |= !value.is_nan();
            cooperation.completed(1).await?;
        }
        if has_observation {
            nonmissing.push(group);
        }
    }
    let mut groups = nonmissing;
    if groups.is_empty() {
        return Err(invalid(
            "boxplot: data must contain at least one non-NaN value",
        ));
    }
    apply_group_order(&mut groups, options, cooperation).await?;
    let mut boxes = Vec::new();
    boxes
        .try_reserve_exact(groups.len())
        .map_err(|_| invalid("boxplot: statistics exceed available memory"))?;
    for (idx, (label, values)) in groups.into_iter().enumerate() {
        let display = group_label_text(&label);
        cooperation.completed(group_key_work_units(&label)).await?;
        let mut stats =
            stats_for_values_cooperative(display, values, options.whisker, cooperation).await?;
        stats.group_key = label;
        stats.color = color_for(idx, &options.box_colors);
        boxes.push(stats);
    }
    super::factor_layout::prepare_layout_cooperative(boxes, options, cooperation).await
}

async fn ungrouped_series(
    tensor: &Tensor,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<(GroupLabel, Vec<f64>)>> {
    if tensor.rows == 1 || tensor.cols == 1 {
        let mut values = Vec::new();
        values
            .try_reserve_exact(tensor::tensor_element_len(tensor))
            .map_err(|_| invalid("boxplot: data exceeds available memory"))?;
        for index in 0..tensor::tensor_element_len(tensor) {
            values.push(tensor::tensor_value_f64(tensor, index));
            cooperation.completed(1).await?;
        }
        return Ok(vec![(vec![KeyAtom::Text("1".to_string())], values)]);
    }
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(tensor.cols)
        .map_err(|_| invalid("boxplot: groups exceed available memory"))?;
    for col in 0..tensor.cols {
        let mut values = Vec::new();
        values
            .try_reserve_exact(tensor.rows)
            .map_err(|_| invalid("boxplot: data exceeds available memory"))?;
        for row in 0..tensor.rows {
            values.push(tensor::tensor_value_f64(tensor, col * tensor.rows + row));
            cooperation.completed(1).await?;
        }
        groups.push((vec![KeyAtom::Text((col + 1).to_string())], values));
        cooperation.completed(1).await?;
    }
    Ok(groups)
}

async fn grouped_series(
    tensor: &Tensor,
    group: Value,
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<(GroupLabel, Vec<f64>)>> {
    let tensor_len = tensor::tensor_element_len(tensor);
    let labels = match group_labels(&group, tensor_len) {
        Ok(labels) => labels,
        Err(err) if tensor.rows > 1 && tensor.cols > 1 => {
            let row_labels = group_labels(&group, tensor.rows).map_err(|_| err)?;
            let mut labels = Vec::new();
            labels
                .try_reserve_exact(tensor_len)
                .map_err(|_| invalid("boxplot: grouping labels exceed available memory"))?;
            for _col in 0..tensor.cols {
                for label in &row_labels {
                    cooperation
                        .completed(label.as_ref().map_or(1, |key| group_key_work_units(key)))
                        .await?;
                    labels.push(label.clone());
                }
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
    let mut float_order = true;
    let mut integer_order = true;
    for label in labels.iter().flatten() {
        float_order &= matches!(label.as_slice(), [KeyAtom::Number(_)]);
        integer_order &= matches!(label.as_slice(), [KeyAtom::Integer(_)]);
        cooperation.completed(group_key_work_units(label)).await?;
    }
    let order = if float_order || integer_order {
        KeyOrder::Sorted
    } else {
        KeyOrder::FirstAppearance
    };
    let grouped = GroupIndex::build_cooperative(labels, order, cooperation.budget())
        .await
        .map_err(|error| match error {
            GroupIndexBuildError::Invalid(message) => invalid(format!("boxplot: {message}")),
            GroupIndexBuildError::Service(error) => cooperation.service_error(error),
        })?;
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(grouped.keys.len())
        .map_err(|_| invalid("boxplot: groups exceed available memory"))?;
    for (key, rows) in grouped.keys.into_iter().zip(grouped.row_groups) {
        let mut values = Vec::new();
        values
            .try_reserve_exact(rows.len())
            .map_err(|_| invalid("boxplot: group observations exceed available memory"))?;
        for row in rows {
            values.push(tensor::tensor_value_f64(tensor, row));
            cooperation.completed(1).await?;
        }
        groups.push((key, values));
        cooperation.completed(1).await?;
    }

    if groups.is_empty() && options.group_order.is_some() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(groups)
}

async fn apply_group_order(
    groups: &mut Vec<(GroupLabel, Vec<f64>)>,
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<()> {
    let Some(order) = &options.group_order else {
        return Ok(());
    };
    // GroupOrder arrives as public text. Resolve it to one exact typed key here;
    // the presentation label must never become the internal grouping identity.
    let mut by_label = BTreeMap::<String, Vec<usize>>::new();
    for (index, (key, _)) in groups.iter().enumerate() {
        cooperation.completed(group_key_work_units(key)).await?;
        by_label
            .entry(group_label_text(key))
            .or_default()
            .push(index);
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(groups.len())
        .map_err(|_| invalid("boxplot: group order exceeds available memory"))?;
    for group in std::mem::take(groups) {
        entries.push(Some(group));
        cooperation.completed(1).await?;
    }
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
        cooperation.completed(1).await?;
    }
    if groups.is_empty() {
        return Err(invalid("boxplot: GroupOrder did not match any data groups"));
    }
    Ok(())
}
