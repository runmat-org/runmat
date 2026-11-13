//! Cross-entropy's permissive external format admission over the shared typed axes.

use super::*;

pub(super) fn parse_loss_format(
    format: Option<&str>,
    shape: &[usize],
    label: &'static str,
) -> BuiltinResult<Option<Layout>> {
    let Some(format) = format.filter(|format| !format.is_empty()) else {
        return Ok(None);
    };
    if format.chars().count() != shape.len() {
        return Err(deep_learning_error(
            "crossentropy",
            format!("crossentropy: {label} length must match array rank"),
        ));
    }
    // Preserve public diagnostic precedence while parsing external labels.
    for unique in ['C', 'B', 'T'] {
        if format
            .chars()
            .filter(|candidate| candidate.eq_ignore_ascii_case(&unique))
            .count()
            > 1
        {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {label} cannot repeat '{unique}' labels"),
            ));
        }
    }
    let canonical = format.to_ascii_uppercase();
    Layout::parse(&canonical, shape.len())
        .map(Some)
        .map_err(|_| {
            deep_learning_error(
                "crossentropy",
                format!("crossentropy: {label} contains unsupported dimension labels"),
            )
        })
}

pub(super) fn batch_size(predictions: &LossPayload, format: Option<&Layout>) -> BuiltinResult<f64> {
    Ok(batch_size_for_shape(
        &predictions.shape,
        predictions.format_or_default(format),
    ))
}

pub(super) fn batch_size_for_shape(shape: &[usize], format: Option<&Layout>) -> f64 {
    let Some(format) = format else {
        return shape.iter().copied().product::<usize>() as f64;
    };
    format
        .index(Axis::Batch)
        .map_or(1.0, |batch| shape[batch] as f64)
}

pub(super) fn reduce_mask_included(
    predictions: &LossPayload,
    weighted_losses: &[f64],
    mask: &[f64],
    format: Option<&Layout>,
) -> BuiltinResult<f64> {
    let batch_dim = predictions
        .format_or_default(format)
        .and_then(|layout| layout.index(Axis::Batch));
    let batch_count = batch_dim
        .map(|dim| predictions.shape[dim])
        .unwrap_or(1)
        .max(1);
    let pred_strides = column_major_strides(&predictions.shape);
    let mut included = vec![0usize; batch_count];
    let mut totals = vec![0.0; batch_count];
    for idx in 0..weighted_losses.len() {
        if mask[idx] == 0.0 {
            continue;
        }
        let batch = batch_dim
            .map(|dim| (idx / pred_strides[dim]) % predictions.shape[dim])
            .unwrap_or(0);
        included[batch] += 1;
        totals[batch] += weighted_losses[idx];
    }
    if included.iter().all(|count| *count == 0) {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: Mask must include at least one element",
        ));
    }
    let mut normalized = 0.0;
    for (total, count) in totals.iter().zip(included.iter()) {
        if *count > 0 {
            normalized += total / *count as f64;
        }
    }
    Ok(normalized / batch_count as f64)
}

pub(super) fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(*dim);
    }
    strides
}
