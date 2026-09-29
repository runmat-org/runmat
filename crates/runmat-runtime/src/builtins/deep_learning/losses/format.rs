use super::*;

pub(super) fn batch_size(predictions: &LossPayload, format: Option<&str>) -> BuiltinResult<f64> {
    let Some(format) = predictions.format_or_default(format) else {
        return Ok(predictions.data.len() as f64);
    };
    let Some(batch_dim) = format
        .chars()
        .position(|label| label.eq_ignore_ascii_case(&'B'))
    else {
        return Ok(1.0);
    };
    Ok(predictions.shape.get(batch_dim).copied().unwrap_or(1) as f64)
}

pub(super) fn batch_size_for_shape(shape: &[usize], format: Option<&str>) -> f64 {
    let Some(format) = format else {
        return shape.iter().copied().product::<usize>() as f64;
    };
    let Some(batch_dim) = format
        .chars()
        .position(|label| label.eq_ignore_ascii_case(&'B'))
    else {
        return 1.0;
    };
    shape.get(batch_dim).copied().unwrap_or(1) as f64
}

pub(super) fn reduce_mask_included(
    predictions: &LossPayload,
    weighted_losses: &[f64],
    mask: &[f64],
    format: Option<&str>,
) -> BuiltinResult<f64> {
    let batch_dim = label_index(predictions, format, 'B');
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

fn label_index(predictions: &LossPayload, format: Option<&str>, label: char) -> Option<usize> {
    predictions.format_or_default(format).and_then(|format| {
        format
            .chars()
            .position(|candidate| candidate.eq_ignore_ascii_case(&label))
    })
}

pub(super) fn label_index_for_shape(format: Option<&str>, label: char) -> Option<usize> {
    format.and_then(|format| {
        format
            .chars()
            .position(|candidate| candidate.eq_ignore_ascii_case(&label))
    })
}

pub(super) fn validate_format_for_shape(
    format: Option<&str>,
    shape: &[usize],
    label: &'static str,
) -> BuiltinResult<()> {
    let Some(format) = format else {
        return Ok(());
    };
    if format.chars().count() != shape.len() {
        return Err(deep_learning_error(
            "crossentropy",
            format!("crossentropy: {label} length must match array rank"),
        ));
    }
    for required_unique in ['C', 'B', 'T'] {
        if format
            .chars()
            .filter(|candidate| candidate.eq_ignore_ascii_case(&required_unique))
            .count()
            > 1
        {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {label} cannot repeat '{required_unique}' labels"),
            ));
        }
    }
    if format.chars().any(|candidate| {
        !"SCBTU"
            .chars()
            .any(|allowed| allowed.eq_ignore_ascii_case(&candidate))
    }) {
        return Err(deep_learning_error(
            "crossentropy",
            format!("crossentropy: {label} contains unsupported dimension labels"),
        ));
    }
    Ok(())
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
