//! Shared outlier statistics and detection along numeric dimensions.
use super::*;
use std::cmp::Ordering;

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn outlier_masks_preserve_exact_scalar_and_vector_shapes() {
    for (shape, values, mask) in [
        (vec![], vec![1.0], vec![0]),
        (vec![3], vec![1.0, 1.0, 9.0], vec![0, 0, 1]),
    ] {
        let input = Tensor::new(values, shape.clone()).unwrap();
        let options = DetectionOptions {
            dim: Some(0),
            ..DetectionOptions::default()
        };
        let result = detect_outliers(&input, &options, "isoutlier").unwrap();
        assert_eq!(result.mask_shape, shape);
        assert_eq!(result.mask, mask);
        let mismatched = DetectionOptions {
            forced_mask: Some((vec![0; input.len()], vec![1, input.len()])),
            ..options
        };
        assert!(detect_outliers(&input, &mismatched, "isoutlier")
            .unwrap_err()
            .contains("exactly the same shape"));
    }
}

pub(super) fn detect_outliers(
    input: &Tensor,
    options: &DetectionOptions,
    builtin: &'static str,
) -> BuiltinResult<DetectionResult> {
    let (input_values, integer_offset) = statistical_values(input, builtin)?;
    if let Some((_, mask_shape)) = &options.forced_mask {
        if mask_shape.as_slice() != input.shape() {
            return Err(invalid_argument(
                builtin,
                "OutlierLocations must have exactly the same shape as the input",
            ));
        }
    }
    let shape = tensor::default_shape_for(input.shape(), input_values.len());
    let dim = options.dim.unwrap_or_else(|| first_non_singleton(&shape));
    let mut result = if dim == 0 {
        detect_all(input, &input_values, options)?
    } else {
        let axis = dim - 1;
        if axis >= shape.len() {
            detect_trailing_singleton(input, &input_values, options)?
        } else if matches!(
            options.method,
            DetectionMethod::MovingMedian(_) | DetectionMethod::MovingMean(_)
        ) {
            detect_moving(input, &input_values, &shape, axis, options)?
        } else {
            detect_by_slice(input, &input_values, &shape, axis, options)?
        }
    };
    if let Some((mask, _)) = &options.forced_mask {
        result.mask.clone_from(mask);
    }
    if integer_offset != 0.0 {
        for values in [
            &mut result.lower,
            &mut result.upper,
            &mut result.center,
            &mut result.lower_full,
            &mut result.upper_full,
            &mut result.center_full,
        ] {
            for value in values {
                *value += integer_offset;
            }
        }
    }
    Ok(result)
}

fn detect_trailing_singleton(
    input: &Tensor,
    input_values: &[f64],
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    // A dimension beyond the stored rank is an implicit singleton. Its
    // trailing ones need no shape allocation, even for a very large dim.
    let mut mask = Vec::with_capacity(input_values.len());
    let mut lower = Vec::with_capacity(input_values.len());
    let mut upper = Vec::with_capacity(input_values.len());
    let mut center = Vec::with_capacity(input_values.len());
    for &value in input_values {
        let stats = slice_stats(&[value], &options.method, threshold_factor(options))?;
        mask.push(u8::from(is_outlier_value(value, stats.lower, stats.upper)));
        lower.push(stats.lower);
        upper.push(stats.upper);
        center.push(stats.center);
    }
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape().to_vec(),
        lower: lower.clone(),
        upper: upper.clone(),
        center: center.clone(),
        lower_full: lower,
        upper_full: upper,
        center_full: center,
        threshold_shape: input.shape().to_vec(),
    })
}

fn statistical_values(input: &Tensor, builtin: &'static str) -> BuiltinResult<(Vec<f64>, f64)> {
    let Some(storage) = input.integer_storage() else {
        return Ok((tensor::tensor_values_f64(input), 0.0));
    };
    macro_rules! signed_offsets {
        ($values:expr) => {{
            let minimum = $values.iter().copied().min().unwrap_or(0) as i128;
            let maximum = $values.iter().copied().max().unwrap_or(0) as i128;
            let span = (maximum - minimum) as u128;
            if span > MAX_EXACT_INTEGER_F64 {
                return Err(invalid_argument(
                    builtin,
                    "typed-integer input span exceeds 2^53 and cannot enter the lossless outlier-statistics input domain",
                ));
            }
            Ok((
                $values
                    .iter()
                    .map(|value| ((*value as i128) - minimum) as f64)
                    .collect(),
                minimum as f64,
            ))
        }};
    }
    macro_rules! unsigned_offsets {
        ($values:expr) => {{
            let minimum = $values.iter().copied().min().unwrap_or(0) as u128;
            let maximum = $values.iter().copied().max().unwrap_or(0) as u128;
            let span = maximum - minimum;
            if span > MAX_EXACT_INTEGER_F64 {
                return Err(invalid_argument(
                    builtin,
                    "typed-integer input span exceeds 2^53 and cannot enter the lossless outlier-statistics input domain",
                ));
            }
            Ok((
                $values
                    .iter()
                    .map(|value| ((*value as u128) - minimum) as f64)
                    .collect(),
                minimum as f64,
            ))
        }};
    }
    match storage {
        IntegerStorage::I8(values) => signed_offsets!(values),
        IntegerStorage::I16(values) => signed_offsets!(values),
        IntegerStorage::I32(values) => signed_offsets!(values),
        IntegerStorage::I64(values) => signed_offsets!(values),
        IntegerStorage::U8(values) => unsigned_offsets!(values),
        IntegerStorage::U16(values) => unsigned_offsets!(values),
        IntegerStorage::U32(values) => unsigned_offsets!(values),
        IntegerStorage::U64(values) => unsigned_offsets!(values),
    }
}

fn detect_all(
    input: &Tensor,
    input_values: &[f64],
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    let stats = slice_stats(input_values, &options.method, threshold_factor(options))?;
    let mask = input_values
        .iter()
        .map(|value| u8::from(is_outlier_value(*value, stats.lower, stats.upper)))
        .collect();
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape().to_vec(),
        lower: vec![stats.lower],
        upper: vec![stats.upper],
        center: vec![stats.center],
        lower_full: vec![stats.lower; input_values.len()],
        upper_full: vec![stats.upper; input_values.len()],
        center_full: vec![stats.center; input_values.len()],
        threshold_shape: vec![1, 1],
    })
}

fn detect_by_slice(
    input: &Tensor,
    input_values: &[f64],
    shape: &[usize],
    axis: usize,
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    let axis_len = shape[axis];
    let pre: usize = shape[..axis].iter().product();
    let post: usize = shape[axis + 1..].iter().product();
    let mut threshold_shape = shape.to_vec();
    threshold_shape[axis] = 1;
    let threshold_len = tensor::element_count(&threshold_shape);
    let mut mask = vec![0u8; input_values.len()];
    let mut lower = vec![f64::NAN; threshold_len];
    let mut upper = vec![f64::NAN; threshold_len];
    let mut center = vec![f64::NAN; threshold_len];
    let mut lower_full = vec![f64::NAN; input_values.len()];
    let mut upper_full = vec![f64::NAN; input_values.len()];
    let mut center_full = vec![f64::NAN; input_values.len()];
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            let mut indices = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let linear = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[linear]);
                indices.push(linear);
            }
            let stats = slice_stats(&slice, &options.method, threshold_factor(options))?;
            let threshold_idx = prefix + suffix * pre;
            lower[threshold_idx] = stats.lower;
            upper[threshold_idx] = stats.upper;
            center[threshold_idx] = stats.center;
            for (value, linear) in slice.iter().zip(indices) {
                mask[linear] = u8::from(is_outlier_value(*value, stats.lower, stats.upper));
                lower_full[linear] = stats.lower;
                upper_full[linear] = stats.upper;
                center_full[linear] = stats.center;
            }
        }
    }
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape().to_vec(),
        lower,
        upper,
        center,
        lower_full,
        upper_full,
        center_full,
        threshold_shape,
    })
}

fn detect_moving(
    input: &Tensor,
    input_values: &[f64],
    shape: &[usize],
    axis: usize,
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    let axis_len = shape[axis];
    let pre: usize = shape[..axis].iter().product();
    let post: usize = shape[axis + 1..].iter().product();
    let mut mask = vec![0u8; input_values.len()];
    let mut lower = vec![f64::NAN; input_values.len()];
    let mut upper = vec![f64::NAN; input_values.len()];
    let mut center = vec![f64::NAN; input_values.len()];
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            let mut indices = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let linear = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[linear]);
                indices.push(linear);
            }
            let window = match options.method {
                DetectionMethod::MovingMedian(window) | DetectionMethod::MovingMean(window) => {
                    window
                }
                _ => unreachable!(),
            };
            for idx in 0..axis_len {
                let (start, end) = centered_window(idx, axis_len, window);
                let stats = slice_stats(
                    &slice[start..end],
                    &options.method,
                    threshold_factor(options),
                )?;
                let linear = indices[idx];
                lower[linear] = stats.lower;
                upper[linear] = stats.upper;
                center[linear] = stats.center;
                mask[linear] = u8::from(is_outlier_value(slice[idx], stats.lower, stats.upper));
            }
        }
    }
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape().to_vec(),
        lower: lower.clone(),
        upper: upper.clone(),
        center: center.clone(),
        lower_full: lower,
        upper_full: upper,
        center_full: center,
        threshold_shape: shape.to_vec(),
    })
}

#[derive(Clone, Copy)]
struct SliceStats {
    lower: f64,
    upper: f64,
    center: f64,
}

fn slice_stats(values: &[f64], method: &DetectionMethod, factor: f64) -> BuiltinResult<SliceStats> {
    let mut clean = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if clean.is_empty() {
        return Ok(SliceStats {
            lower: f64::NAN,
            upper: f64::NAN,
            center: f64::NAN,
        });
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    match method {
        DetectionMethod::Median | DetectionMethod::MovingMedian(_) => {
            let center = median_sorted(&clean);
            let mut deviations = clean
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let spread = MAD_SCALE * median_sorted(&deviations);
            Ok(SliceStats {
                lower: center - factor * spread,
                upper: center + factor * spread,
                center,
            })
        }
        DetectionMethod::Mean | DetectionMethod::MovingMean(_) => {
            let center = clean.iter().sum::<f64>() / clean.len() as f64;
            let spread = sample_std(&clean, center);
            Ok(SliceStats {
                lower: center - factor * spread,
                upper: center + factor * spread,
                center,
            })
        }
        DetectionMethod::Quartiles => {
            let q1 = quantile_sorted(&clean, 0.25);
            let q3 = quantile_sorted(&clean, 0.75);
            let iqr = q3 - q1;
            Ok(SliceStats {
                lower: q1 - factor * iqr,
                upper: q3 + factor * iqr,
                center: median_sorted(&clean),
            })
        }
        DetectionMethod::Percentiles(lo, hi) => Ok(SliceStats {
            lower: quantile_sorted(&clean, *lo),
            upper: quantile_sorted(&clean, *hi),
            center: median_sorted(&clean),
        }),
    }
}

fn is_outlier_value(value: f64, lower: f64, upper: f64) -> bool {
    !value.is_nan() && (value < lower || value > upper)
}

fn threshold_factor(options: &DetectionOptions) -> f64 {
    options.threshold_factor.unwrap_or(match options.method {
        DetectionMethod::Quartiles => 1.5,
        DetectionMethod::Percentiles(_, _) => 1.0,
        _ => 3.0,
    })
}

pub(super) fn centered_window(idx: usize, len: usize, window: usize) -> (usize, usize) {
    let before = window / 2;
    let after = window.saturating_sub(before + 1);
    let start = idx.saturating_sub(before);
    let end = (idx + after + 1).min(len);
    (start, end)
}

fn median_sorted(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn quantile_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let w = pos - lo as f64;
        values[lo] * (1.0 - w) + values[hi] * w
    }
}

fn sample_std(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}
