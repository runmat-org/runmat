//! Replacement of detected outliers within each numeric slice.
use super::*;

pub(super) fn fill_outlier_tensor(
    input: &Tensor,
    result: &DetectionResult,
    options: &DetectionOptions,
    method: &FillMethod,
) -> BuiltinResult<(Value, Vec<u8>)> {
    let input_values = tensor::tensor_values_f64_cow(input);
    let mut data = input_values.to_vec();
    // Slices are disjoint. The immutable input view supplies the original
    // values for all fill rules without copying the array for every slice.
    let original = input_values.as_ref();
    let mut filled_mask = vec![0; data.len()];
    let shape = tensor::default_shape_for(input.shape(), input_values.len());
    let dim = options.dim.unwrap_or_else(|| first_non_singleton(&shape));
    if dim == 0 {
        fill_slice(
            &mut data,
            original,
            &(0..input_values.len()).collect::<Vec<_>>(),
            result,
            method,
            &mut filled_mask,
        );
    } else {
        let axis = dim - 1;
        if axis >= shape.len() {
            for linear in 0..input_values.len() {
                fill_slice(
                    &mut data,
                    original,
                    &[linear],
                    result,
                    method,
                    &mut filled_mask,
                );
            }
        } else {
            let axis_len = shape[axis];
            let pre: usize = shape[..axis].iter().product();
            let post: usize = shape[axis + 1..].iter().product();
            for prefix in 0..pre {
                for suffix in 0..post {
                    let indices = (0..axis_len)
                        .map(|idx| prefix + idx * pre + suffix * pre * axis_len)
                        .collect::<Vec<_>>();
                    fill_slice(
                        &mut data,
                        original,
                        &indices,
                        result,
                        method,
                        &mut filled_mask,
                    );
                }
            }
        }
    }
    let output_dtype = match input.numeric_dtype() {
        NumericDType::F32 => NumericDType::F32,
        _ => NumericDType::F64,
    };
    Tensor::new_with_dtype(data, input.shape().to_vec(), output_dtype)
        .map(tensor::tensor_into_value)
        .map(|value| (value, filled_mask))
        .map_err(|err| internal_error(FILLOUTLIERS_NAME, format!("filloutliers: {err}")))
}

pub(super) fn fill_slice(
    data: &mut [f64],
    original: &[f64],
    indices: &[usize],
    result: &DetectionResult,
    method: &FillMethod,
    filled_mask: &mut [u8],
) {
    for (pos, linear) in indices.iter().copied().enumerate() {
        if result.mask[linear] == 0 {
            continue;
        }
        let replacement = match method {
            FillMethod::Center => Some(result.center_full[linear]),
            FillMethod::Clip => {
                Some(original[linear].clamp(result.lower_full[linear], result.upper_full[linear]))
            }
            FillMethod::Constant(value) => Some(*value),
            FillMethod::Previous => previous_good(original, result, indices, pos),
            FillMethod::Next => next_good(original, result, indices, pos),
            FillMethod::Nearest => Some(
                nearest_good(original, result, indices, pos).unwrap_or(result.center_full[linear]),
            ),
            FillMethod::Linear => Some(
                linear_interp(original, result, indices, pos).unwrap_or_else(|| {
                    nearest_good(original, result, indices, pos)
                        .unwrap_or(result.center_full[linear])
                }),
            ),
        };
        if let Some(replacement) = replacement {
            data[linear] = replacement;
            filled_mask[linear] = 1;
        } else {
            data[linear] = f64::NAN;
        }
    }
}

fn previous_good(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    indices[..pos]
        .iter()
        .rev()
        .copied()
        .find(|idx| result.mask[*idx] == 0 && original[*idx].is_finite())
        .map(|idx| original[idx])
}

fn next_good(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    indices[pos + 1..]
        .iter()
        .copied()
        .find(|idx| result.mask[*idx] == 0 && original[*idx].is_finite())
        .map(|idx| original[idx])
}

fn nearest_good(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    let prev = indices[..pos]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(p, idx)| (pos - p, original[*idx]));
    let next = indices[pos + 1..]
        .iter()
        .enumerate()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(offset, idx)| (offset + 1, original[*idx]));
    match (prev, next) {
        (Some((pd, _)), Some((nd, nv))) if nd < pd => Some(nv),
        (Some((_, pv)), _) => Some(pv),
        (_, Some((_, nv))) => Some(nv),
        _ => None,
    }
}

fn linear_interp(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    let prev = indices[..pos]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(p, idx)| (p, original[*idx]));
    let next = indices[pos + 1..]
        .iter()
        .enumerate()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(offset, idx)| (pos + 1 + offset, original[*idx]));
    match (prev, next) {
        (Some((p, pv)), Some((n, nv))) if n > p => {
            let w = (pos - p) as f64 / (n - p) as f64;
            Some(pv * (1.0 - w) + nv * w)
        }
        (Some((_, pv)), _) => Some(pv),
        (_, Some((_, nv))) => Some(nv),
        _ => None,
    }
}
