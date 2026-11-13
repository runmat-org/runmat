//! Host normalization calculations for real and complex data.
use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;
use std::cmp::Ordering;

pub(super) fn evaluate(input: NumericInput, parsed: ParsedArgs) -> BuiltinResult<NormalizeEval> {
    match input {
        NumericInput::Real { data, shape } => evaluate_real(data, shape, parsed),
        NumericInput::Complex { data, shape } => evaluate_complex(data, shape, parsed),
    }
}

fn evaluate_real(
    data: Vec<f64>,
    shape: Vec<usize>,
    parsed: ParsedArgs,
) -> BuiltinResult<NormalizeEval> {
    let axis = parsed
        .dim
        .unwrap_or_else(|| first_non_singleton(&shape))
        .saturating_sub(1);
    let rank = shape.len().max(axis + 1).max(2);
    let mut padded_shape = shape;
    padded_shape.resize(rank, 1);
    let mut param_shape = padded_shape.clone();
    param_shape[axis] = 1;
    let buckets = buckets_for(&padded_shape, axis);
    match parsed.real_plan {
        RealPlan::CenterScale { center, scale } => {
            let center = compute_real_center(&data, &buckets, &param_shape, &center)?;
            let scale = compute_real_scale(&data, &buckets, &param_shape, &scale)?;
            let n_data = normalize_real_center_scale(&data, &padded_shape, &center, &scale)?;
            Ok(NormalizeEval {
                n: tensor_value(n_data, padded_shape)?,
                c: real_param_value(center)?,
                s: real_param_value(scale)?,
            })
        }
        RealPlan::Range { bounds } => {
            let (center_values, scale_values) = compute_range_params(&data, &buckets, bounds);
            let center = ParamReal::Computed {
                data: center_values,
                shape: param_shape.clone(),
            };
            let scale = ParamReal::Computed {
                data: scale_values,
                shape: param_shape,
            };
            let n_data = normalize_real_center_scale(&data, &padded_shape, &center, &scale)?;
            Ok(NormalizeEval {
                n: tensor_value(n_data, padded_shape)?,
                c: real_param_value(center)?,
                s: real_param_value(scale)?,
            })
        }
    }
}

fn evaluate_complex(
    data: Vec<(f64, f64)>,
    shape: Vec<usize>,
    parsed: ParsedArgs,
) -> BuiltinResult<NormalizeEval> {
    let axis = parsed
        .dim
        .unwrap_or_else(|| first_non_singleton(&shape))
        .saturating_sub(1);
    let rank = shape.len().max(axis + 1).max(2);
    let mut padded_shape = shape;
    padded_shape.resize(rank, 1);
    let mut param_shape = padded_shape.clone();
    param_shape[axis] = 1;
    let buckets = buckets_for(&padded_shape, axis);
    let ComplexPlan::CenterScale { center, scale } = parsed.complex_plan;
    let center = compute_complex_center(&data, &buckets, &param_shape, &center)?;
    let scale = compute_complex_scale(&data, &buckets, &param_shape, &scale)?;
    let n_data = normalize_complex_center_scale(&data, &padded_shape, &center, &scale)?;
    Ok(NormalizeEval {
        n: complex_value(n_data, padded_shape)?,
        c: complex_param_value(center)?,
        s: real_param_value(scale)?,
    })
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for idx in 1..shape.len() {
        strides[idx] = strides[idx - 1] * shape[idx - 1];
    }
    strides
}

fn buckets_for(shape: &[usize], axis: usize) -> Vec<Vec<usize>> {
    let mut param_shape = shape.to_vec();
    param_shape[axis] = 1;
    let param_len = tensor::element_count(&param_shape);
    let strides = strides_for(shape);
    let param_strides = strides_for(&param_shape);
    let mut buckets = vec![Vec::new(); param_len];
    for linear in 0..tensor::element_count(shape) {
        let mut dst = 0usize;
        for dim in 0..shape.len() {
            let coord = (linear / strides[dim]) % shape[dim];
            if dim != axis {
                dst += coord * param_strides[dim];
            }
        }
        buckets[dst].push(linear);
    }
    buckets
}

fn non_nan(values: impl Iterator<Item = f64>) -> Vec<f64> {
    values.filter(|value| !value.is_nan()).collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_sample(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return 0.0;
    }
    let mean = mean(values);
    let m2 = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    (m2 / (values.len() - 1) as f64).sqrt()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let n = values.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn quantile_linear(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let frac = pos - lower as f64;
    values[lower] * (1.0 - frac) + values[upper] * frac
}

fn iqr(values: Vec<f64>) -> f64 {
    quantile_linear(values.clone(), 0.75) - quantile_linear(values, 0.25)
}

fn mad(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let med = median(values.to_vec());
    median(values.iter().map(|value| (value - med).abs()).collect())
}

fn p_norm(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if p.is_infinite() {
        return values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    }
    values
        .iter()
        .map(|value| value.abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

fn compute_real_center(
    data: &[f64],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &CenterSpec,
) -> BuiltinResult<ParamReal> {
    match spec {
        CenterSpec::Explicit(tensor) => Ok(ParamReal::Explicit(tensor.clone())),
        CenterSpec::None => Ok(ParamReal::Computed {
            data: vec![0.0; buckets.len()],
            shape: shape.to_vec(),
        }),
        CenterSpec::Mean => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| mean(&non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        CenterSpec::Median => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| median(non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn compute_real_scale(
    data: &[f64],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &ScaleSpec,
) -> BuiltinResult<ParamReal> {
    match spec {
        ScaleSpec::Explicit(tensor) => Ok(ParamReal::Explicit(tensor.clone())),
        ScaleSpec::One => Ok(ParamReal::Computed {
            data: vec![1.0; buckets.len()],
            shape: shape.to_vec(),
        }),
        ScaleSpec::Std => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| std_sample(&non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Mad => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| mad(&non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Iqr => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| iqr(non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::First => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| {
                    indices
                        .iter()
                        .map(|&idx| data[idx])
                        .find(|value| !value.is_nan())
                        .unwrap_or(f64::NAN)
                })
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Norm(p) => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| p_norm(&non_nan(indices.iter().map(|&idx| data[idx])), *p))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn compute_range_params(
    data: &[f64],
    buckets: &[Vec<usize>],
    bounds: RangeBounds,
) -> (Vec<f64>, Vec<f64>) {
    let mut centers = Vec::with_capacity(buckets.len());
    let mut scales = Vec::with_capacity(buckets.len());
    for indices in buckets {
        let values = non_nan(indices.iter().map(|&idx| data[idx]));
        if values.is_empty() {
            centers.push(f64::NAN);
            scales.push(f64::NAN);
            continue;
        }
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if min == max {
            centers.push(f64::NAN);
            scales.push(f64::NAN);
        } else {
            let scale = (max - min) / (bounds.upper - bounds.lower);
            centers.push(min - bounds.lower * scale);
            scales.push(scale);
        }
    }
    (centers, scales)
}

fn complex_mean(values: &[(f64, f64)]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let (sum_re, sum_im) = values
        .iter()
        .fold((0.0, 0.0), |acc, value| (acc.0 + value.0, acc.1 + value.1));
    (sum_re / values.len() as f64, sum_im / values.len() as f64)
}

fn complex_median(mut values: Vec<(f64, f64)>) -> (f64, f64) {
    values.sort_by(|a, b| {
        a.0.hypot(a.1)
            .partial_cmp(&b.0.hypot(b.1))
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                a.1.atan2(a.0)
                    .partial_cmp(&b.1.atan2(b.0))
                    .unwrap_or(Ordering::Equal)
            })
    });
    if values.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        values[values.len() / 2]
    }
}

fn complex_values(data: &[(f64, f64)], indices: &[usize]) -> Vec<(f64, f64)> {
    indices
        .iter()
        .map(|&idx| data[idx])
        .filter(|(re, im)| !re.is_nan() && !im.is_nan())
        .collect()
}

fn complex_std(values: &[(f64, f64)]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return 0.0;
    }
    let mean = complex_mean(values);
    let m2 = values
        .iter()
        .map(|value| {
            let re = value.0 - mean.0;
            let im = value.1 - mean.1;
            re * re + im * im
        })
        .sum::<f64>();
    (m2 / (values.len() - 1) as f64).sqrt()
}

fn complex_mad(values: &[(f64, f64)]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let med = complex_median(values.to_vec());
    median(
        values
            .iter()
            .map(|value| (value.0 - med.0).hypot(value.1 - med.1))
            .collect(),
    )
}

fn complex_iqr(values: &[(f64, f64)]) -> f64 {
    iqr(values.iter().map(|value| value.0.hypot(value.1)).collect())
}

fn complex_norm(values: &[(f64, f64)], p: f64) -> f64 {
    p_norm(
        &values
            .iter()
            .map(|value| value.0.hypot(value.1))
            .collect::<Vec<_>>(),
        p,
    )
}

fn compute_complex_center(
    data: &[(f64, f64)],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &ComplexCenterSpec,
) -> BuiltinResult<ParamComplex> {
    match spec {
        ComplexCenterSpec::ExplicitReal(tensor) => Ok(ParamComplex::ExplicitReal(tensor.clone())),
        ComplexCenterSpec::ExplicitComplex(tensor) => {
            Ok(ParamComplex::ExplicitComplex(tensor.clone()))
        }
        ComplexCenterSpec::None => Ok(ParamComplex::Computed {
            data: vec![(0.0, 0.0); buckets.len()],
            shape: shape.to_vec(),
        }),
        ComplexCenterSpec::Mean => Ok(ParamComplex::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_mean(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ComplexCenterSpec::Median => Ok(ParamComplex::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_median(complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn compute_complex_scale(
    data: &[(f64, f64)],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &ScaleSpec,
) -> BuiltinResult<ParamReal> {
    match spec {
        ScaleSpec::Explicit(tensor) => Ok(ParamReal::Explicit(tensor.clone())),
        ScaleSpec::One => Ok(ParamReal::Computed {
            data: vec![1.0; buckets.len()],
            shape: shape.to_vec(),
        }),
        ScaleSpec::Std => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_std(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Mad => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_mad(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Iqr => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_iqr(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::First => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| {
                    indices
                        .iter()
                        .map(|&idx| data[idx])
                        .find(|(re, im)| !re.is_nan() && !im.is_nan())
                        .map(|value| value.0.hypot(value.1))
                        .unwrap_or(f64::NAN)
                })
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Norm(p) => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_norm(&complex_values(data, indices), *p))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn normalize_real_center_scale(
    data: &[f64],
    shape: &[usize],
    center: &ParamReal,
    scale: &ParamReal,
) -> BuiltinResult<Vec<f64>> {
    let center_values = real_param_values(center);
    let center_shape = real_param_shape(center, center_values.len());
    let scale_values = real_param_values(scale);
    let scale_shape = real_param_shape(scale, scale_values.len());
    let center_plan = BroadcastPlan::new(shape, &center_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if center_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: center parameter is not compatible with input shape",
        ));
    }
    let scale_plan = BroadcastPlan::new(shape, &scale_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if scale_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: scale parameter is not compatible with input shape",
        ));
    }
    Ok(data
        .iter()
        .zip(center_plan.iter().zip(scale_plan.iter()))
        .map(|(value, ((_, _, c_idx), (_, _, s_idx)))| {
            let c = center_values[c_idx];
            let s = scale_values[s_idx];
            normalize_scalar(*value, c, s)
        })
        .collect())
}

fn normalize_complex_center_scale(
    data: &[(f64, f64)],
    shape: &[usize],
    center: &ParamComplex,
    scale: &ParamReal,
) -> BuiltinResult<Vec<(f64, f64)>> {
    let center_values = complex_param_values(center);
    let center_shape = complex_param_shape(center, center_values.len());
    let scale_values = real_param_values(scale);
    let scale_shape = real_param_shape(scale, scale_values.len());
    let center_plan = BroadcastPlan::new(shape, &center_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if center_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: center parameter is not compatible with input shape",
        ));
    }
    let scale_plan = BroadcastPlan::new(shape, &scale_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if scale_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: scale parameter is not compatible with input shape",
        ));
    }
    Ok(data
        .iter()
        .zip(center_plan.iter().zip(scale_plan.iter()))
        .map(|(value, ((_, _, c_idx), (_, _, s_idx)))| {
            let c = center_values[c_idx];
            let s = scale_values[s_idx];
            let re = normalize_scalar(value.0, c.0, s);
            let im = normalize_scalar(value.1, c.1, s);
            (re, im)
        })
        .collect())
}

fn normalize_scalar(value: f64, center: f64, scale: f64) -> f64 {
    if value.is_nan() || scale == 0.0 || scale.is_nan() {
        f64::NAN
    } else {
        (value - center) / scale
    }
}
