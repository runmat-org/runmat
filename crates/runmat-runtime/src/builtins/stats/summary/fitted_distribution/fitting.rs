use super::*;

pub(super) fn parse_sample(tensor: Tensor, options: FitOptions) -> BuiltinResult<WeightedSample> {
    if tensor
        .shape()
        .iter()
        .copied()
        .filter(|dim| *dim > 1)
        .count()
        > 1
    {
        return Err(invalid("fitdist: data must be a vector"));
    }
    let sample_len = tensor::tensor_element_len(&tensor);
    let frequency = options.frequency.unwrap_or_else(|| vec![1.0; sample_len]);
    if frequency.len() != sample_len {
        return Err(invalid(
            "fitdist: Frequency must contain one value per observation",
        ));
    }
    let mut values = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0;
    let sample_values = tensor::tensor_values_f64(&tensor);
    for (value, weight) in sample_values.into_iter().zip(frequency) {
        if weight.is_nan() || weight < 0.0 || weight.fract() != 0.0 {
            return Err(invalid(
                "fitdist: Frequency values must be nonnegative finite integer counts",
            ));
        }
        if !weight.is_finite() {
            return Err(invalid(
                "fitdist: Frequency values must be nonnegative finite integer counts",
            ));
        }
        if value.is_nan() || weight == 0.0 {
            continue;
        }
        if !value.is_finite() {
            return Err(invalid("fitdist: data must not contain Inf values"));
        }
        values.push(value);
        weights.push(weight);
        total_weight += weight;
    }
    if values.is_empty() || total_weight <= 0.0 {
        return Err(invalid(
            "fitdist: at least one finite observation is required",
        ));
    }
    Ok(WeightedSample {
        values,
        weights,
        total_weight,
    })
}

pub(super) fn fit_distribution(
    kind: DistributionKind,
    sample: &WeightedSample,
) -> BuiltinResult<FittedDistribution> {
    let parameters = match kind {
        DistributionKind::Normal => fit_normal(sample)?,
        DistributionKind::Exponential => fit_exponential(sample)?,
        DistributionKind::Lognormal => fit_lognormal(sample)?,
        DistributionKind::Gamma => fit_gamma(sample)?,
        DistributionKind::Weibull => fit_weibull(sample)?,
        DistributionKind::Poisson => fit_poisson(sample)?,
    };
    let fit = FittedDistribution {
        kind,
        nlogl: sample
            .values
            .iter()
            .zip(sample.weights.iter())
            .map(|(value, weight)| -weight * pdf_scalar_raw(kind, &parameters, *value).ln())
            .sum(),
        parameters,
        observations: sample.total_weight,
    };
    Ok(fit)
}

pub(super) fn fit_normal(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    let mu = weighted_mean(sample);
    let variance = sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * (value - mu).powi(2))
        .sum::<f64>()
        / sample.total_weight;
    if variance < 0.0 {
        return Err(numerical("fitdist: normal variance is invalid"));
    }
    Ok(vec![mu, variance.sqrt()])
}

pub(super) fn fit_exponential(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value >= 0.0, "Exponential")?;
    let mu = weighted_mean(sample);
    if mu <= 0.0 {
        return Err(invalid("fitdist: Exponential data must have positive mean"));
    }
    Ok(vec![mu])
}

pub(super) fn fit_lognormal(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Lognormal")?;
    let logs = transformed_sample(sample, |value| value.ln());
    fit_normal(&logs)
}

pub(super) fn fit_gamma(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Gamma")?;
    let mean = weighted_mean(sample);
    let mean_log = sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * value.ln())
        .sum::<f64>()
        / sample.total_weight;
    let s = mean.ln() - mean_log;
    if s <= 0.0 {
        return Err(numerical(
            "fitdist: Gamma shape is undefined for nearly constant data",
        ));
    }
    let mut shape =
        ((3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s)).max(MIN_POSITIVE);
    for _ in 0..64 {
        let f = shape.ln() - digamma(shape) - s;
        let fp = 1.0 / shape - trigamma(shape);
        let step = f / fp;
        let candidate = shape - step;
        if candidate.is_finite() && candidate > 0.0 {
            shape = candidate;
        } else {
            shape *= 0.5;
        }
        if step.abs() <= 1.0e-12 * shape.max(1.0) {
            break;
        }
    }
    if !shape.is_finite() || shape <= 0.0 {
        return Err(numerical("fitdist: Gamma fit did not converge"));
    }
    Ok(vec![shape, mean / shape])
}

pub(super) fn fit_weibull(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Weibull")?;
    let logs = transformed_sample(sample, |value| value.ln());
    let log_mean = weighted_mean(&logs);
    let log_var = logs
        .values
        .iter()
        .zip(logs.weights.iter())
        .map(|(value, weight)| weight * (value - log_mean).powi(2))
        .sum::<f64>()
        / logs.total_weight;
    let mut shape = (std::f64::consts::PI / (6.0 * log_var.max(1.0e-12)).sqrt()).clamp(0.1, 100.0);
    for _ in 0..80 {
        let (a, b, c) = weibull_sums(sample, shape);
        let g = a / b - log_mean - 1.0 / shape;
        let gp = c / b - (a / b).powi(2) + 1.0 / shape.powi(2);
        let step = g / gp;
        let candidate = shape - step;
        if candidate.is_finite() && candidate > 0.0 {
            shape = candidate;
        } else {
            shape *= 0.5;
        }
        if step.abs() <= 1.0e-11 * shape.max(1.0) {
            break;
        }
    }
    if !shape.is_finite() || shape <= 0.0 {
        return Err(numerical("fitdist: Weibull fit did not converge"));
    }
    let scale = (sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * value.powf(shape))
        .sum::<f64>()
        / sample.total_weight)
        .powf(1.0 / shape);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(numerical("fitdist: Weibull scale is invalid"));
    }
    Ok(vec![scale, shape])
}

pub(super) fn fit_poisson(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(
        sample,
        |value| value >= 0.0 && value.fract() == 0.0,
        "Poisson",
    )?;
    Ok(vec![weighted_mean(sample)])
}

pub(super) fn require_range(
    sample: &WeightedSample,
    pred: impl Fn(f64) -> bool,
    name: &str,
) -> BuiltinResult<()> {
    if sample.values.iter().copied().all(pred) {
        Ok(())
    } else {
        Err(invalid(format!(
            "fitdist: {name} distribution data are outside the supported range"
        )))
    }
}

pub(super) fn transformed_sample(
    sample: &WeightedSample,
    transform: impl Fn(f64) -> f64,
) -> WeightedSample {
    WeightedSample {
        values: sample.values.iter().copied().map(transform).collect(),
        weights: sample.weights.clone(),
        total_weight: sample.total_weight,
    }
}

pub(super) fn weighted_mean(sample: &WeightedSample) -> f64 {
    sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / sample.total_weight
}

pub(super) fn weibull_sums(sample: &WeightedSample, shape: f64) -> (f64, f64, f64) {
    let mut weighted_xk_log = 0.0;
    let mut weighted_xk = 0.0;
    let mut weighted_xk_log2 = 0.0;
    for (value, weight) in sample.values.iter().zip(sample.weights.iter()) {
        let log_value = value.ln();
        let xk = value.powf(shape);
        weighted_xk_log += weight * xk * log_value;
        weighted_xk += weight * xk;
        weighted_xk_log2 += weight * xk * log_value * log_value;
    }
    (weighted_xk_log, weighted_xk, weighted_xk_log2)
}
