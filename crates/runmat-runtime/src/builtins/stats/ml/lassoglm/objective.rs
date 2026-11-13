use super::*;

pub(super) fn gradient_at(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
) -> BuiltinResult<Vec<f64>> {
    let (grad, _, _) = gradient_and_lipschitz(data, distribution, beta, intercept)?;
    Ok(grad)
}

pub(super) fn gradient_and_lipschitz(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
) -> BuiltinResult<(Vec<f64>, f64, f64)> {
    let mut grad = vec![0.0; data.cols];
    let mut grad_intercept = 0.0;
    let mut max_weight = 0.0_f64;
    for row in 0..data.rows {
        let eta = linear_predictor(data, beta, intercept, row, true);
        let (_, residual, curvature) = mean_residual_curvature(distribution, data.y[row], eta)?;
        let weight = data.weights[row];
        for (col, grad_value) in grad.iter_mut().enumerate() {
            *grad_value += weight * residual * x_at(data, row, col);
        }
        grad_intercept += weight * residual;
        max_weight = max_weight.max(weight * curvature);
    }
    let norm_bound = (0..data.rows)
        .map(|row| {
            (0..data.cols)
                .map(|col| x_at(data, row, col).abs())
                .sum::<f64>()
                + 1.0
        })
        .map(|row_norm| row_norm * row_norm)
        .fold(1.0, f64::max);
    let lip = max_weight.max(EPS) * norm_bound;
    Ok((grad, grad_intercept, lip))
}

pub(super) fn mean_residual_curvature(
    distribution: Distribution,
    y: f64,
    eta: f64,
) -> BuiltinResult<(f64, f64, f64)> {
    match distribution {
        Distribution::Normal => Ok((eta, eta - y, 1.0)),
        Distribution::Binomial => {
            let p = logistic(eta);
            Ok((p, p - y, (p * (1.0 - p)).max(1.0e-6)))
        }
        Distribution::Poisson => {
            let mu = safe_exp(eta)?;
            Ok((mu, mu - y, mu.max(1.0e-6)))
        }
    }
}

pub(super) fn objective_value(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
    lambda: f64,
    options: &Options,
) -> BuiltinResult<f64> {
    let mut loss = 0.0;
    for row in 0..data.rows {
        let eta = linear_predictor(data, beta, intercept, row, true);
        loss += data.weights[row] * unit_negative_log_likelihood(distribution, data.y[row], eta)?;
    }
    let l1 = beta.iter().map(|value| value.abs()).sum::<f64>();
    let l2 = beta.iter().map(|value| value * value).sum::<f64>();
    Ok(loss + lambda * (options.alpha * l1 + 0.5 * (1.0 - options.alpha) * l2))
}

pub(super) fn unit_negative_log_likelihood(
    distribution: Distribution,
    y: f64,
    eta: f64,
) -> BuiltinResult<f64> {
    match distribution {
        Distribution::Normal => {
            let err = y - eta;
            Ok(0.5 * err * err)
        }
        Distribution::Binomial => Ok(log1pexp(eta) - y * eta),
        Distribution::Poisson => Ok(safe_exp(eta)? - y * eta),
    }
}

pub(super) fn initial_intercept(
    data: &PreparedData,
    distribution: Distribution,
    intercept: bool,
) -> f64 {
    if !intercept {
        return 0.0;
    }
    let mean = data
        .y
        .iter()
        .zip(data.weights.iter())
        .map(|(y, w)| y * w)
        .sum::<f64>();
    match distribution {
        Distribution::Normal => mean,
        Distribution::Binomial => {
            let p = mean.clamp(1.0e-6, 1.0 - 1.0e-6);
            (p / (1.0 - p)).ln()
        }
        Distribution::Poisson => mean.max(1.0e-6).ln(),
    }
}

pub(super) fn model_deviance(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
) -> BuiltinResult<f64> {
    let mut dev = 0.0;
    for row in 0..data.rows {
        let eta = data.offset[row]
            + intercept
            + (0..data.cols)
                .map(|col| data.x_original[row + col * data.rows] * beta[col])
                .sum::<f64>();
        let contribution = deviance_contribution(distribution, data.y[row], eta)?;
        dev += data.weights[row] * contribution;
    }
    Ok(dev)
}

pub(super) fn deviance_contribution(
    distribution: Distribution,
    y: f64,
    eta: f64,
) -> BuiltinResult<f64> {
    match distribution {
        Distribution::Normal => {
            let err = y - eta;
            Ok(err * err)
        }
        Distribution::Binomial => {
            let p = logistic(eta).clamp(EPS, 1.0 - EPS);
            let y = y.clamp(0.0, 1.0);
            Ok(binomial_deviance(y, p))
        }
        Distribution::Poisson => {
            let mu = safe_exp(eta)?.max(EPS);
            Ok(2.0 * (poisson_term(y, y.max(EPS)) - poisson_term(y, mu)))
        }
    }
}

pub(super) fn binomial_deviance(y: f64, p: f64) -> f64 {
    if y <= EPS {
        2.0 * (1.0 / (1.0 - p).max(EPS)).ln()
    } else if y >= 1.0 - EPS {
        2.0 * (1.0 / p.max(EPS)).ln()
    } else {
        2.0 * (y * (y / p.max(EPS)).ln() + (1.0 - y) * ((1.0 - y) / (1.0 - p).max(EPS)).ln())
    }
}

pub(super) fn poisson_term(y: f64, mu: f64) -> f64 {
    if y <= EPS {
        -y + mu
    } else {
        y * (y / mu).ln() - y + mu
    }
}

pub(super) fn linear_predictor(
    data: &PreparedData,
    beta: &[f64],
    intercept: f64,
    row: usize,
    standardized: bool,
) -> f64 {
    let base = data.offset[row] + intercept;
    if standardized {
        base + (0..data.cols)
            .map(|col| x_at(data, row, col) * beta[col])
            .sum::<f64>()
    } else {
        base + (0..data.cols)
            .map(|col| data.x_original[row + col * data.rows] * beta[col])
            .sum::<f64>()
    }
}

pub(super) fn x_at(data: &PreparedData, row: usize, col: usize) -> f64 {
    data.x_work[row + col * data.rows]
}

pub(super) fn unstandardize_coefficients(data: &PreparedData, beta_scaled: &[f64]) -> Vec<f64> {
    beta_scaled
        .iter()
        .zip(data.x_scales.iter())
        .map(|(beta, scale)| beta / scale)
        .collect()
}

pub(super) fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

pub(super) fn safe_exp(value: f64) -> BuiltinResult<f64> {
    if value > 700.0 {
        return Err(convergence("lassoglm: exponential mean overflowed"));
    }
    Ok(value.exp())
}

pub(super) fn log1pexp(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}
