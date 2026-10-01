use super::*;

pub(super) fn default_lambda_sequence(
    data: &PreparedData,
    distribution: Distribution,
    options: &Options,
) -> BuiltinResult<Vec<f64>> {
    let intercept = initial_intercept(data, distribution, options.intercept);
    let beta = vec![0.0; data.cols];
    let gradient = gradient_at(data, distribution, &beta, intercept)?;
    let lambda_max =
        gradient.iter().map(|value| value.abs()).fold(0.0, f64::max) / options.alpha.max(EPS);
    if options.num_lambda == 1 {
        return Ok(vec![lambda_max]);
    }
    if lambda_max <= EPS {
        return Ok(vec![0.0]);
    }
    let ratio = options.lambda_ratio;
    if ratio <= EPS {
        let mut values = geometric_sequence(lambda_max, 1.0e-4, options.num_lambda);
        if let Some(last) = values.last_mut() {
            *last = 0.0;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        return Ok(values);
    }
    let mut values = geometric_sequence(lambda_max, ratio, options.num_lambda);
    values.sort_by(|a, b| a.total_cmp(b));
    Ok(values)
}

pub(super) fn sorted_unique_lambdas(lambda: &[f64]) -> BuiltinResult<Vec<f64>> {
    super::super::regularized::sorted_unique_lambdas(lambda, EPS)
        .ok_or_else(|| invalid("lassoglm: Lambda must not be empty"))
}

pub(super) fn fit_path(
    data: &PreparedData,
    distribution: Distribution,
    lambdas_ascending: &[f64],
    options: &Options,
) -> BuiltinResult<FitPath> {
    let mut descending = lambdas_ascending.to_vec();
    descending.sort_by(|a, b| b.total_cmp(a));
    let mut beta_scaled = vec![0.0; data.cols];
    let mut intercept_scaled = initial_intercept(data, distribution, options.intercept);
    if !options.intercept {
        intercept_scaled = 0.0;
    }
    let mut entries = Vec::with_capacity(descending.len());
    for lambda in descending {
        let iterations = fit_single_lambda(
            data,
            distribution,
            &mut beta_scaled,
            &mut intercept_scaled,
            lambda,
            options,
        )?;
        let beta_original = unstandardize_coefficients(data, &beta_scaled);
        let intercept = if options.intercept {
            intercept_scaled
                - beta_original
                    .iter()
                    .zip(data.x_means.iter())
                    .map(|(beta, mean)| beta * mean)
                    .sum::<f64>()
        } else {
            0.0
        };
        let deviance = model_deviance(data, distribution, &beta_original, intercept)?;
        let df = beta_original
            .iter()
            .filter(|value| value.abs() > 1.0e-8)
            .count() as f64;
        entries.push((
            lambda,
            beta_original,
            intercept,
            deviance,
            df,
            iterations as f64,
        ));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(FitPath {
        lambdas: entries.iter().map(|entry| entry.0).collect(),
        coefficients: entries.iter().map(|entry| entry.1.clone()).collect(),
        intercepts: entries.iter().map(|entry| entry.2).collect(),
        deviance: entries.iter().map(|entry| entry.3).collect(),
        df: entries.iter().map(|entry| entry.4).collect(),
        iterations: entries.iter().map(|entry| entry.5).collect(),
    })
}

pub(super) fn fit_single_lambda(
    data: &PreparedData,
    distribution: Distribution,
    beta: &mut [f64],
    intercept: &mut f64,
    lambda: f64,
    options: &Options,
) -> BuiltinResult<usize> {
    let mut objective = objective_value(data, distribution, beta, *intercept, lambda, options)?;
    for iter in 1..=options.max_iter {
        let (grad_beta, grad_intercept, lip) =
            gradient_and_lipschitz(data, distribution, beta, *intercept)?;
        let mut step = 1.0 / lip.max(EPS);
        let old_beta = beta.to_vec();
        let old_intercept = *intercept;
        let mut accepted = false;
        let mut new_objective = objective;
        for _ in 0..30 {
            let candidate_intercept = if options.intercept {
                old_intercept - step * grad_intercept
            } else {
                0.0
            };
            let mut candidate_beta = vec![0.0; beta.len()];
            let ridge = lambda * (1.0 - options.alpha);
            for col in 0..beta.len() {
                let raw = old_beta[col] - step * grad_beta[col];
                candidate_beta[col] =
                    soft_threshold(raw, step * lambda * options.alpha) / (1.0 + step * ridge);
            }
            new_objective = objective_value(
                data,
                distribution,
                &candidate_beta,
                candidate_intercept,
                lambda,
                options,
            )?;
            if new_objective.is_finite() && new_objective <= objective + 1.0e-10 {
                beta.copy_from_slice(&candidate_beta);
                *intercept = candidate_intercept;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            return Err(convergence(
                "lassoglm: proximal-gradient step could not decrease the objective",
            ));
        }
        let max_delta = beta
            .iter()
            .zip(old_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold((*intercept - old_intercept).abs(), f64::max);
        let max_param = beta
            .iter()
            .map(|value| value.abs())
            .fold(intercept.abs(), f64::max)
            .max(1.0);
        objective = new_objective;
        if max_delta <= options.rel_tol * max_param {
            return Ok(iter);
        }
    }
    Ok(options.max_iter)
}
